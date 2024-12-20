import ufl
import dolfin as dlf
import dolfin_adjoint as d_ad
import pyadjoint as pyad

from mesh_utils import convert_dof_array_to_function
from scipy import sparse

# Turn off annoying log messages
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BuildDolfinProblem(object):
    """ Generic dolfin problem builder. Generates predictions and reduced functionals for calculating dJ/dF """
    def __init__(self, mesh):
        self.mesh = mesh

    def vertex_to_dof_order(self, target):
        """ Assume the target is a [2, N] ndarray representing mesh coordinates
            Convert the target to a flattened [N, 2] -> [N x 2] array representing
            Fenics dofs
            dof_to_vertex_map gives the corresponding vertex for each ordered dof
            Important to transpose because fenics does [N, C] but torch does [C, N]
        """
        d2v = dlf.dof_to_vertex_map(self.function_space)
        output = target.T.flatten()[d2v]
        return output

    def lhs(self, trial_function, test_function):
        raise NotImplementedError
    
    def rhs(self, trial_function, test_function):
        raise NotImplementedError

    def forward(self, force):
        """ Forward problem solved by dolfin """
        trial = dlf.TrialFunction(self.function_space)
        test = dlf.TestFunction(self.function_space)

        a = self.lhs(trial, test)
        L = self.rhs(force, test)

        # Assemble and solve the problem
        pred = d_ad.Function(self.function_space)
        d_ad.solve(a == L, pred, self.bc)
        return pred
    
    def reduced_functional(self, target):
        """ Reduced loss functional using dolfin adjoint """

        # Assume the target is a [2, N] ndarray representing mesh coordinates
        # This is what we typically do on dataset creation

        # dof_to_vertex_map gives the corresponding vertex for each ordered dof
        d2v = dlf.dof_to_vertex_map(self.function_space)
        # Important to transpose because fenics does [N, C] but torch does [C, N]
        output = target.T.flatten()[d2v]
        output = convert_dof_array_to_function(output, self.function_space)

        # Build problem functions
        force = d_ad.Function(self.function_space)
        force.vector()[:] = 0.

        # Call forward problem to get prediction
        pred = self.forward(force)

        # Build loss functional as squared error b/w prediction and self.output
        loss = ufl.dot(pred - output, pred - output) * ufl.dx
        J = d_ad.assemble(loss)

        # Build controls to allow modification of the source term
        control = d_ad.Control(force)
        return pyad.reduced_functional_numpy.ReducedFunctionalNumPy(J, control)
    
    def assemble_problem(self, target):
        """ Return the solution matrix and dof-ordered output """
        if not hasattr(self, 'solution_matrix'):
            logger.info(f'Assembling {self.__class__.__name__} for the first time')

            # Assemble the problem
            trial = dlf.TrialFunction(self.function_space)
            test = dlf.TestFunction(self.function_space)

            A = dlf.assemble( self.lhs( trial, test ) )
            self.bc.apply(A)

            B = dlf.assemble( self.rhs( trial, test ) )

            # Precompute matrix for OLS solution
            A = sparse.csr_matrix(A.array())
            B = sparse.csr_matrix(B.array())
            solution_matrix = sparse.linalg.inv(A.T @ A) @ A.T @ B

            # Save for future use
            self.solution_matrix = solution_matrix.todense()

        return self.solution_matrix, self.vertex_to_dof_order(target)

class BuildPoissonProblem(BuildDolfinProblem):
    """ Poisson equation with source term """
    def __init__(self, mesh, nu=0.005):
        super().__init__(mesh)
        self.nu = nu

        self.element = ufl.FiniteElement('CG', mesh.ufl_cell(), 1)
        self.function_space = dlf.FunctionSpace(mesh, self.element)
    
        # Create the boundary condition
        self.bc = d_ad.DirichletBC(self.function_space, d_ad.Constant(0.), 'on_boundary')
    
    def lhs(self, u, v):
        return -self.nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    def rhs(self, f, v):
        return f * v * ufl.dx

class BuildStokesProblem(BuildDolfinProblem):
    """ Incompressible Stokes equation on a square box """
    def __init__(self, mesh, nu=0.005):
        super().__init__(mesh)

        self.nu = nu # Viscosity coefficient
        self.delta = 0.2 * dlf.CellDiameter(mesh)**2 # For first-order element stabilization

        self.vector_element = ufl.VectorElement('CG', mesh.ufl_cell(), 1)
        self.scalar_element = ufl.FiniteElement('CG', mesh.ufl_cell(), 1)
        self.mixed_element = self.vector_element * self.scalar_element

        self.function_space = dlf.FunctionSpace(mesh, self.vector_element)
        self.mixed_function_space = dlf.FunctionSpace(mesh, self.mixed_element)

        self.bc = d_ad.DirichletBC(self.mixed_function_space.sub(0), d_ad.Constant((0, 0)), 'on_boundary')
    
    def lhs(self, trial, test):
        """ Stabilized first order formulation for mixed elements
        """
        u, p = trial
        v, q = test
        return - self.nu * ufl.inner( ufl.grad(u), ufl.grad(v) ) * ufl.dx \
               + ufl.div(v) * p * ufl.dx \
               - ufl.div(u) * q * ufl.dx \
               - self.delta * ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
    
    def rhs(self, force, test):
        """ Stabilized first order formulation for mixed elements
        """
        v, q = test
        return - ufl.inner(force, v) * ufl.dx \
               + self.delta * ufl.inner(force, ufl.grad(q)) * ufl.dx

    def forward(self, force):
        """ Solve incompressible stokes equation and return only velocity 
        """
        u, p = dlf.TrialFunctions(self.mixed_function_space)
        v, q = dlf.TestFunctions(self.mixed_function_space)

        # Stabilized first order formulation for mixed elements
        a = self.lhs( (u, p), (v, q) )
        L = self.rhs( force, (v, q) )

        # Assemble and solve the problem
        pred = d_ad.Function(self.mixed_function_space)
        d_ad.solve(a == L, pred, self.bc)
        u, p = pred.split()
        return u

    def assemble_problem(self, target):
        """ Get the solution matrix but only for the subspace we care about
        """
        if not hasattr(self, 'solution_matrix'):
            logger.info(f'Assembling {self.__class__.__name__} problem for the first time')
            u, p = dlf.TrialFunctions(self.mixed_function_space)
            v, q = dlf.TestFunctions(self.mixed_function_space)

            A = dlf.assemble( self.lhs( (u, p), (v, q) ))
            self.bc.apply(A)

            B = dlf.assemble( self.rhs( u, (v, q) ))

            # Precompute matrix for OLS solution
            A = sparse.csr_matrix(A.array())
            B = sparse.csr_matrix(B.array())
            solution_matrix = sparse.linalg.inv(A.T @ A) @ A.T @ B

            # We only care about the vector solution
            vector_dofs = self.mixed_function_space.sub(0).dofmap().dofs()
            solution_matrix = solution_matrix[vector_dofs, :][:, vector_dofs]

            # Save for future use
            self.solution_matrix = solution_matrix.todense()

        return self.solution_matrix, self.vertex_to_dof_order(target)

def symmetric_gradient(u):
    return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

def stress(u, alpha):
    return symmetric_gradient(u) + alpha * ufl.div(u) * ufl.Identity(2)

class BuildElasticityAdhesionProblem(BuildDolfinProblem):
    """ Linear elasticity + adhesion equation from Oakes et al (2014) 

        The wrinkle here is that what we predict (displacement) must
        multiply with our input (adhesion stiffness) to get our target 
        (traction force).

        This further complicates things but can be handled using the same
        procedure as above.
    """
    def __init__(self, mesh, alpha=0.1, sigma_a=1.):
        super().__init__(mesh)

        self.alpha = alpha # Aggregated material properties
        self.sigma_a = sigma_a # Bulk active pressure

        self.scalar_element = ufl.FiniteElement('CG', mesh.ufl_cell(), 1)
        self.vector_element = ufl.VectorElement('CG', mesh.ufl_cell(), 1)

        self.function_space = dlf.FunctionSpace(mesh, self.scalar_element)
        self.vector_function_space = dlf.FunctionSpace(mesh, self.vector_element)

    def lhs(self, u, v, Y):
        """ Bulk term """
        return ufl.inner(stress(u, self.alpha), symmetric_gradient(v)) * ufl.dx + \
               ufl.dot(Y * u, v) * ufl.dx

    def rhs(self, n, v):
        """ Boundary term """
        return ufl.dot(-self.sigma_a * n, v) * ufl.ds

    def forward(self, Y):
        u = dlf.TrialFunction(self.vector_function_space)
        v = dlf.TestFunction(self.vector_function_space)
        n = dlf.FacetNormal(self.mesh)

        a = self.lhs(u, v, Y)
        L = self.rhs(n, v)

        # Assemble and solve the problem
        pred = d_ad.Function(self.vector_function_space)
        d_ad.solve(a == L, pred)
        return pred
    
    def reduced_functional(self, target):
        """ Reduced loss functional using dolfin adjoint """

        # Assume the target is a [2, N] ndarray representing mesh coordinates
        # This is what we typically do on dataset creation

        # dof_to_vertex_map gives the corresponding vertex for each ordered dof
        d2v = dlf.dof_to_vertex_map(self.vector_function_space)
        # Important to transpose because fenics does [N, C] but torch does [C, N]
        output = target.T.flatten()[d2v]
        output = convert_dof_array_to_function(output, self.vector_function_space)

        # Build problem functions
        Y = d_ad.Function(self.function_space)
        Y.vector()[:] = 0.

        pred = Y * self.forward(Y)

        # Build loss functional as squared error b/w prediction and self.output
        loss = ufl.dot(pred - output, pred - output) * ufl.dx
        J = d_ad.assemble(loss)

        # Build controls to allow modification of the source term
        control = d_ad.Control(Y)
        return pyad.reduced_functional_numpy.ReducedFunctionalNumPy(J, control)

    def assemble_problem(self, target):
        """ Return the solution matrix and dof-ordered output """
        if not hasattr(self, 'solution_matrix'):
            logger.info(f'Assembling {self.__class__.__name__} problem for the first time')
            u = dlf.TrialFunction(self.vector_function_space)
            v = dlf.TestFunction(self.vector_function_space)
            n = dlf.FacetNormal(self.mesh)

            A1 = dlf.assemble( ufl.inner(stress(u, self.alpha), symmetric_gradient(v)) * ufl.dx )
            A2 = dlf.assemble( ufl.inner(u, v) * ufl.dx )
            B = dlf.assemble( self.rhs( n, v) )

            A1 = sparse.csr_matrix(A1.array())
            A2 = sparse.csr_matrix(A2.array())
            B = B[:]

            # Save for future use
            self.solution_matrix = (A1, A2, B)

        return self.solution_matrix, self.vertex_to_dof_order(target)