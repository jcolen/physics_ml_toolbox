import ufl
import dolfin as dlf
import dolfin_adjoint as d_ad
import pyadjoint as pyad

from mesh_utils import convert_dof_array_to_function

class BuildDolfinProblem(object):
    """ Generic dolfin problem builder. Generates predictions and reduced functionals for calculating dJ/dF """
    def __init__(self, mesh):
        self.mesh = mesh
    
    def forward(self, force):
        """ Forward problem solved by dolfin """
        raise NotImplementedError

    def reduced_functional(self, target):
        """ Reduced loss functional using dolfin adjoint """

        # Assume the target is a [2, N] ndarray representing mesh coordinates
        # This is what we typically do on dataset creation

        # dof_to_vertex_map gives the corresponding vertex for each ordered dof
        d2v = dlf.dof_to_vertex_map(self.function_space)
        output = target.flatten()[d2v]
        output = convert_dof_array_to_function(output, self.function_space)

        # Build problem functions
        force = d_ad.Function(self.function_space)
        force.vector()[:] = 0.

        pred = self.forward(force)

        # Build loss functional as squared error b/w prediction and self.output
        loss = ufl.inner(pred - output, pred - output) * ufl.dx
        J = d_ad.assemble(loss)

        # Build controls to allow modification of the source term
        control = d_ad.Control(force)
        return pyad.reduced_functional_numpy.ReducedFunctionalNumPy(J, control)

class BuildPoissonProblem(BuildDolfinProblem):
    """ Poisson equation with source term """
    def __init__(self, mesh):
        super().__init__(mesh)
        self.element = ufl.FiniteElement('CG', mesh.ufl_cell(), 1)
        self.function_space = dlf.FunctionSpace(mesh, self.element)
    
        # Create the boundary condition
        self.bc = d_ad.DirichletBC(self.function_space, d_ad.Constant(0.), 'on_boundary')

    def forward(self, force):
        u = dlf.TrialFunction(self.function_space)
        v = dlf.TestFunction(self.function_space)

        # Assemble the problem
        a = -ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = force * v * ufl.dx

        pred = d_ad.Function(self.function_space)
        d_ad.solve(a == L, pred, self.bc)
        return pred

class BuildStokesProblem(BuildDolfinProblem):
    """ Incompressible Stokes equation on a square box """
    def __init__(self, mesh, nu=0.005):
        super().__init__(mesh)

        self.vector_element = ufl.VectorElement('CG', mesh.ufl_cell(), 1)
        self.scalar_element = ufl.FiniteElement('CG', mesh.ufl_cell(), 1)
        self.mixed_element = self.vector_element * self.scalar_element

        self.function_space = dlf.FunctionSpace(mesh, self.vector_element)
        self.mixed_function_space = dlf.FunctionSpace(mesh, self.mixed_element)

        self.nu = nu # Viscosity coefficient
        self.delta = 0.2 * dlf.CellDiameter(mesh)**2

        self.bc = d_ad.DirichletBC(self.mixed_function_space.sub(0), d_ad.Constant((0, 0)), 'on_boundary')


    def forward(self, force):
        u, p = dlf.TrialFunctions(self.mixed_function_space)
        v, q = dlf.TestFunctions(self.mixed_function_space)

        # Stabilized first order formulation for mixed elements
        a = - self.nu * ufl.inner( ufl.grad(u), ufl.grad(v) ) * ufl.dx \
            + ufl.div(v) * p * ufl.dx \
            - ufl.div(u) * q * ufl.dx \
            - self.delta * ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
        
        L = -ufl.inner(force, v) * ufl.dx + self.delta * ufl.inner(force, ufl.grad(q)) * ufl.dx


        # Assemble and solve the problem
        pred = d_ad.Function(self.mixed_function_space)
        d_ad.solve(a == L, pred, self.bc)
        u, p = pred.split()
        return u

class BuildElasticityAdhesionProblem(BuildDolfinProblem):
    """ Linear elasticity + adhesion equation from Oakes et al (2014) """
    def __init__(self, mesh, alpha=0.1, sigma_a=1.):
        super().__init__(mesh)

        self.scalar_element = ufl.FiniteElement('CG', mesh.ufl_cell(), 1)
        self.vector_element = ufl.VectorElement('CG', mesh.ufl_cell(), 1)

        self.function_space = dlf.FunctionSpace(mesh, self.scalar_element)
        self.vector_function_space = dlf.FunctionSpace(mesh, self.vector_element)

        self.alpha = alpha
        self.sigma_a = sigma_a

    def forward(self, Y):
        u = dlf.TrialFunction(self.vector_function_space)
        v = dlf.TestFunction(self.vector_function_space)
        n = dlf.FacetNormal(self.mesh)

        def symmetric_gradient(u):
            return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

        def stress(u):
            return symmetric_gradient(u) + self.alpha * ufl.div(u) * ufl.Identity(2)

        a = ufl.inner(stress(u), symmetric_gradient(v)) * ufl.dx + ufl.dot(Y * u, v) * ufl.dx
        L = ufl.dot(-self.sigma_a * n, v) * ufl.ds

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
        output = target.flatten()[d2v]
        output = convert_dof_array_to_function(output, self.vector_function_space)

        # Build problem functions
        Y = d_ad.Function(self.function_space)
        Y.vector()[:] = 0.

        u_pred = self.forward(Y)

        # Build loss functional as squared error b/w prediction and self.output
        loss = ufl.dot(Y * u_pred - output, Y * u_pred - output) * ufl.dx
        J = d_ad.assemble(loss)

        # Build controls to allow modification of the source term
        control = d_ad.Control(Y)
        return pyad.reduced_functional_numpy.ReducedFunctionalNumPy(J, control)