import ufl
import dolfin as dlf
import dolfin_adjoint as d_ad
import pyadjoint as pyad

from mesh_utils import scalar_img_to_mesh

class BuildDolfinProblem(object):
    """ Generic dolfin problem builder. Generates predictions and reduced functionals for calculating dJ/dF """
    def __init__(self, mesh):
        self.mesh = mesh
    
    def forward(self, force):
        """ Forward problem solved by dolfin """
        raise NotImplementedError

    def reduced_functional(self, target):
        """ Reduced loss functional using dolfin adjoint """

        # Ensure target is a dolfin function, not an array
        if not isinstance(target, d_ad.Function):
            output = scalar_img_to_mesh(
                target,
                x=self.mesh.coordinate()[:, 0],
                y=self.mesh.coordinates()[:, 1],
                function_space=self.function_space,
                use_torch=False,
                vals_only=False,
            )
        else:
            output = target

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
    def __init__(self, mesh):
        super().__init__(mesh)

        self.vector_element = ufl.VectorElement('CG', mesh.ufl_cell(), 1)
        self.scalar_element = ufl.FiniteElement('CG', mesh.ufl_cell(), 1)
        self.mixed_element = self.vector_element * self.scalar_element

        self.function_space = dlf.FunctionSpace(mesh, self.vector_element)
        self.mixed_function_space = dlf.FunctionSpace(mesh, self.mixed_element)

        self.delta = 0.2 * dlf.CellDiameter(mesh)**2

        self.bc = d_ad.DirichletBC(self.mixed_function_space.sub(0), d_ad.Constant((0, 0)), 'on_boundary')


    def forward(self, force):
        u, p = dlf.TrialFunctions(self.mixed_function_space)
        v, q = dlf.TestFunctions(self.mixed_function_space)

        # Stabilized first order formulation for mixed elements
        a = ufl.inner( ufl.grad(u), ufl.grad(v) ) * ufl.dx - \
            ufl.div(v) * p * ufl.dx + \
            ufl.div(u) * q * ufl.dx + \
            self.delta * ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
        
        L = ufl.inner(force, v) * ufl.dx + self.delta * ufl.inner(force, ufl.grad(q)) * ufl.dx


        # Assemble and solve the problem
        pred = d_ad.Function(self.mixed_function_space)
        d_ad.solve(a == L, pred, self.bc)
        u, p = pred.split()
        return u