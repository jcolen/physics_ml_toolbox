import ufl
import dolfin as dlf
import dolfin_adjoint as d_ad
import pyadjoint as pyad

from mesh_utils import scalar_img_to_mesh

class BuildDolfinProblem(object):
    """ Generic dolfin problem builder. Generates predictions and reduced functionals for calculating dJ/dF """
    def __init__(self, mesh):
        # Build problem space
        self.mesh = mesh
        self.element = ufl.FiniteElement('CG', mesh.ufl_cell(), 1)
        self.function_space = dlf.FunctionSpace(mesh, self.element)
    
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

    def forward(self, force):
        u = dlf.TrialFunction(self.function_space)
        v = dlf.TestFunction(self.function_space)

        # Assemble the problem
        a = -ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = force * v * ufl.dx

        # Create the boundary condition
        bc = d_ad.DirichletBC(self.function_space, d_ad.Constant(0.), 'on_boundary')

        pred = d_ad.Function(self.function_space)
        d_ad.solve(a == L, pred, bc)
        return pred
