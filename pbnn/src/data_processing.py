import torch
import h5py
import numpy as np

import ufl
import dolfin as dlf
import dolfin_adjoint as d_ad
import pyadjoint as pyad

from scipy.interpolate import griddata

def scalar_img_to_mesh(img, x, y, function_space, vals_only=True, use_torch=True):
    ''' Nearest neighbor interpolation on a mesh '''
    dof_coords = function_space.tabulate_dof_coordinates().copy() # [N, 2]

    if use_torch:
        dof_coords = torch.tensor(dof_coords, dtype=img.dtype, device=img.device)
        xmin, ymin = x.min(), y.min()
        dx = x[1,1]-x[0,0]
        dy = y[1,1]-y[0,0]
        dof_x = (dof_coords[:, 0] - xmin) / dx
        dof_y = (dof_coords[:, 1] - ymin) / dy
        
        mesh_vals = img[dof_y.long(), dof_x.long()]
    else:
        mesh_vals = griddata((x.flatten(), y.flatten()), img.flatten(), dof_coords, method='nearest')
    
    if vals_only: 
        return mesh_vals

    mesh_func = d_ad.Function(function_space)
    if use_torch:
        mesh_func.vector()[:] = mesh_vals.detach().cpu().numpy()
    else:
        mesh_func.vector()[:] = mesh_vals
    
    return mesh_func

class BuildPoissonProblem:
    '''
    Generate reduced functional for calculating dJ/dF
    '''
    def __init__(self, mesh):
        # Build problem space
        self.mesh = mesh
        self.element = ufl.FiniteElement('CG', mesh.ufl_cell(), 1)
        self.function_space = dlf.FunctionSpace(mesh, self.element)

    def forward(self, f):
        u = dlf.TrialFunction(self.function_space)
        v = dlf.TestFunction(self.function_space)

        # Assemble the problem
        a = -ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx

        # Create the boundary condition
        bc = d_ad.DirichletBC(self.function_space, d_ad.Constant(0.), 'on_boundary')

        pred = d_ad.Function(self.function_space)
        d_ad.solve(a == L, pred, bc)
        return pred

    def __call__(self, sample):
        output = scalar_img_to_mesh(
            sample['output'], 
            sample['mesh_x'], 
            sample['mesh_y'], 
            self.function_space,
            use_torch=False,
            vals_only=False)

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

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, path = '../data/dataset.hdf5'):
        super().__init__()

        self.path = path

        # Load data from hdf5 dataset
        with h5py.File(path, 'r') as h5f:
            self.mesh_size = h5f['mesh_size'][()]

            self.inputs = h5f['inputs'][()]
            self.forces = h5f['forces'][()]
            self.outputs = h5f['outputs'][()]
        
        self.mesh = d_ad.UnitSquareMesh(nx=self.mesh_size, ny=self.mesh_size)
        self.build_problem = BuildPoissonProblem(self.mesh)
    
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        sample = {
            'inputs': self.inputs[idx],
            'output': self.outputs[idx],
            'force': self.forces[idx], #Not used by PBNN
            'grid_x': self.inputs[idx, 0],
            'grid_y': self.inputs[idx, 1],
            'mesh_x': self.mesh.coordinates()[:, 0],
            'mesh_y': self.mesh.coordinates()[:, 1],
        }

        sample['Jhat'] = self.build_problem(sample)
        sample['function_space'] = self.build_problem.function_space
        sample['inputs'] = torch.FloatTensor(sample['inputs'])

        return sample