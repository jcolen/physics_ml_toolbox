import torch
import h5py
import numpy as np

import dolfin as dlf
import dolfin_adjoint as d_ad

from scipy.interpolate import griddata
from tqdm.auto import trange

import dolfin_problems

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path='../data/poisson_dataset.hdf5',
                 mesh='../data/square_mesh.xml',
                 build_problem='dolfin_problems.BuildPoissonProblem'):
        super().__init__()

        self.path = path

        # Load data from hdf5 dataset
        with h5py.File(path, 'r') as h5f:

            self.inputs = np.stack([h5f[f'{i:06d}/inputs'] for i in trange(h5f.attrs['dataset_size'])])
            self.force = np.stack([h5f[f'{i:06d}/force'] for i in trange(h5f.attrs['dataset_size'])])
            self.output = np.stack([h5f[f'{i:06d}/output'] for i in trange(h5f.attrs['dataset_size'])])
        
        self.mesh = d_ad.Mesh(mesh)
        self.build_problem = eval(build_problem)(self.mesh)
    
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        sample = {
            'inputs': self.inputs[idx],
            'output': self.output[idx],
            'force': self.force[idx], #Not used by PBNN
            'grid_x': self.inputs[idx, 0],
            'grid_y': self.inputs[idx, 1],
            'mesh_x': self.mesh.coordinates()[:, 0],
            'mesh_y': self.mesh.coordinates()[:, 1],
        }

        sample['Jhat'] = self.build_problem.reduced_functional(sample['output'])
        sample['function_space'] = self.build_problem.function_space

        return sample