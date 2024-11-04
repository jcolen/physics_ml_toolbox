import torch
import h5py
import numpy as np

import dolfin as dlf
import dolfin_adjoint as d_ad

from scipy.interpolate import griddata

from dolfin_problems import *

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path='../data/poisson_dataset.hdf5',
                 build_problem=BuildPoissonProblem):
        super().__init__()

        self.path = path

        # Load data from hdf5 dataset
        with h5py.File(path, 'r') as h5f:
            self.mesh_size = h5f['mesh_size'][()]

            self.inputs = h5f['inputs'][()]
            self.forces = h5f['forces'][()]
            self.outputs = h5f['outputs'][()]
        
        self.mesh = d_ad.UnitSquareMesh(nx=self.mesh_size, ny=self.mesh_size)
        self.build_problem = build_problem(self.mesh)
    
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

        sample['Jhat'] = self.build_problem.reduced_functional(sample['output'])
        sample['function_space'] = self.build_problem.function_space
        sample['inputs'] = torch.FloatTensor(sample['inputs'])

        return sample