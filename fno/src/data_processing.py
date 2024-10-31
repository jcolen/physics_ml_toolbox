import logging
import pathlib
from scipy.io import loadmat
import h5py
import torch
import numpy as np

from torch.utils.data import TensorDataset
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_burgers_1d_data(
        path='../data/Burgers/burgers_data_R10.mat',
        space_resolution=1,
        positional_encoding=True,
        validation_split=0.2,
        random_seed=42):
    
    # Load mat file
    logger.info(f'Loading dataset from {path}')
    data = loadmat(path)
    
    # Load input and output data
    x_data = data['a'][:, ::space_resolution]
    y_data = data['u'][:, ::space_resolution]

    logger.info(f'Input shape: {x_data.shape}\tOutput shape: {y_data.shape}')

    # Convert to torch tensors
    N, X = x_data.shape
    x_data = torch.FloatTensor(x_data[:, None]) #[N, 1, X]
    y_data = torch.FloatTensor(y_data[:, None]) #[N, 1, X]

    if positional_encoding:
        logger.info('Adding positional encoding')
        xmin, xmax = 0, 1
        x_grid = torch.FloatTensor(np.linspace(xmin, xmax, X))[None, None, :] #[1, 1, X]

        x_data = torch.concatenate([
            x_data,
            x_grid.repeat([N, 1, 1]), 
        ], dim=1) #[N, 2, X]

    # Create dataset
    logger.info(f'Creating TensorDataset with input shape: {x_data.shape}, output shape: {y_data.shape}')
    dataset = TensorDataset(x_data.float(), y_data.float())

    if validation_split is not None and validation_split > 0:
        train, val = torch.utils.data.random_split(
            dataset, 
            [1-validation_split, validation_split],
            generator=torch.Generator().manual_seed(random_seed)
        )
        logger.info(f'Training dataset length: {len(train)}')
        logger.info(f'Validation dataset length: {len(val)}')
        return train, val

    else:
        logger.info(f'Dataset length: {len(dataset)}')
        return dataset, None
    
def load_burgers_2d_data(
        path='../data/Burgers/burgers_v100_t200_r1024_N2048.mat',
        space_resolution=1,
        time_resolution=1,
        spacetime_encoding=True,
        validation_split=0.2,
        random_seed=42):
    
    # Load mat file
    logger.info(f'Loading dataset from {path}')
    data = loadmat(path)

    # Load metadata
    visc = data['visc']
    logger.info(f'Loaded Burgers dataset with nu = {np.squeeze(visc):g}')
    
    # Load input and output data
    x_data = data['input'][:, ::space_resolution]
    y_data = data['output'][:, ::time_resolution, ::space_resolution]

    logger.info(f'Input shape: {x_data.shape}\tOutput shape: {y_data.shape}')

    # Convert to torch tensors
    x_data = torch.FloatTensor(x_data) #[N, X]
    y_data = torch.FloatTensor(y_data) #[N, T, X]

    # Pad x_data to have same shape as y_data
    N, T, X = y_data.shape
    x_data = x_data[:, None, None, :].repeat([1, 1, T, 1]) #[N, 1, T, X]
    y_data = y_data[:, None, :, :] #[N, 1, T, X]

    # Ensure y_data has correct initial condition too
    y_data[:, :, 0, :] = x_data[:, :, 0, :]

    if spacetime_encoding:
        logger.info('Adding positional and temporal encoding')
        xmin, xmax = 0, 1
        tmin, tmax = 0, 1
        x_grid = torch.FloatTensor(np.linspace(xmin, xmax, X))[None, None, None, :] #[1, 1, 1, X]
        t_grid = torch.FloatTensor(np.linspace(tmin, tmax, T))[None, None, :, None] #[1, 1, T, 1]

        x_data = torch.concatenate([
            x_data,
            x_grid.repeat([N, 1, T, 1]), 
            t_grid.repeat([N, 1, 1, X]),
        ], dim=1) #[N, 3, T, X]

    # Create dataset
    logger.info(f'Creating TensorDataset with input shape: {x_data.shape}, output shape: {y_data.shape}')
    dataset = TensorDataset(x_data.float(), y_data.float())

    if validation_split is not None and validation_split > 0:
        train, val = torch.utils.data.random_split(
            dataset, 
            [1-validation_split, validation_split],
            generator=torch.Generator().manual_seed(random_seed)
        )
        logger.info(f'Training dataset length: {len(train)}')
        logger.info(f'Validation dataset length: {len(val)}')
        return train, val

    else:
        logger.info(f'Dataset length: {len(dataset)}')
        return dataset, None
    
def load_darcy_2d_data(
        train_path='../data/Darcy/piececonst_r421_N1024_smooth1.mat',
        val_path='../data/Darcy/piececonst_r421_N1024_smooth1.mat',
        space_resolution=4,
        positional_encoding=True):
    
    def load_and_process(path):
        data = loadmat(path)

        # Load input and output data
        x_data = data['coeff'][:, ::space_resolution,::space_resolution]
        y_data = data['sol'][:, ::space_resolution, ::space_resolution]        

        logger.info(f'Input shape: {x_data.shape}\tOutput shape: {y_data.shape}')

        # Convert to torch tensors
        N, Y, X = x_data.shape
        x_data = torch.FloatTensor(x_data[:, None]) #[N, 1, Y, X]
        y_data = torch.FloatTensor(y_data[:, None]) #[N, 1, Y, X]

        if positional_encoding:
            logger.info('Adding positional encoding')
            xmin, xmax = 0, 1
            ymin, ymax = 0, 1
            y_grid = torch.FloatTensor(np.linspace(ymin, ymax, Y))[None, None, :, None] #[1, Y, 1]
            x_grid = torch.FloatTensor(np.linspace(xmin, xmax, X))[None, None, None, :] #[1, 1, X]

            x_data = torch.concatenate([
                x_data,
                y_grid.repeat([N, 1, 1, X]), 
                x_grid.repeat([N, 1, Y, 1]),
            ], dim=1)
        
        # Create torch dataset
        logger.info(f'Creating TensorDataset with input shape: {x_data.shape}, output shape: {y_data.shape}')
        dataset = TensorDataset(x_data.float(), y_data.float())
        return dataset

    # Load mat file
    logger.info(f'Loading train_dataset from {train_path}')
    train = load_and_process(train_path)

    logger.info(f'Loading val dataset from {val_path}')
    val = load_and_process(val_path)

    logger.info(f'Training dataset length: {len(train)}')
    logger.info(f'Validation dataset length: {len(val)}')

    return train, val