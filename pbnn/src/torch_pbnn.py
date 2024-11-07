import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import v2

from mesh_utils import multichannel_img_to_dofs
import models

class TorchPBNN(nn.Module):
    def __init__(self,
                 model_type : str = 'models.LatentNet',
                 model_kwargs : dict = {},
                 positive_definite : bool = True):
        super().__init__()

        self.convnet = eval(model_type)(**model_kwargs)
        self.positive_definite = positive_definite
        self.blur = v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.))

    def training_step(self, sample):
        # Get force prediction
        force = self.forward(
            sample['inputs'], 
            function_space=sample['function_space'],
            coords=(sample['grid_x'], sample['grid_y']))
        force = force.T.flatten() # Reshape for application to Jhat

        # Get output prediction
        rhs = torch.FloatTensor(sample['rhs']).to(force.device)
        diag = torch.FloatTensor(sample['diag']).to(force.device)
        target = torch.FloatTensor(sample['target']).to(force.device)

        pred = rhs @ force * diag

        loss = torch.sum( (pred - target) * (pred - target) * diag )
        loss.backward()
        
        return force, loss

    def validation_step(self, sample):
        # Get force prediction
        force = self.forward(
            sample['inputs'], 
            function_space=sample['function_space'],
            coords=(sample['grid_x'], sample['grid_y']))
        force = force.T.flatten() # Reshape for application to Jhat
        
        # Get output prediction
        rhs = torch.FloatTensor(sample['rhs']).to(force.device)
        diag = torch.FloatTensor(sample['diag']).to(force.device)
        target = torch.FloatTensor(sample['target']).to(force.device)

        pred = rhs @ force * diag

        loss = torch.sum( (pred - target) * (pred - target) * diag )

        return force, loss
    
    def forward(self, x, function_space=None, coords=None):
        # Add fictitious batch dimension and apply convnet
        y = self.convnet(x[None])

        if self.positive_definite:
            y = y.exp()

        y = self.blur(y)

        # Remove fictitious batch dimension
        y = y[0]

        if function_space is None or coords is None:
            return y
        
        # Move the source term to the mesh vertices
        return multichannel_img_to_dofs(y, *coords, function_space, return_function=False)