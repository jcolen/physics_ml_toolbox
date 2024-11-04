import torch
import torch.nn as nn
import torch.nn.functional as F

import dolfin as dlf
import dolfin_adjoint as d_ad

from torchvision.transforms import v2

from mesh_utils import multichannel_img_to_mesh

class Sin(nn.Module):
    ''' Sin activation '''
    def forward(self, x):
        return torch.sin(x)

class ConvNextBlock(nn.Module):
    ''' Convolutional block, ignoring layernorm for now '''
    def __init__(self, in_channels, out_channels, kernel_size=7, dropout_rate=0.):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding='same', padding_mode='replicate', groups=in_channels)
        self.conv2 = nn.Conv2d(out_channels, 4*out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(4*out_channels, out_channels, kernel_size=1)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = torch.sin(x)
        x = self.conv3(x)
        
        x = self.dropout(x)
        return x

class DolfinPBNN(nn.Module):
    def __init__(self,
                 input_dim=3,
                 output_dim=1,
                 num_hidden=8,
                 hidden_dim=64,
                 dropout_rate=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.read_in = nn.Sequential(
            nn.Conv2d(input_dim, 4, kernel_size=1),
            ConvNextBlock(4, hidden_dim, dropout_rate=dropout_rate)
        )
        self.downsample = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=4),
                nn.GELU()
            )
        self.read_out = nn.Conv2d(2*hidden_dim, output_dim, kernel_size=1)

        self.cnn1 = nn.ModuleList()
        self.cnn2 = nn.ModuleList()
        for i in range(num_hidden):
            self.cnn1.append(ConvNextBlock(hidden_dim, hidden_dim, dropout_rate=dropout_rate))
            self.cnn2.append(ConvNextBlock(hidden_dim, hidden_dim, dropout_rate=dropout_rate))

        self.blur = v2.GaussianBlur(kernel_size=7, sigma=(1., 3.))

    def training_step(self, sample):
        # Get force prediction
        force = self.forward(
            sample['inputs'], 
            function_space=sample['function_space'],
            coords=(sample['grid_x'], sample['grid_y']))
        force = force.T.flatten() # Reshape for application to Jhat
        
        # Get loss and gradient
        Jhat = sample['Jhat']
        loss = Jhat(force.detach().cpu().numpy())
        grad = torch.tensor(Jhat.derivative(), device=force.device)

        # Apply gradients and backprop
        force.backward(gradient=grad)
        
        return force, loss

    def validation_step(self, sample):
        # Get force prediction
        force = self.forward(
            sample['inputs'], 
            function_space=sample['function_space'],
            coords=(sample['grid_x'], sample['grid_y']))
        force = force.T.flatten() # Reshape for application to Jhat
        
        # Get loss
        Jhat = sample['Jhat']
        loss = Jhat(force.detach().cpu().numpy())

        return force, loss
    
    def forward(self, x, function_space=None, coords=None):
        # Add fictitious batch dimension
        x = x[None]
        # CNN part of computation operating on grid
        x = self.read_in(x)
        for cell in self.cnn1:
            x = x + cell(x)
        
        latent = self.downsample(x)
        for cell in self.cnn2:
            latent = latent + cell(latent)
        latent = F.interpolate(latent, x.shape[-2:])

        x = torch.cat([x, latent], dim=1)
        force = self.read_out(x)

        # Apply gaussian blur for smoothing purposes
        # Fixed nearest-neighbor interpolation causes some issues
        force = self.blur(force.exp())

        # Remove fictitious batch dimension
        force = force[0]

        if function_space is None or coords is None:
            return force
        
        # Move the source term to the mesh vertices
        force_mesh = multichannel_img_to_mesh(force, *coords, function_space, return_function=False)
        return force_mesh

