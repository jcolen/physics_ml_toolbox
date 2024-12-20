import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.special import legendre
from numpy.polynomial.legendre import leggauss

from pinn import PINN

class PoissonPINN(PINN):
    def __init__(self,
                 data: dict,
                 layers: list = [2, 50, 50, 50, 50, 50, 50, 1],
                 num_params: int = 0,
                 lr: float = 1e1,
                 act: nn.Module = nn.Tanh,
                 optimizer_type: str = 'lbfgs',
                 log_dir: str = './tb_logs/Poisson'):

        super().__init__(
            data=data,
            layers=layers,
            num_params=num_params,
            lr=lr,
            act=act,
            optimizer_type=optimizer_type,
            log_dir=log_dir,
        )


    def parse_data(self, data):
        self.x_f = nn.Parameter(torch.FloatTensor(data['x_f']), requires_grad=True)
        self.y_f = nn.Parameter(torch.FloatTensor(data['y_f']), requires_grad=True)
        self.u_f = nn.Parameter(torch.FloatTensor(data['u_f']), requires_grad=False)

        X = torch.cat([self.x_f, self.y_f], dim=1)
        self.lb = nn.Parameter(X.min(0)[0], requires_grad=False)
        self.ub = nn.Parameter(X.max(0)[0], requires_grad=False)

    def get_params(self):
        raise NotImplementedError

    def phys_loss(self):
        """ Weak-form loss for Poisson equation """
        u_f = self.forward(torch.stack([self.x_f, self.y_f], dim=-1))

        u_x = torch.autograd.grad(u_f.sum(), self.x_f, create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), self.x_f, create_graph=True)[0]

        u_y = torch.autograd.grad(u_f.sum(), self.y_f, create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), self.y_f, create_graph=True)[0]

        lhs = u_xx + u_yy

        
        A, B, omega = self.get_params()
        f_f = -torch.sin(omega * self.y_f) * (2 * A * omega**2 * torch.sin(omega * self.x_f) + B * (omega**2 * self.x_f**2 - 2))
        rhs = f_f

        phys_loss = (lhs - rhs).pow(2).mean()
        return phys_loss

    def mse_loss(self):
        """ Reconstruction loss on predicted u """
        u_f = self.forward(torch.cat([self.x_f, self.y_f], dim=-1))
        mse_loss = (u_f - self.u_f).pow(2).mean()
        return mse_loss

class PoissonBVPPINN(PoissonPINN):
    def __init__(self,
                 data: dict,
                 layers: list = [2, 50, 50, 50, 50, 50, 50, 1],
                 lr: float = 1e1,
                 act: nn.Module = nn.Tanh,
                 optimizer_type: str = 'lbfgs',
                 log_dir: str = './tb_logs/Poisson'):
        super().__init__(
            data=data,
            layers=layers,
            num_params=0,
            lr=lr,
            act=act,
            optimizer_type=optimizer_type,
            log_dir=log_dir,
        )

    def parse_data(self, data):
        super().parse_data(data)

        # Also parse boundary terms used for reconstruction error
        self.x_b = nn.Parameter(torch.FloatTensor(data['x_b']), requires_grad=False)
        self.y_b = nn.Parameter(torch.FloatTensor(data['y_b']), requires_grad=False)
        self.u_b = nn.Parameter(torch.FloatTensor(data['u_b']), requires_grad=False)
    
    def get_params(self):
        return 0.1, 1, 2*np.pi

    def mse_loss(self):
        """ Only compute MSE loss at boundary points """
        u_b = self.forward(torch.cat([self.x_b, self.y_b], dim=-1))
        mse_loss = (u_b - self.u_b).pow(2).mean()
        return mse_loss