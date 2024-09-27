import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pinn import PINN

class BurgersPINN(PINN):
    def parse_data(self, data):
        self.x_f = nn.Parameter(torch.FloatTensor(data['x_f']), requires_grad=True)
        self.t_f = nn.Parameter(torch.FloatTensor(data['t_f']), requires_grad=True)
        self.u_f = nn.Parameter(torch.FloatTensor(data['u_f']), requires_grad=True)

        X = torch.cat([self.x_f, self.t_f], dim=1)
        self.lb = nn.Parameter(X.min(0)[0], requires_grad=False)
        self.ub = nn.Parameter(X.max(0)[0], requires_grad=False)

    def get_params(self):
        raise NotImplementedError

    def phys_loss(self):
        """ Loss for Burgers' equation """
        u_f = self.forward(torch.cat([self.x_f, self.t_f], dim=-1))

        u_t = torch.autograd.grad(u_f.sum(), self.t_f, create_graph=True)[0]
        u_x = torch.autograd.grad(u_f.sum(), self.x_f, create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), self.x_f, create_graph=True)[0]

        lambda1, eta = self.get_params()
        eq = u_t + lambda1 * u_f * u_x - eta * u_xx
        phys_loss = eq.pow(2).mean()
        return phys_loss

    def mse_loss(self):
        """ Reconstruction loss on predicted u """
        u_f = self.forward(torch.cat([self.x_f, self.t_f], dim=-1))
        mse_loss = (u_f - self.u_f).pow(2).mean()
        return mse_loss

class BurgersBVPPINN(BurgersPINN):
    def __init__(self, data: dict, log_dir: str):
        super().__init__(
            data=data,
            layers=[2, 50, 50, 50, 50, 1],
            num_params=0,
            log_dir=log_dir
        )

    def parse_data(self, data):
        super().parse_data(data)

        # Also parse boundary terms used for reconstruction error
        self.x_b = nn.Parameter(torch.FloatTensor(data['x_b']), requires_grad=False)
        self.t_b = nn.Parameter(torch.FloatTensor(data['t_b']), requires_grad=False)
        self.u_b = nn.Parameter(torch.FloatTensor(data['u_b']), requires_grad=False)

    def get_params(self):
        """ Known parameters for solving BVP """
        return 1., 0.01 / np.pi

    def mse_loss(self):
        """ Only compute MSE loss at boundary points """
        u_b = self.forward(torch.cat([self.x_b, self.t_b], dim=-1))
        mse_loss = (u_b - self.u_b).pow(2).mean()
        return mse_loss

class BurgersParameterPINN(BurgersPINN):
    def __init__(self, data: dict, log_dir: str):
        super().__init__(
            data=data,
            layers=[2, 50, 50, 50, 50, 1],
            num_params=2,
            log_dir=log_dir
        )

    def get_params(self):
        lambda1 = self.params[0]
        eta = self.params[1].exp()
        return lambda1, eta

    def log_tensorboard(self, loss, mse_loss, phys_loss):
        super().log_tensorboard(loss,  mse_loss, phys_loss)

        lambda1, eta = self.get_params()
        self.writer.add_scalar('lambda', lambda1.detach().item(), self.iter)
        self.writer.add_scalar('eta', eta.detach().item(), self.iter)