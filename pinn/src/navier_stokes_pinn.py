import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pinn import PINN

class NavierStokesParameterPINN(PINN):
    def __init__(self, 
                 data: dict , 
                 layers: list = [3, 100, 100, 100, 100, 100, 3], 
                 num_params: int = 2,
                 lr: float = 1e0,
                 act: nn.Module = nn.Tanh,
                 optimizer_type: str ='lbfgs',
                 log_dir: str = "./tb_logs/NavierStokes"):

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
        self.t_f = nn.Parameter(torch.FloatTensor(data['t_f']), requires_grad=True)

        self.u_f = nn.Parameter(torch.FloatTensor(data['u_f']), requires_grad=False)
        self.v_f = nn.Parameter(torch.FloatTensor(data['v_f']), requires_grad=False)

        X = torch.cat([self.x_f, self.y_f, self.t_f], dim=1)
        self.lb = nn.Parameter(X.min(0)[0], requires_grad=False)
        self.ub = nn.Parameter(X.max(0)[0], requires_grad=False)

    def get_params(self):
        lambda_1 = self.params[0]
        lambda_2 = self.params[1]
        return lambda_1, lambda_2

    def phys_loss(self):
        """ Loss for incompressible Navier Stokes' equation """
        uvp_f = self.forward(torch.cat([self.x_f, self.y_f, self.t_f], dim=-1))
        u_f = uvp_f[:, 0:1]
        v_f = uvp_f[:, 1:2]
        p_f = uvp_f[:, 2:3]

        u_t = torch.autograd.grad(u_f.sum(), self.t_f, create_graph=True)[0]
        u_x = torch.autograd.grad(u_f.sum(), self.x_f, retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), self.x_f, create_graph=True)[0]
        u_y = torch.autograd.grad(u_f.sum(), self.y_f, retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), self.y_f, create_graph=True)[0]  
        
        v_t = torch.autograd.grad(v_f.sum(), self.t_f, create_graph=True)[0]
        v_x = torch.autograd.grad(v_f.sum(), self.x_f, retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), self.x_f, create_graph=True)[0]
        v_y = torch.autograd.grad(v_f.sum(), self.y_f, retain_graph=True, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), self.y_f, create_graph=True)[0]  
        
        p_x = torch.autograd.grad(p_f.sum(), self.x_f, create_graph=True)[0]
        p_y = torch.autograd.grad(p_f.sum(), self.y_f, create_graph=True)[0]

        lambda_1, lambda_2 = self.get_params()
        eq_u = u_t + lambda_1 * (u_f * u_x + v_f * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        eq_v = v_t + lambda_1 * (u_f * v_x + v_f * v_y) + p_y - lambda_2 * (v_xx + v_yy)

        phys_loss = eq_u.pow(2).mean() + eq_v.pow(2).mean()
        return phys_loss

    def mse_loss(self):
        """ Reconstruction loss on predicted u """
        uvp_f = self.forward(torch.cat([self.x_f, self.y_f, self.t_f], dim=-1))
        u_f = uvp_f[:, 0:1]
        v_f = uvp_f[:, 1:2]
        mse_loss = (u_f - self.u_f).pow(2).mean() + (v_f - self.v_f).pow(2).mean()
        return mse_loss


    def log_tensorboard(self, loss, mse_loss, phys_loss):
        super().log_tensorboard(loss,  mse_loss, phys_loss)

        lambda_1, lambda_2 = self.get_params()
        self.writer.add_scalar('lambda_1', lambda_1.detach().item(), self.iter)
        self.writer.add_scalar('lambda_2', lambda_2.detach().item(), self.iter)