import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pinn import PINN

class SchrodingerBVPPINN(PINN):
    def __init__(self, data: dict, log_dir: str):
        super().__init__(
            data=data,
            layers=[2, 100, 100, 100, 100, 2],
            num_params=0,
            lr=1e0,
            log_dir=log_dir,
            act=nn.Tanh,
            optimizer_type='lbfgs',
        )

    def parse_data(self, data):
        # Collocation points for physics-informed loss
        self.x_f = nn.Parameter(torch.FloatTensor(data['x_f']), requires_grad=True)
        self.t_f = nn.Parameter(torch.FloatTensor(data['t_f']), requires_grad=True)

        # Also parse boundary terms used for reconstruction error
        self.x_b = nn.Parameter(torch.FloatTensor(data['x_b']), requires_grad=False)
        self.t_b = nn.Parameter(torch.FloatTensor(data['t_b']), requires_grad=False)
        self.u_b = nn.Parameter(torch.FloatTensor(data['u_b'].real), requires_grad=False)
        self.v_b = nn.Parameter(torch.FloatTensor(data['u_b'].imag), requires_grad=False)

        X = torch.cat([self.x_b, self.t_b], dim=1)
        self.lb = nn.Parameter(X.min(0)[0], requires_grad=False)
        self.ub = nn.Parameter(X.max(0)[0], requires_grad=False)

        # Additional boundary conditions
        tb = torch.linspace(self.lb[1], self.ub[1], 50)[:, None]
        self.x_lb = nn.Parameter(0*tb + self.lb[0], requires_grad=True)
        self.t_lb = nn.Parameter(tb, requires_grad=False)
        self.x_ub = nn.Parameter(0*tb + self.ub[0], requires_grad=True)
        self.t_ub = nn.Parameter (tb, requires_grad=False)

    def phys_loss(self):
        """ Loss for Schrodinger equation with boundary conditions """
        #Periodic BCs
        h_lb = self(torch.cat([self.x_lb, self.t_lb], dim=-1))
        u_lb = h_lb[:, 0:1]
        v_lb = h_lb[:, 1:2]
        ux_lb = torch.autograd.grad(u_lb.sum(), self.x_lb, create_graph=True)[0]
        vx_lb = torch.autograd.grad(v_lb.sum(), self.x_lb, create_graph=True)[0]
        
        h_ub = self(torch.cat([self.x_ub, self.t_ub], dim=-1))
        u_ub = h_ub[:, 0:1]
        v_ub = h_ub[:, 1:2]
        ux_ub = torch.autograd.grad(u_ub.sum(), self.x_ub, create_graph=True)[0]
        vx_ub = torch.autograd.grad(v_ub.sum(), self.x_ub, create_graph=True)[0]
        
        bc1 = (u_lb - u_ub).pow(2).sum() + (v_lb - v_ub).pow(2).sum()
        bc2 = (ux_lb - ux_ub).pow(2).sum() + (vx_lb - vx_ub).pow(2).sum()
        
        #Schrodinger loss
        h_f = self(torch.cat([self.x_f, self.t_f], dim=-1))
        u_f = h_f[:, 0:1]
        v_f = h_f[:, 1:2]
        
        u_t = torch.autograd.grad(u_f.sum(), self.t_f, create_graph=True)[0]
        u_x = torch.autograd.grad(u_f.sum(), self.x_f, retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), self.x_f, create_graph=True)[0]
        v_t = torch.autograd.grad(v_f.sum(), self.t_f, create_graph=True)[0]
        v_x = torch.autograd.grad(v_f.sum(), self.x_f, retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), self.x_f, create_graph=True)[0]
        
        f_u = u_t + 0.5*v_xx + (u_f**2 + v_f**2) * v_f
        f_v = v_t - 0.5*u_xx - (u_f**2 + v_f**2) * u_f
        
        phys = f_u.pow(2).sum() + f_v.pow(2).sum()
        return phys + bc1 + bc2

    def mse_loss(self):
        """ Reconstruction loss on predicted u """
        h_b = self.forward(torch.cat([self.x_b, self.t_b], dim=-1))
        u_b = h_b[:, 0:1]
        v_b = h_b[:, 1:2]
        mse_loss = (u_b - self.u_b).pow(2).sum() + (v_b - self.v_b).pow(2).sum()
        return mse_loss