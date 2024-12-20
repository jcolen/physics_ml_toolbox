import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.special import legendre
from numpy.polynomial.legendre import leggauss

from pinn import PINN

class PoissonVariationalPINN(PINN):
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

        self.init_grids()

    def parse_data(self, data):
        self.x_u = nn.Parameter(torch.FloatTensor(data['x_u']), requires_grad=False)
        self.y_u = nn.Parameter(torch.FloatTensor(data['y_u']), requires_grad=False)
        self.u_u = nn.Parameter(torch.FloatTensor(data['u_u']), requires_grad=False)

        X = torch.cat([self.x_u, self.y_u], dim=1)
        self.lb = nn.Parameter(X.min(0)[0], requires_grad=False)
        self.ub = nn.Parameter(X.max(0)[0], requires_grad=False)

    def init_grids(self, H=50):
        """ Initialize subgrids for variational computation of derivatives
            H: Number of quadrature points
        """

        xx, ww = leggauss(H) # Sample points and quadrature weights

        # Resample to observed boundary
        x_f, y_f = np.meshgrid(
            self.lb[0].item() + (xx + 1) / 2. * (self.ub[0].item() - self.lb[0].item()),
            self.lb[1].item() + (xx + 1) / 2. * (self.ub[1].item() - self.lb[1].item()),
        )
        self.x_f = nn.Parameter(torch.FloatTensor(x_f), requires_grad=False)
        self.y_f = nn.Parameter(torch.FloatTensor(y_f), requires_grad=False)
        self.w_f = nn.Parameter(torch.FloatTensor(ww), requires_grad=False)

        base_polynomial = np.poly1d([1, 0, -1]) ** 3
        leg_max = 25 # Highest order Legendre polynomial

        phi = []
        phi_xx = []
        phi_yy = []
        for l_x in range(leg_max):
            px = base_polynomial * legendre(l_x)
            px_xx = ww * px.deriv(2)(xx)
            px = ww * px(xx)

            for l_y in range(leg_max):
                py = base_polynomial * legendre(l_y)
                py_yy = ww * py.deriv(2)(xx)
                py = ww * py(xx)

                phi.append(px[:, None] * py[None, :])
                phi_xx.append(px_xx[:, None] * py[None, :])
                phi_yy.append(px[:, None] * py_yy[None, :])

        self.phi = nn.Parameter(torch.FloatTensor(np.stack(phi)), requires_grad=False)
        self.phi_xx = nn.Parameter(torch.FloatTensor(np.stack(phi_xx)), requires_grad=False)
        self.phi_yy = nn.Parameter(torch.FloatTensor(np.stack(phi_yy)), requires_grad=False)


    def get_params(self):
        raise NotImplementedError

    def batch_inner(self, u, v):
        return torch.einsum('ixy,jxy->ij', u, v)

    def phys_loss(self):
        """ Weak-form loss for Poisson equation """
        u_f = self.forward(torch.stack([self.x_f, self.y_f], dim=-1))[None, :, :, 0]
        lhs = self.batch_inner(u_f, self.phi_xx) + self.batch_inner(u_f, self.phi_yy)
        
        A, B, omega = self.get_params()
        f_f = -torch.sin(omega * self.y_f) * (2 * A * omega**2 * torch.sin(omega * self.x_f) + B * (omega**2 * self.x_f**2 - 2))

        rhs = self.batch_inner(f_f[None], self.phi)

        phys_loss = (lhs - rhs).pow(2).mean()
        return 1e2 * phys_loss

    def mse_loss(self):
        """ Reconstruction loss on predicted u """
        u_u = self.forward(torch.cat([self.x_u, self.y_u], dim=-1))
        mse_loss = (u_u - self.u_u).pow(2).mean()
        return mse_loss

class PoissonBVPVariationalPINN(PoissonVariationalPINN):
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
    
    def get_params(self):
        return 0.1, 1, 2*np.pi