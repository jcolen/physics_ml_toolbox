import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

class Sin(nn.Module):
    """ Sin activation function"""
    def forward(self, x):
        return torch.sin(x)

class PINN(nn.Module):
    def __init__(self, 
                 data: dict , 
                 layers: list, 
                 num_params: int,
                 lr: float = 1e0,
                 act=nn.Tanh,
                 optimizer_type='lbfgs',
                 log_dir: str = "./tb_logs"):
        super().__init__()

        self.act = act
        self.layers = layers
        self.num_params = num_params
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.log_dir = log_dir

        self.parse_data(data)
        self.init_model()
        self.init_optimizers()

        # Get day month year hour minute second timestamp
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f'{self.log_dir}/{self.__class__.__name__}_{timestamp}')

    def parse_data(self, data):
        """ Parse the relevant data and assign variables to self"""
        raise NotImplementedError
    
    def init_model(self):
        """ Initialize neural network and learnable parameters """
        layer_list = []
        for i in range(len(self.layers)-1):
            layer_list.append(nn.Linear(self.layers[i], self.layers[i+1]))
            if i != len(self.layers)-2:
                layer_list.append(self.act())

        self.model = nn.Sequential(*layer_list)

        if self.num_params > 0:
            self.params = nn.Parameter(torch.randn(self.num_params), requires_grad=True)
            self.model.register_parameter('params', self.params)

        self.apply(init_weights)

    def init_optimizers(self):
        """ Initialize optimizers and learning rate scheduler """
        if self.optimizer_type == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                lr=self.lr,
                max_iter=50000,
                history_size=50,
                tolerance_grad=1e-9,
                tolerance_change=1.0 * np.finfo(float).eps,
                line_search_fn="strong_wolfe",
            )
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        elif self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = None
    
    def train(self, num_iter):
        """ Train the model using the appropriate optimizer """
        self.iter = 0
        while self.iter < num_iter:
            if self.optimizer_type == 'lbfgs':
                self.optimizer.step(self.loss_closure)
            else:
                self.loss_closure()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
    
    def mse_loss(self):
        """ Reconstruction loss """
        raise NotImplementedError

    def phys_loss(self):
        """ Loss for physical model """
        raise NotImplementedError

    def loss_closure(self):
        """ Compute the loss function and optionally print the model"""
        mse_loss = self.mse_loss()
        phys_loss = self.phys_loss()
        loss = mse_loss + phys_loss

        self.optimizer.zero_grad()
        loss.backward()
        
        self.iter += 1
        if self.iter % 10 == 0:
            self.log_tensorboard(loss, mse_loss, phys_loss)
        if self.iter % 1000 == 0:
            self.print(loss, mse_loss, phys_loss)
        
        return loss
    
    def log_tensorboard(self, loss, mse_loss, phys_loss):
        """ Log the loss to tensorboard """
        self.writer.add_scalar("Total Loss", loss, self.iter)
        self.writer.add_scalar("MSE Loss", mse_loss, self.iter)
        self.writer.add_scalar("Physics Loss", phys_loss, self.iter)
    
    def print(self, loss, mse_loss, phys_loss):
        """ Print the loss to console """
        print(f"Iteration {self.iter}, Loss: {loss.item():.5e}, MSE: {mse_loss.item():.5e}, Phys: {phys_loss.item():.5e}")
    
    def forward(self, x):
        """ Model prediction on scaled inputs"""
        H = 2. * (x - self.lb) / (self.ub - self.lb) - 1.
        return self.model(H)