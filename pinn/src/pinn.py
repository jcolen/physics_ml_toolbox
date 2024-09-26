import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

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
                 act=Sin,
                 log_dir: str = "./tb_logs"):
        super().__init__()

        self.act = act
        self.layers = layers
        self.num_params = num_params
        self.lr = lr
        self.log_dir = log_dir

        self.parse_data(data)
        self.init_model()
        self.init_optimizers()

        self.writer = SummaryWriter(log_dir=self.log_dir)

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
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.lr,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_lbfgs, gamma=0.95)
    
    def train(self, lbfgsIter):
        """ Train the model using LBFGS optimizer """
        self.iter = 0
        self.optimizer = self.optimizer_lbfgs
        for i in range(lbfgsIter):
            self.optimizer.step(self.loss_func)
            self.scheduler.step()

    def loss_func(self):
        """ Compute the loss function and optionally print the model"""
        _, mse, phys = self.predict_with_loss()
        loss = mse + phys
        self.optimizer.zero_grad()
        loss.backward()
        
        self.iter += 1
        if self.iter % 10 == 0:
            self.log_tensorboard(loss, mse, phys)
        if self.iter % 1000 == 0:
            self.print(loss, mse, phys)
        
        return loss
    
    def log_tensorboard(self, loss, mse, phys):
        """ Log the loss to tensorboard """
        self.writer.add_scalar("Total Loss", loss, self.iter)
        self.writer.add_scalar("MSE Loss", mse, self.iter)
        self.writer.add_scalar("Physics Loss", phys, self.iter)
    
    def print(self, loss, mse, phys):
        """ Print the loss to console """
        print(f"Iteration {self.iter}, Loss: {loss.item():.5e}, MSE: {mse.item():.5e}, Phys: {phys.item():.5e}")

    def predict_with_loss(self):
        """ Compute the predictions and losses """
        raise NotImplementedError
    
    def forward(self, x):
        """ Model prediction on scaled inputs"""
        H = 2. * (x - self.lb) / (self.ub - self.lb) - 1.
        return self.model(H)