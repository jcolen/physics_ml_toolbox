import numpy as np
from scipy.io import loadmat
from argparse import ArgumentParser
import logging
import warnings
import datetime
import yaml
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import data_processing
import fno_1d
import fno_1p1d
import fno_2d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def add_noise(x, noise=0.01):
    #Add some random noise to prevent focus on microscopic perturbations
    rmse = x.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
    return x + torch.randn_like(x) * rmse * noise

def get_dataset(config):
    logger.info(f'Loading dataset using {config["method_path"]}')
    train, val = eval(config["method_path"])(**config["args"])
    return train, val

def get_model(config):
    class_type = eval(config['class_path'])
    logger.info(f'Building a {class_type.__name__}')
    model = class_type(**config['args'])
    
    if 'weights' in config and config['weights'] is not None:
        logger.info(f'Loading model weights from {config["weights"]}')
        info = torch.load(config['weights'])
        model.load_state_dict(info['state_dict'])

        logger.info(f'Model reached loss={info["loss"]:.3g} at epoch {info["epoch"]:d}')
        
    return model

def iterate_loader(model, loader, device, optimizer=None):
    running_loss = 0.
    for sample in loader:
        x, y0 = sample
        x = add_noise(x).to(device)
        y0 = add_noise(y0).to(device)

        x = x[:, :model.in_channels]
        y = model(x)
        mse = F.mse_loss(y, y0)

        running_loss += mse.item() / len(loader)
        loss = mse

        if model.training and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return running_loss    

def train_model(model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                save_path,
                num_epochs):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model.to(device)
    
    best_loss = 1e10
    for epoch in range(num_epochs):
        model.train()
        loss = iterate_loader(model, train_loader, device, optimizer)

        outstr = f'Train Loss={loss:.3g}'
        logger.info(f'Epoch={epoch:d}/{num_epochs:d}\t{outstr}')

        if val_loader is not None:
            with torch.no_grad():
                model.eval()
                loss = iterate_loader(model, val_loader, device)
                outstr = f'Val Loss={loss:.3g}'
                logger.info(f'\t\t{outstr}')

        if scheduler is not None:
            scheduler.step(loss)

        if loss < best_loss:
            best_loss = loss
            save_dict = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
            }
            torch.save(save_dict, f'{save_path}/model_weight.ckpt')

def run_training(config):
    #Load model
    model = get_model(config['model'])   
    
    #Load datasets
    train, val = get_dataset(config['dataset'])
    train_loader = torch.utils.data.DataLoader(train, **config['loader'])
    if val is not None:
        val_loader = torch.utils.data.DataLoader(val, **config['loader'])
    else:
        val_loader = None
        
    #Load optimizer and scheduler
    class_type = eval(config['optimizer']['class_path'])
    logger.info(f'Building a {class_type.__name__}')
    optimizer = class_type(model.parameters(), **config['optimizer']['args'])
    if 'scheduler' in config:
        class_type = eval(config['scheduler']['class_path'])
        logger.info(f'Building a {class_type.__name__}')
        scheduler = class_type(optimizer, **config['scheduler']['args'])
    else:
        logger.info('Proceeding with no learning rate scheduler')

    config['model']['weights'] = f'{config["save_path"]}/model_weight.ckpt'

    os.makedirs(config['save_path'], exist_ok=True)
    with open(f'{config["save_path"]}/config.yaml', 'w') as file:
        yaml.dump(config, file)
    
    train_model(model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                config['save_path'],
                **config['training'])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, default='../configs/burgers_config_2d.yaml')
    args = parser.parse_args()
    
    with open(args.config_file) as file:
        config = yaml.safe_load(file)
    
    logger.info(f'Loading configuration from {args.config_file}')
    
    run_training(config)