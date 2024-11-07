import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np

from data_processing import HDF5Dataset
import dolfin_pbnn
import torch_pbnn

import yaml
import os
from tqdm import tqdm
from time import time
from argparse import ArgumentParser

import dolfin as dlf
import dolfin_adjoint as d_ad

# Turn off annoying log messages
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('FFC').setLevel(logging.ERROR)
logging.getLogger('UFL').setLevel(logging.ERROR)
dlf.set_log_level(40)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_dataset(config, random_seed=42):
    logger.info(f'Loading dataset from {config["path"]}')
    dataset = HDF5Dataset(**config)

    gen = torch.Generator()
    gen.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=gen)
    logger.info(f'Train dataset length: {len(train_dataset)}')
    logger.info(f'Val dataset length: {len(val_dataset)}')

    return train_dataset, val_dataset

def get_model(config):
    class_type = eval(config['class_path'])
    logger.info(f'Building a {class_type.__name__}')
    model = class_type(**config['args'])
    
    if 'weights' in config and config['weights'] is not None:
        logger.info(f'Loading model weights from {config["weights"]}')
        info = torch.load(config['weights'], map_location='cpu')
        model.load_state_dict(info['state_dict'])

        logger.info(f'Model reached loss={info["loss"]:.3g} at epoch {info["epoch"]:d}')
    
    return model

def get_optimizer_scheduler(config, model):
    class_type = eval(config['optimizer']['class_path'])
    logger.info(f'Building a {class_type.__name__}')
    optimizer = class_type(model.parameters(), **config['optimizer']['args'])
    if 'scheduler' in config:
        class_type = eval(config['scheduler']['class_path'])
        logger.info(f'Building a {class_type.__name__}')
        scheduler = class_type(optimizer, **config['scheduler']['args'])
    else:
        logger.info('Proceeding with no learning rate scheduler')
        scheduler = None
    
    return optimizer, scheduler

def run_training(config):
    # Load model
    model = get_model(config['model'])

    # Load datasets
    train, val = get_dataset(config['dataset'])

    # Load optimizer and scheduler
    optimizer, scheduler = get_optimizer_scheduler(config, model)

    # Dump configuration with model weight save path
    config['model']['weights'] = f'{config["save_path"]}/model_weight.ckpt'
    os.makedirs(config['save_path'], exist_ok=True)
    with open(f'{config["save_path"]}/config.yaml', 'w') as file:
        yaml.dump(config, file)

    # Execute training loop
    train_idxs = np.arange(len(train), dtype=int)
    epochs = config['training'].get('epochs', 500)
    batch_size = config['training'].get('batch_size', 16)

    best_loss = 1e10

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model.to(device)

    logger.info(f'Starting to train on device {device}')

    for epoch in range(epochs):
        np.random.shuffle(train_idxs)

        train_loss = 0.
        t = time()

        for i in range(len(train)):
            d_ad.set_working_tape(d_ad.Tape())
            sample = train[train_idxs[i]]
            sample['inputs'] = torch.FloatTensor(sample['inputs'])
            sample['inputs'] = sample['inputs'].to(device)

            force, loss = model.training_step(sample)
            train_loss += loss / len(train)
            
            if i % batch_size == 0:
                clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
        
        model.eval()
        with torch.no_grad():
            val_loss = 0.
            for i in range(len(val)):
                d_ad.set_working_tape(d_ad.Tape())
                sample = val[i]
                sample['inputs'] = torch.FloatTensor(sample['inputs'])
                sample['inputs'] = sample['inputs'].to(device)

                force, loss = model.validation_step(sample)
                val_loss += loss / len(val)

        scheduler.step()
        logger.info(f'Epoch {epoch}\tTrain Loss = {train_loss:.3g}\tVal Loss = {val_loss:.3g}\t{time()-t:.3g} s')

        if val_loss < best_loss:
            best_loss = val_loss
            save_dict = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
            }
            torch.save(save_dict, f'{config["save_path"]}/model_weight.ckpt')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, default='../configs/poisson_config.yaml')
    args = parser.parse_args()
    
    with open(args.config_file) as file:
        config = yaml.safe_load(file)
    
    logger.info(f'Loading configuration from {args.config_file}')
    run_training(config)
