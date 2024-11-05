from dolfin_pbnn import DolfinPBNN
from data_processing import HDF5Dataset
import torch
import numpy as np

from tqdm import tqdm
from time import time
import logging
import dolfin as dlf
import dolfin_adjoint as d_ad

# Turn off annoying log messages
logging.getLogger('FFC').setLevel(logging.ERROR)
dlf.set_log_level(40)

if __name__ == '__main__':
    # Load dataset
    dataset = HDF5Dataset()
    print('Dataset length:', len(dataset))

    gen = torch.Generator()
    gen.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=gen)
    print('Train dataset length:', len(train_dataset))
    print('Val dataset length:', len(val_dataset), flush=True)

    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DolfinPBNN().to(device)

    # Build optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-7)

    train_idxs = np.arange(len(train_dataset), dtype=int)
    num_epochs = 20
    batch_size = 16

    best_loss = 1e10

    for epoch in range(500):
        np.random.shuffle(train_idxs)
        d_ad.set_working_tape(d_ad.Tape())

        train_loss = 0.
        t = time()

        for i in range(len(train_dataset)):
            sample = train_dataset[train_idxs[i]]
            sample['inputs'] = sample['inputs'].to(device)

            force, loss = model.training_step(sample)
            train_loss += loss / len(train_dataset)
            
            if i % batch_size == 0:
                optimizer.step()
                d_ad.set_working_tape(d_ad.Tape())
                optimizer.zero_grad()
        
        with torch.no_grad():
            val_loss = 0.
            for i in range(len(val_dataset)):
                sample = val_dataset[i]
                sample['inputs'] = sample['inputs'].to(device)

                force, loss = model.validation_step(sample)
                val_loss += loss / len(val_dataset)

        scheduler.step(val_loss)
        print(f'Epoch {epoch}\tTrain Loss = {train_loss:.3g}\tVal Loss = {val_loss:.3g}\t{time()-t:.3g} s', flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            save_dict = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
            }
            torch.save(save_dict, f'../models/poisson_weight.ckpt')

