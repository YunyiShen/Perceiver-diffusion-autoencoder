import torch
import glob
import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from daep.data_util import SpectraDatasetFromnp, collate_fn_stack, to_device
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore
from daep.daep import unimodaldaep
import math 
import os
from tqdm import tqdm



def train(epoch=1000, lr = 2.5e-4, bottlenecklen = 4, bottleneckdim = 4, 
          concat = True, cross_attn_only = False,
          model_dim = 128, encoder_layers = 4, 
          decoder_layers = 4,regularize = 0.0001, 
          batch = 64, aug = 5, save_every = 20):
    
    data = np.load("../data/train_data_align_with_simu_minimal.npz")

    ### spectra ###
    flux, wavelength, mask = data['flux'], data['wavelength'], data['mask']
    phase = data['phase']

    flux = torch.tensor(flux, dtype=torch.float32)
    wavelength = torch.tensor(wavelength, dtype=torch.float32)
    mask = torch.tensor(mask == 0)
    phase = torch.tensor(phase, dtype=torch.float32)

    #### do some data augmentation ####
    factor = aug
    flux = flux.repeat((factor,1))
    wavelength = wavelength.repeat((factor,1))
    mask = mask.repeat((factor,1))
    phase = phase.repeat((factor))

    

    flux = flux + torch.randn_like(flux) * 0.01 # some noise 
    phase = phase + torch.randn_like(phase) * 0.001
    mask = torch.logical_or(mask, torch.rand_like(flux)<=0.1) # do some random masking

    training_data = SpectraDatasetFromnp(flux, wavelength, phase, mask)

    training_loader = DataLoader(training_data, batch_size = batch, collate_fn = collate_fn_stack)
    
    spectraEncoder = spectraTransceiverEncoder(
                    bottleneck_length = bottlenecklen,
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    ff_dim = model_dim,
                    num_layers = encoder_layers,
                    concat = concat
                    ).to(device)

    spectraScore = spectraTransceiverScore(
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    ff_dim = model_dim,
                    num_layers = decoder_layers,
                    concat = concat,
                    cross_attn_only = cross_attn_only
                    ).to(device)


    mydaep = unimodaldaep(spectraEncoder, spectraScore, regularize = regularize).to(device)
    
    mydaep.train()
    optimizer = AdamW(mydaep.parameters(), lr=lr)
    epoch_loss = []
    epoches = []
    target_save = None
    progress_bar = tqdm(range(epoch))
    for ep in progress_bar:
        losses = []
        for x in training_loader:
            x = to_device(x)
            optimizer.zero_grad()
            loss = mydaep(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        this_epoch = np.array(losses).mean().item()
        epoch_loss.append(math.log(this_epoch))
        epoches.append(ep)
        if (ep+1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
            target_save = f"../ckpt/ZTFspectra_daep_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_concat{concat}_corrattnonly{cross_attn_only}_lr{lr}_epoch{ep+1}_batch{batch}_reg{regularize}_aug{aug}.pth"
            torch.save(mydaep, target_save)
            plt.plot(epoches, epoch_loss)
            plt.show()
            plt.savefig(f"./logs/ZTFspectra_daep_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_concat{concat}_corrattnonly{cross_attn_only}_lr{lr}_batch{batch}_reg{regularize}_aug{aug}.png")
            plt.close()
        progress_bar.set_postfix(loss=f"epochs:{ep}, {math.log(this_epoch):.4f}") 
    


import fire           

if __name__ == '__main__':
    fire.Fire(train)
