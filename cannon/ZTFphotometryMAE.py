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
from daep.data_util import PhotoDatasetFromnp, collate_fn_stack, to_device
from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverMAEDecoder
from daep.mae import unimodalmae
import math 
import os
from tqdm import tqdm



def train(epoch=1000, lr = 2.5e-4, bottlenecklen = 4, bottleneckdim = 4, 
          concat = True, cross_attn_only = False,
          model_dim = 128, encoder_layers = 4, 
          decoder_layers = 4,mask_rate = 0.3, 
          batch = 64, aug = 5, save_every = 20):
    
    data = np.load("../data/train_data_align_with_simu_minimal.npz")
    factor = aug
    ### photometric ###
    
    photoflux, phototime, photoband = data['photoflux'], data['phototime'], data['photowavelength']
    photomask = data['photomask']

    
    photoflux = torch.tensor(photoflux, dtype = torch.float32)
    phototime = torch.tensor(phototime, dtype = torch.float32)
    photoband = torch.tensor(photoband, dtype = torch.long)
    photomask = torch.tensor(photomask == 0)
    
    photoflux = photoflux.repeat((factor, 1))
    phototime = phototime.repeat((factor, 1))
    photoband = photoband.repeat((factor, 1))
    photomask = photomask.repeat((factor, 1))
    

    
    photoflux = photoflux + torch.randn_like(photoflux) * 0.01 # some noise 
    phototime = phototime + torch.randn_like(phototime) * 0.001
    photomask = torch.logical_or(photomask, torch.rand_like(photoflux)<=0.1) # do some random masking


    training_data = PhotoDatasetFromnp(photoflux, phototime, photoband, photomask)

    training_loader = DataLoader(training_data, batch_size = batch, collate_fn = collate_fn_stack)
    
    photometricEncoder = photometricTransceiverEncoder(
                    num_bands = 2, 
                    bottleneck_length = bottlenecklen,
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    ff_dim = model_dim,
                    num_layers = encoder_layers,
                    concat = concat
                    ).to(device)

    photometricDecoder = photometricTransceiverMAEDecoder(
                    num_bands = 2, 
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    ff_dim = model_dim,
                    num_layers = decoder_layers,
                    concat = concat,
                    cross_attn_only = cross_attn_only
                    ).to(device)


    mymae = unimodalmae(photometricEncoder, photometricDecoder, mask_rate).to(device)
    
    mymae.train()
    optimizer = AdamW(mymae.parameters(), lr=lr)
    epoch_loss = []
    epoches = []
    target_save = None
    progress_bar = tqdm(range(epoch))
    for ep in progress_bar:
        losses = []
        for x in training_loader:
            x = to_device(x)
            optimizer.zero_grad()
            loss = mymae(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        this_epoch = np.array(losses).mean().item()
        epoch_loss.append(math.log(this_epoch))
        epoches.append(ep)
        if (ep+1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
            target_save = f"../ckpt/ZTFphotometric_mae_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_concat{concat}_corrattnonly{cross_attn_only}_lr{lr}_epoch{ep+1}_batch{batch}_mask{mask_rate}_aug{aug}.pth"
            torch.save(mymae, target_save)
            plt.plot(epoches, epoch_loss)
            plt.show()
            plt.savefig(f"./logs/ZTFphotometric_mae_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_concat{concat}_corrattnonly{cross_attn_only}_lr{lr}_batch{batch}_mask{mask_rate}_aug{aug}.png")
            plt.close()
        progress_bar.set_postfix(loss=f"epochs:{ep}, {math.log(this_epoch):.4f}") 
    


import fire           

if __name__ == '__main__':
    fire.Fire(train)
