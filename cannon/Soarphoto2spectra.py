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
from daep.data_util import PhotoSpectraDatasetFromnp, collate_fn_stack, to_device, padding_collate_fun
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore, spectraTransceiverScore2stages
from daep.daep import crossmodaldaep
from daep.tokenizers import photometryTokenizer, spectraTokenizer
from daep.Perceiver import PerceiverEncoder
from daep.PhotometricLayers import photometricTransceiverEncoder


import math 
import os
from tqdm import tqdm
from functools import partial







def train(epoch=1000, lr = 2.5e-4, 
          bottlenecklen = 16, bottleneckdim = 16, 
          concat = True, 
          model_dim = 128, encoder_layers = 4, 
          decoder_layers = 4,regularize = 0.000, 
          batch = 64, aug = 1, save_every = 20, 
          encoder_head = 4,
          score_head = 4,
          
          which_data = "soar"):
    assert which_data in ["soar", "gemini"]
    
    data = np.load(f'../data/{which_data}_dataset_full_minphot20_minspec80.npz')
    training_idx = data['training_idx']
    testing_idx = data['testing_idx']

    ### spectra ###
    flux, wavelength, mask = data['flux'][training_idx], data['wavelength'][training_idx], data['mask'][training_idx]
    phase = data['phase'][training_idx]
    
    photoflux, phototime, photoband = data['photoflux'][training_idx], data['photophase'][training_idx], data['photowavelength'][training_idx]
    photomask = data['photomask'][training_idx]

    flux = torch.tensor(flux, dtype=torch.float32)
    wavelength = torch.tensor(wavelength, dtype=torch.float32)
    mask = torch.tensor(mask == 1)
    phase = torch.tensor(phase, dtype=torch.float32)
    
    photoflux = torch.tensor(photoflux, dtype = torch.float32)
    phototime = torch.tensor(phototime, dtype = torch.float32)
    photoband = torch.tensor(photoband, dtype = torch.long)
    photomask = torch.tensor(photomask == 1)
    #breakpoint()
    #### do some data augmentation ####
    factor = aug
    flux = flux.repeat((factor,1))
    wavelength = wavelength.repeat((factor,1))
    mask = mask.repeat((factor,1))
    phase = phase.repeat((factor))
    
    
    photoflux = photoflux.repeat((factor, 1))
    phototime = phototime.repeat((factor, 1))
    photoband = photoband.repeat((factor, 1))
    photomask = photomask.repeat((factor, 1))
    
    
    flux = flux + torch.randn_like(flux) * 0.01 # some noise 
    phase = phase + torch.randn_like(phase) * 0.001
    mask = torch.logical_or(mask, torch.rand_like(flux)<=0.1) # do some random masking
    
    photoflux = photoflux + torch.randn_like(photoflux) * 0.01 # some noise 
    phototime = phototime + torch.randn_like(phototime) * 0.001
    photomask = torch.logical_or(photomask, torch.rand_like(photoflux)<=0.1) # do some random masking
    
    

    training_data = PhotoSpectraDatasetFromnp(flux, wavelength, phase, 
                 photoflux, phototime, photoband
                 ,mask, photomask)
    

    training_loader = DataLoader(training_data, batch_size = batch, 
                                 collate_fn = padding_collate_fun(supply = ['flux', 'wavelength', 'time', "band"], mask_by = "flux", multimodal = True))
    
    
    encoder = photometricTransceiverEncoder(
            
            num_bands = 6, 
            bottleneck_length = bottlenecklen,
            bottleneck_dim = bottleneckdim,
            model_dim = model_dim, 
            ff_dim = model_dim,
            num_heads = encoder_head
        )
    
    
    scores = spectraTransceiverScore(
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    ff_dim = model_dim,
                    num_layers = decoder_layers,
                    concat = concat,
                    num_heads = score_head
                    )
    

    mydaep = crossmodaldaep(encoder, scores, 
                            source_modality = "photometry",
                            target_modality = "spectra",
                            name = "flux",
                            query_name = "wavelength").to(device)
    
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
            #breakpoint()
            optimizer.zero_grad()
            loss = mydaep(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            #print(loss.item())
        this_epoch = np.array(losses).mean().item()
        epoch_loss.append(math.log(this_epoch))
        epoches.append(ep)
        if (ep+1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
            target_save = f"../ckpt/{which_data.upper()}photo_to_spectra_daep_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_heads-{encoder_head}-{score_head}_concat{concat}_lr{lr}_epoch{ep+1}_batch{batch}_reg{regularize}_aug{aug}.pth"
            torch.save(mydaep, target_save)
            plt.plot(epoches, epoch_loss)
            plt.show()
            plt.savefig(f"./logs/{which_data.upper()}photo_to_spectra_daep_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_heads-{encoder_head}-{score_head}_concat{concat}_lr{lr}_batch{batch}_reg{regularize}_aug{aug}.png")
            plt.close()
        progress_bar.set_postfix(loss=f"epochs:{ep}, {math.log(this_epoch):.4f}") 
    


import fire           

if __name__ == '__main__':
    fire.Fire(train)