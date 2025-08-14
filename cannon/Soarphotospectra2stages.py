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
from daep.daep import multimodaldaep, modality_drop
from daep.tokenizers import photometryTokenizer, spectraTokenizer
from daep.Perceiver import PerceiverEncoder
from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore


import math 
import os
from tqdm import tqdm
from functools import partial







def train(epoch=1000, lr = 2.5e-4, bottlenecklen = 4, bottleneckdim = 4, 
          concat = True, 
          mixer_selfattn = False,
          spectra_tokens = 64,
          photometry_tokens = 64,
          model_dim = 128, encoder_layers = 4, 
          decoder_layers = 4,regularize = 0.000, 
          persample_dropping = False,
          dropping_prob = 0.2,
          batch = 64, aug = 1, save_every = 20, 
          tokenizer_head = 4,
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
    
    
    tokenizers = {
        "spectra": spectraTransceiverEncoder(
            bottleneck_length = spectra_tokens,
            bottleneck_dim = model_dim,
            model_dim = model_dim,
            ff_dim = model_dim,
            num_heads = tokenizer_head
        ), 
        "photometry": photometricTransceiverEncoder(
            
            num_bands = 6, 
            bottleneck_length = photometry_tokens,
            bottleneck_dim = model_dim,
            model_dim = model_dim, 
            ff_dim = model_dim,
            num_heads = tokenizer_head
        )
    }
    
    encoder = PerceiverEncoder(
                    bottleneck_length = bottlenecklen,
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    num_layers = encoder_layers,
                    ff_dim = model_dim,
                    num_heads = encoder_head,
                    selfattn = mixer_selfattn
    )
    
    
    scores = {
        "spectra":spectraTransceiverScore(
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    ff_dim = model_dim,
                    num_layers = decoder_layers,
                    concat = concat,
                    num_heads = score_head
                    ), 
        "photometry": photometricTransceiverScore(
            bottleneck_dim = bottleneckdim,
                 num_bands = 6,
                 model_dim = model_dim,
                 ff_dim = model_dim,
                 num_layers = decoder_layers,
                 concat = concat,
                 num_heads = score_head
        )
    }

    

    if persample_dropping:
        mydaep = multimodaldaep(tokenizers, encoder, scores, 
                            measurement_names = {"spectra":"flux", "photometry": "flux"}, 
                            modality_dropping_during_training = None,
                            persample_dropping_p = dropping_prob,
                            ).to(device)
    
    else:

        mydaep = multimodaldaep(tokenizers, encoder, scores, 
                            measurement_names = {"spectra":"flux", "photometry": "flux"}, 
                            modality_dropping_during_training = partial(modality_drop, p_drop=dropping_prob)).to(device)
     
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
            print(loss.item())
        this_epoch = np.array(losses).mean().item()
        epoch_loss.append(math.log(this_epoch))
        epoches.append(ep)
        if (ep+1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
            target_save = f"../ckpt/{which_data.upper()}photospectra_daep2stages_{bottlenecklen}-{bottleneckdim}-{spectra_tokens}-{photometry_tokens}-{encoder_layers}-{decoder_layers}-{model_dim}_heads{tokenizer_head}-{encoder_head}-{score_head}_concat{concat}_mixerselfattn{mixer_selfattn}_lr{lr}_persampledrop{persample_dropping}_modaldropP{dropping_prob}_epoch{ep+1}_batch{batch}_reg{regularize}_aug{aug}.pth"
            torch.save(mydaep, target_save)
            plt.plot(epoches, epoch_loss)
            plt.show()
            plt.savefig(f"./logs/{which_data.upper()}photospectra_daep2stages_{bottlenecklen}-{bottleneckdim}-{spectra_tokens}-{photometry_tokens}-{encoder_layers}-{decoder_layers}-{model_dim}_heads{tokenizer_head}-{encoder_head}-{score_head}_concat{concat}_mixerselfattn{mixer_selfattn}_lr{lr}_persampledrop{persample_dropping}_modaldropP{dropping_prob}_batch{batch}_reg{regularize}_aug{aug}.png")
            plt.close()
        progress_bar.set_postfix(loss=f"epochs:{ep}, {math.log(this_epoch):.4f}") 
    


import fire           

if __name__ == '__main__':
    fire.Fire(train)