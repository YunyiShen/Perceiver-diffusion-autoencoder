import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'


from VAESNe.PhotometricVAE import PhotometricVAE
from VAESNe.training_util import training_step
from VAESNe.losses import elbo
from tqdm import tqdm
import os

def train(
    aug = 5,
    lr = 2.5e-4,
    batch_size = 128,
    epochs = 1000,
    latent_len = 4,
    latent_dim = 2,
    model_dim = 32, 
    num_heads = 4, 
    num_layers = 4,
    beta = 0.5,
    save_every = 20
):
    


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


    # split loaded data into training and validation
    train_dataset = TensorDataset(photoflux, phototime, photoband, photomask)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    my_vaesne = PhotometricVAE(
        num_bands = 2,
        # model parameters
        latent_len = latent_len,
        latent_dim = latent_dim,
        model_dim = model_dim, 
        num_heads = num_heads, 
        ff_dim = model_dim, 
        num_layers = num_layers,
        dropout = 0.1,
        selfattn = False,#True
        beta = beta
        ).to(device)

    optimizer = AdamW(my_vaesne.parameters(), lr=lr)
    all_losses = np.ones(epochs) + np.nan
    steps = np.arange(epochs)

    progress_bar = tqdm(range(epochs))
    target_save = None
    for i in progress_bar:
        loss = training_step(my_vaesne, optimizer, train_loader, elbo)
        all_losses[i] = loss
        if (i + 1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
            target_save = f'../ckpt/ZTF_photometry_vaesne_{latent_len}-{latent_dim}-{model_dim}-{num_layers}_heads{num_heads}_{lr}_epoch{i+1}_batch{batch_size}_aug{aug}_beta{beta}.pth'
        
            plt.plot(steps, all_losses)
            plt.show()
            plt.savefig(f"./logs/ZTF_photometry_vaesne_{latent_len}-{latent_dim}-{model_dim}-{num_layers}_heads{num_heads}_{lr}_batch{batch_size}_aug{aug}_beta{beta}.png")
            plt.close()
            torch.save(my_vaesne, target_save)
        progress_bar.set_postfix(loss=f"epochs:{i}, {loss:.4f}")
        
        
import fire           

if __name__ == '__main__':
    fire.Fire(train)