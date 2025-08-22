import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from matplotlib import pyplot as plt
import os


from VAESNe.SpectraVAE import BrightSpectraVAE, SpectraVAE
from VAESNe.training_util import training_step
from VAESNe.losses import elbo, m_iwae, _m_iwae
from VAESNe.data_util import multimodalDataset
from VAESNe.mmVAE import photospecMMVAE
from tqdm import tqdm

torch.manual_seed(0)

### dataset ###

def train(aug = 5, 
    batch_size=16,
    lr = 1e-3, #2.5e-4
    epochs = 200,
    latent_len = 4,
    latent_dim = 4,
    beta = 0.5,
    model_dim = 32,
    num_heads = 4,
    num_layers = 4,
    save_every = 20
    ):

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
    mask = torch.logical_or(mask, torch.rand_like(flux)<=0.05) # do some random masking



    spectra_train_dataset = TensorDataset(flux, wavelength, phase, mask)

    train_loader = DataLoader(spectra_train_dataset, batch_size=batch_size, shuffle=True)
    

    my_spectravae = SpectraVAE(
        # model parameters
        latent_len = latent_len,
        latent_dim = latent_dim,
        model_dim = model_dim, 
        num_heads = num_heads, 
        ff_dim = model_dim, 
        num_layers = num_layers,
        dropout = 0.1,
        selfattn = False, #True
        beta = beta
        ).to(device)


    optimizer = AdamW(my_spectravae.parameters(), lr=lr)
    all_losses = np.ones(epochs) + np.nan
    steps = np.arange(epochs)

    target_save = None
    progress_bar = tqdm(range(epochs))
    for i in progress_bar:
        loss = training_step(my_spectravae, optimizer,train_loader, 
                    loss_fn = elbo, 
                    multimodal = False)
        all_losses[i] = loss
        if (i + 1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
                
            target_save = f'../ckpt/ZTF_spectra_vaesne_{latent_len}-{latent_dim}-{model_dim}-{num_layers}_heads{num_heads}_{lr}_epoch{i+1}_batch{batch_size}_aug{aug}_beta{beta}.pth'
            plt.plot(steps, all_losses)
            plt.xlabel("training epochs")
            plt.ylabel("loss")
            plt.show()
            plt.savefig(f"./logs/ZTF_spectra_vaesne_{latent_len}-{latent_dim}-{model_dim}-{num_layers}_heads{num_heads}_{lr}_batch{batch_size}_aug{aug}_beta{beta}.png")
            plt.close()
            torch.save(my_spectravae, target_save)
        progress_bar.set_postfix(loss=f"epochs:{i}, {loss:.4f}")


import fire           

if __name__ == '__main__':
    fire.Fire(train)