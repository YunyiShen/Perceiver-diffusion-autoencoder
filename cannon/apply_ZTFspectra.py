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
from daep.data_util import SpectraDatasetFromnp, collate_fn_stack, to_device, to_np_cpu, save_dictlist
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore
from daep.daep import unimodaldaep
import math 
import os
from tqdm import tqdm
import copy


data = np.load("../data/test_data_align_with_simu_minimal.npz")

### spectra ###
flux, wavelength, mask = data['flux'], data['wavelength'], data['mask']
phase = data['phase']


flux = torch.tensor(flux, dtype=torch.float32)
wavelength = torch.tensor(wavelength, dtype=torch.float32)
mask = torch.tensor(mask == 0)
phase = torch.tensor(phase, dtype=torch.float32)
#breakpoint()
flux_mean, flux_std = data['flux_mean'], data['flux_std']
wavelength_mean, wavelength_std = data['wavelength_mean'], data['wavelength_std']
phase_mean, phase_std = data['spectime_mean'], data['spectime_std']

test_data = SpectraDatasetFromnp(flux, wavelength, phase, mask)
torch.manual_seed(42)
test_loader = DataLoader(test_data, batch_size = 128, collate_fn = collate_fn_stack, shuffle = False)

ckpt = "ZTFspectra_daep_4-4-4-4-128_heads8-8_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_reg0.0_aug5"
trained_daep = torch.load(f"../ckpt/{ckpt}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)

size = 50

for i, x in tqdm(enumerate(test_loader)):


    x = to_device(x)
    x_ori = copy.deepcopy(x)


    torch.manual_seed(12345)
    #x_ori = to_np_cpu(x_ori)
    #recon = to_np_cpu(recon)


    
    rec = []

    for j in range(size):
        x = copy.deepcopy(x_ori)
        recon = trained_daep.reconstruct(x, ddim_steps = 200)
        recon = to_np_cpu(recon)
        recon['flux'] = recon['flux'] * flux_std + flux_mean
    
        rec.append(recon)
    x_ori = to_np_cpu(x_ori)
    x_ori['wavelength'] = x_ori['wavelength'] * wavelength_std + wavelength_mean
    x_ori['flux'] = x_ori['flux'] * flux_std + flux_mean 

    save_dictlist(f"./res/ZTFspectra/{ckpt}_rec_batch{i}.npz", rec)
    np.savez(f"./res/ZTFspectra/{ckpt}_gt_batch{i}.npz",
         **x_ori
         )

'''
fig, axes = plt.subplots(4, 5, figsize=(20, 12))  # 4 rows, 5 columns
axes = axes.flatten()
for i in range(20):
    axes[i].plot(x_ori['wavelength'][i][~x_ori['mask'][i]] * wavelength_std + wavelength_mean,
                 x_ori['flux'][i][~x_ori['mask'][i]] * flux_std + flux_mean,
                 color = "red",
                 label = "ground truth" if i == 0 else None
                 )
    axes[i].plot(recon['wavelength'][i][~recon['mask'][i]] * wavelength_std + wavelength_mean,
                 recon['flux'][i][~recon['mask'][i]] * flux_std + flux_mean,
                 color = "blue",
                 label = "daep" if i == 0 else None
                 )
    axes[i].set_xlabel("wavelength (A)")


axes[0].legend()
axes[0].set_ylabel("logFnu")
axes[5].set_ylabel("logFnu")
plt.tight_layout()
fig.show()
fig.savefig("spectra_recon.pdf")
plt.close()

'''
