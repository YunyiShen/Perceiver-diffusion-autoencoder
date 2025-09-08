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
from daep.data_util import PhotoSpectraDatasetFromnp, collate_fn_stack, to_device, padding_collate_fun, to_device, to_np_cpu
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore
from daep.daep import multimodaldaep, modality_drop
from daep.tokenizers import photometryTokenizer, spectraTokenizer
from daep.Perceiver import PerceiverEncoder
from daep.PhotometricLayers import photometricTransceiverScore
import copy
from daep.plot_util import plot_lsst_lc

import math 
import os
from tqdm import tqdm

which_data = "soar"
data = np.load(f'../data/{which_data}_dataset_full_minphot20_minspec80.npz')
training_idx = data['training_idx']
testing_idx = data['testing_idx']
#breakpoint()
    ### spectra ###
flux, wavelength, mask = data['flux'][testing_idx], data['wavelength'][testing_idx], data['mask'][testing_idx]
phase = data['phase'][testing_idx]
    
photoflux, phototime, photoband = data['photoflux'][testing_idx], data['photophase'][testing_idx], data['photowavelength'][testing_idx]
photomask = data['photomask'][testing_idx]

flux_mean, flux_std = data['flux_mean'], data['flux_std']
wavelength_mean, wavelength_std = data['wavelength_mean'], data['wavelength_std']

photoflux_mean, photoflux_std = data['photoflux_mean'], data['photoflux_std']
phototime_mean, phototime_std = data['photophase_mean'], data['photophase_std']



flux = torch.tensor(flux, dtype=torch.float32)
wavelength = torch.tensor(wavelength, dtype=torch.float32)
mask = torch.tensor(mask == 1)
phase = torch.tensor(phase, dtype=torch.float32)
    
photoflux = torch.tensor(photoflux, dtype = torch.float32)
phototime = torch.tensor(phototime, dtype = torch.float32)
photoband = torch.tensor(photoband, dtype = torch.long)
photomask = torch.tensor(photomask == 1)
#breakpoint()

    


test_data = PhotoSpectraDatasetFromnp(flux, wavelength, phase, 
                 photoflux, phototime, photoband
                 ,mask, photomask)
torch.manual_seed(42)
test_loader = DataLoader(test_data, batch_size = 20, 
                                 collate_fn = padding_collate_fun(supply = ['flux', 'wavelength', 'time', "band"], mask_by = "flux", multimodal = True), shuffle = True)

x = to_device(next(iter(test_loader)))
x_ori = copy.deepcopy(x)

trained_daep = torch.load("../ckpt/SOARphoto_to_spectra_daep_16-16-4-4-256_heads-4-4_concatTrue_lr0.00025_epoch1000_batch256_reg0.0_aug1.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)


torch.manual_seed(1236)
recon = trained_daep.inference(x)
x_ori = to_np_cpu(x_ori)
recon = to_np_cpu(recon)
#breakpoint()

fig, axes = plt.subplots(4, 5, figsize=(20, 12))  # 4 rows, 5 columns
axes = axes.flatten()
for i in range(20):
    axes[i].plot(recon['wavelength'][i][~recon['mask'][i]] * wavelength_std + wavelength_mean,
                 recon['flux'][i][~recon['mask'][i]] * flux_std + flux_mean,
                 color = "blue",
                 label = "daep" if i == 0 else None
                 )
    axes[i].plot(x_ori['spectra']['wavelength'][i][~x_ori['spectra']['mask'][i]] * wavelength_std + wavelength_mean,
                 x_ori['spectra']['flux'][i][~x_ori['spectra']['mask'][i]] * flux_std + flux_mean,
                 color = "red",
                 label = "ground truth" if i == 0 else None
                 )
    
    axes[i].set_xlabel("wavelength (A)")


axes[0].legend()
axes[0].set_ylabel("logFnu")
axes[5].set_ylabel("logFnu")
plt.tight_layout()
fig.show()
fig.savefig(f"{which_data}_spectra_recon_fromLC_directly.pdf")
plt.close()

