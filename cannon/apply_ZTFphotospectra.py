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
from daep.plot_util import plot_ztf_lc

import math 
import os
from tqdm import tqdm


data = np.load("../data/test_data_align_with_simu_minimal.npz")

### spectra ###
flux, wavelength, mask = data['flux'], data['wavelength'], data['mask']
phase = data['phase']
    
photoflux, phototime, photoband = data['photoflux'], data['phototime'], data['photowavelength']
photomask = data['photomask']

flux_mean, flux_std = data['flux_mean'], data['flux_std']
wavelength_mean, wavelength_std = data['wavelength_mean'], data['wavelength_std']
phase_mean, phase_std = data['spectime_mean'], data['spectime_std']

photoflux_mean, photoflux_std = data['combined_mean'], data['combined_std']
phototime_mean, phototime_std = data['combined_time_mean'], data['combined_time_std']

flux = torch.tensor(flux, dtype=torch.float32)
wavelength = torch.tensor(wavelength, dtype=torch.float32)
mask = torch.tensor(mask == 0)
phase = torch.tensor(phase, dtype=torch.float32)
    
photoflux = torch.tensor(photoflux, dtype = torch.float32)
phototime = torch.tensor(phototime, dtype = torch.float32)
photoband = torch.tensor(photoband, dtype = torch.long)
photomask = torch.tensor(photomask == 0)

test_data = PhotoSpectraDatasetFromnp(flux, wavelength, phase, 
                 photoflux, phototime, photoband
                 ,mask, photomask)
torch.manual_seed(123456)
test_loader = DataLoader(test_data, batch_size = 20, 
                                 collate_fn = padding_collate_fun(supply = ['flux', 'wavelength', 'time', "band"], mask_by = "flux", multimodal = True), shuffle = True)

x = to_device(next(iter(test_loader)))
x_ori = copy.deepcopy(x)

trained_daep = torch.load("../ckpt/ZTFphotospectra_daep2stages_8-8-64-64-4-4-256_concatTrue_mixerselfattnTrue_lr0.00025_persampledropFalse_modaldropP0.5_epoch2000_batch256_reg0.0_aug5.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)


torch.manual_seed(123)
recon = trained_daep.reconstruct(x, condition_keys = ["spectra"], out_keys = ['spectra', "photometry"])
x_ori = to_np_cpu(x_ori)
recon = to_np_cpu(recon)

fig, axes = plt.subplots(4, 5, figsize=(20, 12))  # 4 rows, 5 columns
axes = axes.flatten()
for i in range(20):
    axes[i].plot(x_ori['spectra']['wavelength'][i][~x_ori['spectra']['mask'][i]] * wavelength_std + wavelength_mean,
                 x_ori['spectra']['flux'][i][~x_ori['spectra']['mask'][i]] * flux_std + flux_mean,
                 color = "red",
                 label = "ground truth" if i == 0 else None
                 )
    axes[i].plot(recon['spectra']['wavelength'][i][~recon['spectra']['mask'][i]] * wavelength_std + wavelength_mean,
                 recon['spectra']['flux'][i][~recon['spectra']['mask'][i]] * flux_std + flux_mean,
                 color = "blue",
                 label = "daep" if i == 0 else None
                 )
    axes[i].set_xlabel("wavelength (A)")


axes[0].legend()
axes[0].set_ylabel("logFnu")
axes[5].set_ylabel("logFnu")
plt.tight_layout()
fig.show()
fig.savefig("ZTFspectra_recon_fromspectra.pdf")
plt.close()

fig, axes = plt.subplots(3, 10, figsize=(20, 6))  # 4 rows, 5 columns
#axes = axes.flatten()
for i in range(10):
    oritime = x_ori['photometry']['time'][i] * phototime_std + phototime_mean
    oriflux = x_ori['photometry']['flux'][i] * photoflux_std + photoflux_mean
    oriband = x_ori['photometry']['band'][i]
    orimask = x_ori['photometry']['mask'][i]
    
    recflux = recon['photometry']['flux'][i] * photoflux_std + photoflux_mean
    
    
    plot_ztf_lc(oriband, oriflux, oritime, orimask, ax = axes[0, i], label = False, s = 15, lw = 2)
    plot_ztf_lc(oriband, recflux, oritime, orimask, ax = axes[1, i], label = False, s = 15, lw = 2)
    plot_ztf_lc(oriband, (recflux-oriflux)/oriflux, oritime, orimask, ax = axes[2, i], label = False, s = 15, lw = 2)
    axes[2,i].set_ylim(-0.15, 0.15)


plt.tight_layout()
fig.show()
fig.savefig("ZTFLC_recon_fromspectra.pdf")
plt.close()

