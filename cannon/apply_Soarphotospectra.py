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
#breakpoint()
    
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
torch.manual_seed(12345)
test_loader = DataLoader(test_data, batch_size = 20, 
                                 collate_fn = padding_collate_fun(supply = ['flux', 'wavelength', 'time', "band"], mask_by = "flux", multimodal = True), shuffle = False)
import itertools
#breakpoint()
k = 850
x = to_device(next(itertools.islice(iter(test_loader), k, None))) # 8
x_ori = copy.deepcopy(x)
photerrs = data['photoerror'][testing_idx][(k*20):((k+1)*20)]
#breakpoint()
trained_daep = torch.load("../ckpt/SOARphotospectra_daep2stages_4-8-64-64-4-4-256_heads16-8-16_concatTrue_mixerselfattnTrue_lr0.00025_modaldropP0.2_epoch880_batch256_reg0.0_aug1.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)


torch.manual_seed(1236)
recon = trained_daep.reconstruct(x, condition_keys = ["photometry"], out_keys = ['spectra', "photometry"])
x_ori = to_np_cpu(x_ori)
recon = to_np_cpu(recon)

fig, axes = plt.subplots(4, 5, figsize=(20, 12))  # 4 rows, 5 columns
axes = axes.flatten()
for i in range(20):
    print(x_ori['spectra']['phase'][i])
    axes[i].plot(recon['spectra']['wavelength'][i][~recon['spectra']['mask'][i]] * wavelength_std + wavelength_mean,
                 recon['spectra']['flux'][i][~recon['spectra']['mask'][i]] * flux_std + flux_mean,
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
fig.savefig(f"{which_data}_spectra_recon_fromLC.pdf")
plt.close()

fig, axes = plt.subplots(2, 20, figsize=(55, 6))  # 4 rows, 5 columns

for i in range(20):
    oritime = x_ori['photometry']['time'][i] * phototime_std + phototime_mean
    sorttime = np.argsort(oritime)
    oritime = oritime[sorttime]
    oriflux =x_ori['photometry']['flux'][i][sorttime] * photoflux_std + photoflux_mean #np.sinh(x_ori['flux'][i][sorttime] * 5.6163 + 1.9748)
    oriband = x_ori['photometry']['band'][i][sorttime]
    orimask = x_ori['photometry']['mask'][i][sorttime]
    orierror = photerrs[i][sorttime]
    orierror[~np.isfinite(orierror)] = 0
    
    recflux = recon['photometry']['flux'][i][sorttime] * photoflux_std + photoflux_mean #np.sinh(recon['flux'][i][sorttime] * 5.6163 + 1.9748)
    plot_lsst_lc(oriband, oriflux, oritime, orimask, orierror,ax = axes[0, i], label = False, s = 15, lw = 2, flip = True, line = False)
    #breakpoint()
    y_limits = axes[0, i].get_ylim()
    
    #breakpoint()
    axes[0, i].set_ylim(24, y_limits[1])
    axes[0, i].set_xlim(0, 200)
    y_limits = axes[0, i].get_ylim()
    
    plot_lsst_lc(oriband, recflux, oritime, orimask, ax = axes[1, i], label = False, s = 15, lw = 2, flip = True, line = False)
    axes[1, i].set_ylim(y_limits)
    axes[1, i].set_xlim(0, 200)

plt.tight_layout()
fig.show()
fig.savefig(f"{which_data}_LC_recon_fromLC.png")
plt.close()

