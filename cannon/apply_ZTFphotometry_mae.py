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
from daep.data_util import PhotoDatasetFromnp, collate_fn_stack, to_device, to_np_cpu, save_dictlist

import math 
import os
from tqdm import tqdm
import copy

from daep.plot_util import plot_ztf_lc

data = np.load("../data/test_data_align_with_simu_minimal.npz")

### spectra ###
photoflux, phototime, photoband = data['photoflux'], data['phototime'], data['photowavelength']
photomask = data['photomask']

    
photoflux = torch.tensor(photoflux, dtype = torch.float32)
phototime = torch.tensor(phototime, dtype = torch.float32)
photoband = torch.tensor(photoband, dtype = torch.long)
photomask = torch.tensor(photomask == 0)

flux_mean, flux_std = data['combined_mean'], data['combined_std']
time_mean, time_std = data['combined_time_mean'], data['combined_time_std']


test_data = PhotoDatasetFromnp(photoflux, phototime, photoband, photomask)
torch.manual_seed(45)
test_loader = DataLoader(test_data, batch_size = 128, collate_fn = collate_fn_stack, shuffle = False)


size = 50
#ckpt = "ZTFphotometric_daep_2-2-4-4-128_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_reg0.0_aug5"
ckpt = "ZTFphotometric_mae_2-2-4-2-128_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_mask0.3_aug5"

trained_daep = torch.load(f"../ckpt/{ckpt}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)
for i, x in tqdm(enumerate(test_loader)):


    x = to_device(x)
    x_ori = copy.deepcopy(x)


    torch.manual_seed(12345)
    #x_ori = to_np_cpu(x_ori)
    #recon = to_np_cpu(recon)


    
    rec = []

    for j in range(size):
        x = copy.deepcopy(x_ori)
        recon = trained_daep.reconstruct(x, 0.75)
        recon = to_np_cpu(recon)
        recon['flux'] = recon['flux'] * flux_std + flux_mean
    
        rec.append(recon)

    x_ori = to_np_cpu(x_ori)
    x_ori['time'] = x_ori['time'] * time_std + time_mean
    x_ori['flux'] = x_ori['flux'] * flux_std + flux_mean 

    save_dictlist(f"./res/ZTFphotometry/{ckpt}_rec_batch{i}.npz", rec)
    


plt.rcParams.update({'font.size': 20})  # sets default font size
fig, axes = plt.subplots(3, 6, figsize=(20, 8))  # 4 rows, 5 columns
#axes = axes.flatten()
for i in range(6):
    oritime = x_ori['time'][i] * time_std + time_mean
    oriflux = x_ori['flux'][i] * flux_std + flux_mean
    oriband = x_ori['band'][i]
    orimask = x_ori['mask'][i]
    
    recflux = recon['flux'][i] * flux_std + flux_mean
    
    
    plot_ztf_lc(oriband, oriflux, oritime, orimask, ax = axes[0, i], label = i==0, s = 15, lw = 2)
    plot_ztf_lc(oriband, recflux, oritime, orimask, ax = axes[1, i], label = False, s = 15, lw = 2)
    axes[0, i].set_xticks([])         # removes ticks
    axes[1, i].set_xticklabels([])    # removes tick labels

    axes[1,i].set_ylim(axes[0, i].get_ylim())
    plot_ztf_lc(oriband, (recflux-oriflux)/oriflux, oritime, orimask, ax = axes[2, i], label = False, s = 15, lw = 2)
    axes[2,i].set_ylim(-0.15, 0.15)

axes[0, 0].set_ylabel("Mag\noriginal")

axes[1, 0].set_ylabel("Mag\nreconstruction")
axes[2, 0].set_ylabel("Relative error")
axes[2, 2].set_xlabel("Phase")
handles, labels = axes[0, 0].get_legend_handles_labels()

# Add a single legend outside at the bottom center
fig.legend(handles, labels, loc='lower center', ncol=len(labels),
           bbox_to_anchor=(0.5, -0.02), fontsize=12)
plt.tight_layout()
fig.show()
fig.savefig("ZTFLC_recon_mae.pdf")
plt.close()


