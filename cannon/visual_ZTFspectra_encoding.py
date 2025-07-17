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
from daep.data_util import SpectraDatasetFromnp, collate_fn_stack, to_device, to_np_cpu
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore
from daep.daep import unimodaldaep
import math 
import os
from tqdm import tqdm
import copy


data = np.load("../data/train_data_align_with_simu_minimal.npz")

### spectra ###
flux, wavelength, mask = data['flux'], data['wavelength'], data['mask']
phase = data['phase']

flux = torch.tensor(flux, dtype=torch.float32)
wavelength = torch.tensor(wavelength, dtype=torch.float32)
mask = torch.tensor(mask == 0)
phase = torch.tensor(phase, dtype=torch.float32)

flux_mean, flux_std = data['flux_mean'], data['flux_std']
wavelength_mean, wavelength_std = data['wavelength_mean'], data['wavelength_std']
phase_mean, phase_std = data['spectime_mean'], data['spectime_std']

test_data = SpectraDatasetFromnp(flux, wavelength, phase, mask)
#torch.manual_seed(4125)
#breakpoint()
test_loader = DataLoader(test_data, batch_size = 15000, collate_fn = collate_fn_stack, shuffle = False)

types = data['type']
#{"SN Ia-SC": 0, "SN II-pec": 1, "SN Ibn": 2, "SN Ia-CSM": 3, "SN IIP": 4, "SN Icn": 5, 
# "SN Ib/c": 6, "SN II": 7, "SLSN-I": 8, "SN Ia-pec": 9, "SN Ib": 10, "SN Ib-pec": 11, "SN Ia-91T": 12, 
# "SN Iax": 13, "SN Ia-91bg": 14, "SN Ia": 15, "SN Ic": 16, "SN IIb": 17, "SLSN-II": 18, "SN Ic-BL": 19, "SN IIn": 20}
colors = []
for typ in types:
    if typ in [0, 3, 9, 12, 13, 14, 15]: # Ia
        colors.append("blue")
    elif typ in [2, 6, 10, 11, 16, 19]: # Ib/c
        colors.append("red")
    else:
        colors.append("green") # others



x = to_device(next(iter(test_loader)))

trained_daep = torch.load("../ckpt/ZTFspectra_daep_10-4-4-4-128_concatTrue_lr0.00025_epoch2000_batch128_reg0.0_aug5.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)

torch.manual_seed(42)
encode = trained_daep.encode(x)
encode = to_np_cpu(encode)
encode = encode.reshape(encode.shape[0], -1)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)  # set perplexity or other params as needed
encode = tsne.fit_transform(encode)


print(encode.std(axis = 1).mean(axis = 0))

'''
fig, axes = plt.subplots(2, 6, figsize=(20, 12))  # 4 rows, 5 columns

for i in range(2):
    axes[i,0].scatter(encode[:, i, 0], encode[:, i, 1], c = colors)
    axes[i,1].scatter(encode[:, i, 0], encode[:, i, 2], c = colors)
    axes[i,2].scatter(encode[:, i, 0], encode[:, i, 3], c = colors)
    
    axes[i,3].scatter(encode[:, i, 1], encode[:, i, 2], c = colors)
    axes[i,4].scatter(encode[:, i, 1], encode[:, i, 3], c = colors)
    axes[i,5].scatter(encode[:, i, 2], encode[:, i, 3], c = colors)
'''
import matplotlib.patches as mpatches
fig, axes = plt.subplots(1, 1, figsize=(6, 6))  # 4 rows, 5 columns
axes.scatter(encode[:, 0], encode[:, 1], c = colors, s=0.5)
legend_handles = [mpatches.Patch(color="blue", label="Ia"),
                  mpatches.Patch(color="red", label="Ib/c"),
                  mpatches.Patch(color="green", label="other")
                  ]
axes.legend(handles=legend_handles, title='Type')


plt.tight_layout()
fig.show()
fig.savefig("spectra_encode.pdf")
plt.close()


