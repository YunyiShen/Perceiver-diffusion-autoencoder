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


data = np.load("../data/test_data_align_with_simu_minimal.npz")

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
        colors.append("#444444")
    elif typ in [2, 6, 10, 11, 16, 19]: # Ib/c
        colors.append("#D62728")
    else:
        colors.append("#17BECF") # others



x = to_device(next(iter(test_loader)))

ckpt = "ZTFspectra_daep_4-4-4-4-128_heads8-8_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_reg0.0_aug5"
trained_daep = torch.load(f"../ckpt/{ckpt}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)


torch.manual_seed(42)
encode = trained_daep.encode(x)
encode = to_np_cpu(encode)
encode = encode.reshape(encode.shape[0], -1)


#### mae #####
ckpt_mae = "ZTFspectra_mae_4-4-4-2-128_heads8-8_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_mask0.75_aug5"
trained_mae = torch.load(f"../ckpt/{ckpt_mae}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)


torch.manual_seed(42)
encode_mae = trained_mae.encode(x)
encode_mae = to_np_cpu(encode_mae)
encode_mae = encode_mae.reshape(encode_mae.shape[0], -1)


#### VAE ####
x_vae = (x['flux'], x['wavelength'], x['phase'], x['mask'])

vae_ckpt = "ZTF_spectra_vaesne_4-4-128-4_heads8_0.00025_epoch2000_batch128_aug5_beta0.1"
trained_vae = torch.load(f"../ckpt/{vae_ckpt}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)

encode_vae = trained_vae.encode(x_vae)
encode_vae = to_np_cpu(encode_vae)
encode_vae = encode_vae.reshape(encode_vae.shape[0], -1)


np.savez(f"./encodes/{ckpt}_test.npz",
         encode = encode,
         types = types
         )


np.savez(f"./encodes/{vae_ckpt}_test.npz",
         encode = encode_vae,
         types = types
         )


np.savez(f"./encodes/{ckpt_mae}_test.npz",
         encode = encode_mae,
         types = types
         )
