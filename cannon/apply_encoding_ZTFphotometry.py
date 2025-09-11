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
from daep.data_util import SpectraDatasetFromnp, collate_fn_stack, to_device, to_np_cpu, PhotoDatasetFromnp
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore
from daep.daep import unimodaldaep
import math 
import os
from tqdm import tqdm
import copy


data = np.load("../data/train_data_align_with_simu_minimal.npz")

photoflux, phototime, photoband = data['photoflux'], data['phototime'], data['photowavelength']
photomask = data['photomask']

    
photoflux = torch.tensor(photoflux, dtype = torch.float32)
phototime = torch.tensor(phototime, dtype = torch.float32)
photoband = torch.tensor(photoband, dtype = torch.long)
photomask = torch.tensor(photomask == 0)

flux_mean, flux_std = data['combined_mean'], data['combined_std']
time_mean, time_std = data['combined_time_mean'], data['combined_time_std']


test_data = PhotoDatasetFromnp(photoflux, phototime, photoband, photomask)
#torch.manual_seed(4125)
#breakpoint()
test_loader = DataLoader(test_data, batch_size = 15000, collate_fn = collate_fn_stack, shuffle = False)

types = data['type']
x = to_device(next(iter(test_loader)))

ckpt = "ZTFphotometric_daep_2-2-4-4-128_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_reg0.0_aug5"
trained_daep = torch.load(f"../ckpt/{ckpt}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)


torch.manual_seed(42)
encode = trained_daep.encode(x)
encode = to_np_cpu(encode)
encode = encode.reshape(encode.shape[0], -1)

ckpt_mae = "ZTFphotometric_mae_2-2-4-2-128_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_mask0.3_aug5"
trained_mae = torch.load(f"../ckpt/{ckpt_mae}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)


torch.manual_seed(42)
encode_mae = trained_mae.encode(x)
encode_mae = to_np_cpu(encode_mae)
encode_mae = encode.reshape(encode_mae.shape[0], -1)

#breakpoint()

x_vae = (x['flux'], x['time'], x['band'], x['mask'])

vae_ckpt = "ZTF_photometry_vaesne_2-2-128-4_heads4_0.00025_epoch2000_batch128_aug5_beta0.1"
trained_vae = torch.load(f"../ckpt/{vae_ckpt}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)

encode_vae = trained_vae.encode(x_vae)
encode_vae = to_np_cpu(encode_vae)
encode_vae = encode_vae.reshape(encode_vae.shape[0], -1)


np.savez(f"./encodes/{ckpt}.npz",
         encode = encode,
         types = types
         )
np.savez(f"./encodes/{ckpt_mae}.npz",
         encode = encode_mae,
         types = types
         )

np.savez(f"./encodes/{vae_ckpt}.npz",
         encode = encode_vae,
         types = types
         )



