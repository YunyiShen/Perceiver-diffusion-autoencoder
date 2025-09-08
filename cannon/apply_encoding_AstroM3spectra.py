import torch
import glob
import torch
from datasets import load_dataset
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import numpy as np
# optimizer
from torch.optim import AdamW

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from daep.data_util import collate_fn_stack, to_device, to_np_cpu, padding_collate_fun, save_dictlist, load_dictlist
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore
from daep.daep import unimodaldaep
import math 
import os
from tqdm import tqdm
import copy


class AstroM3Dataset(Dataset):
    def __init__(self, name = "full_42", which = "train", aug = None):
        # Load the default full dataset with seed 42
        assert aug is None or (aug >=1 and isinstance(aug, int)), "Augmentation has to be positive integer >=1 or None for not augmenting"
        self.dataset = load_dataset("../../AstroM3Dataset", name=name, trust_remote_code=True)[which]
        self.dataset.set_format(type="torch")
        self.aug = aug if aug is not None else 1
        self.which = which
        if which == "test" and self.aug > 1:
            print("We do not augment test")
            self.aug = 1
        #breakpoint()

    def __len__(self):
        return self.aug * len(self.dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        
        res = {"flux": (torch.log10(self.dataset[idx]['spectra'][:, 1] + (
                        (torch.randn_like(self.dataset[idx]['spectra'][:, 2]) * \
                        self.dataset[idx]['spectra'][:, 2]) if self.aug > 1 else 0. ) ) - 2.8766)/0.7795  # this is the mean
                        , 
               "wavelength": (self.dataset[idx]['spectra'][:, 0] - 6000.1543)/1548.8627, 
               "phase": torch.tensor(0.)#,
               #"label": self.dataset[idx]['label']
               }       
        
        #breakpoint()
        return res

which = "train"
test_data = AstroM3Dataset(which=which)
ckpt = "AstroM3spectra_daep_4-8-4-4-128_8_8_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch64_reg0.0_aug1"
trained_daep = torch.load(f"../ckpt/{ckpt}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)
vae_ckpt = "AstroM3_spectra_vaesne_4-8-128-6_heads8_hiddenlen256_0.00025_epoch200_batch128_aug1_beta0.1"
trained_vae = torch.load(f"../ckpt/{vae_ckpt}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)
#torch.manual_seed(4125)
#breakpoint()
batch_size = 256
test_loader = DataLoader(test_data, batch_size = batch_size, collate_fn = padding_collate_fun(supply=['flux', 'wavelength', 'time'],
                                                          mask_by="flux", multimodal=False), shuffle = False)
from tqdm import tqdm
for i, x in tqdm(enumerate(test_loader)):

    types = test_data.dataset['label'][(i*batch_size):min((i+1)*batch_size, len(test_data))]

    x = to_device(x)




    torch.manual_seed(42)
    encode = trained_daep.encode(x)
    encode = to_np_cpu(encode)
    encode = encode.reshape(encode.shape[0], -1)



    #breakpoint()

    x_vae = (x['flux'], x['wavelength'], x['phase'], x['mask'])



    encode_vae = trained_vae.encode(x_vae)
    encode_vae = to_np_cpu(encode_vae)
    encode_vae = encode_vae.reshape(encode_vae.shape[0], -1)


    np.savez(f"./encodes/AstroM3spectra/{ckpt}_{which}_batch{i}.npz",
         encode = encode,
         types = types
         )


    np.savez(f"./encodes/AstroM3spectra/{vae_ckpt}_{which}_batch{i}.npz",
         encode = encode_vae,
         types = types
         )



