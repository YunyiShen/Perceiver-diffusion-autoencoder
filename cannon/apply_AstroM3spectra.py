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

test_data = AstroM3Dataset(which="test")
torch.manual_seed(42)
size = 50
ckpt = "AstroM3spectra_daep_4-8-4-4-128_8_8_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch64_reg0.0_aug1"
trained_daep = torch.load(f"../ckpt/{ckpt}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)


test_loader = DataLoader(test_data, batch_size = 64, collate_fn = padding_collate_fun(supply=['flux', 'wavelength', 'time'],
                                                          mask_by="flux", multimodal=False), shuffle = False)

for i, x in tqdm(enumerate(test_loader)):
    


    x = to_device(x)
    x_ori = copy.deepcopy(x)
    #breakpoint()




    rec = []

    for j in tqdm(range(size)):
        x = copy.deepcopy(x_ori)
        recon = trained_daep.reconstruct(x, ddim_steps = 200)
        recon = to_np_cpu(recon)
        rec.append(recon)
    x_ori = to_np_cpu(x_ori)

    save_dictlist(f"./res/AstroM3spectra/{ckpt}_rec_batch{i}.npz", rec)
    np.savez(f"./res/AstroM3spectra/{ckpt}_gt_batch{i}.npz",
         **x_ori
         )

'''
fig, axes = plt.subplots(4, 5, figsize=(50, 12))  # 4 rows, 5 columns
axes = axes.flatten()
for i in range(20):
    wavelength = x_ori['wavelength'][i][~x_ori['mask'][i]] * 1548.8627 + 6000.1543
    sortwavelength = np.argsort(wavelength)
    axes[i].plot(wavelength[sortwavelength],
                 recon['flux'][i][~recon['mask'][i]][sortwavelength]* 0.7795 + 2.8766,
                 color = "blue",
                 label = "daep" if i == 0 else None
                 )
    axes[i].plot(wavelength[sortwavelength],
                 x_ori['flux'][i][~x_ori['mask'][i]][sortwavelength] * 0.7795 + 2.8766,
                 color = "red",
                 label = "ground truth" if i == 0 else None
                 )
    
    axes[i].set_xlabel("normalized wavelength")


axes[0].legend()
axes[0].set_ylabel("logFnu")
axes[5].set_ylabel("logFnu")
plt.tight_layout()
fig.show()
fig.savefig("spectra_recon_AstroM3.pdf")
plt.close()
'''