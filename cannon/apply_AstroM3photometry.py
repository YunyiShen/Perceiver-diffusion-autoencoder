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
from daep.data_util import collate_fn_stack, to_device, to_np_cpu, padding_collate_fun
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore
from daep.daep import unimodaldaep
import math 
import os
from tqdm import tqdm
import copy

def truncate_photometry(example):
    phot = example["photometry"]
    phot = torch.unique(phot, dim=0)
    # Ensure it's a tensor
    if isinstance(phot, torch.Tensor):
        if phot.shape[0] > 800:
            phot = phot[:800, :]
    phot[:, 0] -= phot[:, 0].min()
    phot[:, 1] -= phot[:, 1].mean()
    return {"photometry": phot}

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
        self.dataset = self.dataset.map(
            truncate_photometry,
            desc="Centering photometry",
            load_from_cache_file=False
            )
        #breakpoint()
    def __len__(self):
        return self.aug * len(self.dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        
               
        
        return {
            "flux": (self.dataset[idx]['photometry'][:, 1] + (
                        (torch.randn_like(self.dataset[idx]['photometry'][:, 0]) * \
                        self.dataset[idx]['photometry'][:, 2]) if self.aug > 1 else 0.) )/6.8488,#- 22.6879)/27.7245 , # noise added only if augmentation and training
            "time": (self.dataset[idx]['photometry'][:, 0]- 788.0814)/475.3434 # [-3, 3] kinda aribitrary to match standardized range
        } 
        

test_data = AstroM3Dataset(which="test")
torch.manual_seed(42)
test_loader = DataLoader(test_data, batch_size = 20, collate_fn = padding_collate_fun(supply=['flux', 'time', 'time'],
                                                           mask_by="flux", multimodal=False), shuffle = True)

x = to_device(next(iter(test_loader)))
x_ori = copy.deepcopy(x)
#breakpoint()


trained_daep = torch.load("../ckpt/AstroM3photometry_daep_8-8-4-4-256_4_4_concatTrue_corrattnonlyFalse_lr0.00025_epoch480_batch256_reg0.0_aug1.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)

torch.manual_seed(42)
recon = trained_daep.reconstruct(x, ddim_steps = 200)
recon = to_np_cpu(recon)


x_ori = to_np_cpu(x_ori)

fig, axes = plt.subplots(4, 5, figsize=(20, 12))  # 4 rows, 5 columns
axes = axes.flatten()
for i in range(20):
    time = x_ori['time'][i][~x_ori['mask'][i]] * 475.3434 + 788.0814
    sorttime = np.argsort(time)
    axes[i].plot(time[sorttime],
                 recon['flux'][i][~recon['mask'][i]][sorttime] ,
                 color = "blue",
                 label = "daep" if i == 0 else None
                 )
    axes[i].plot(time[sorttime],
                 x_ori['flux'][i][~x_ori['mask'][i]][sorttime], #* 27.7245 + 22.6879,
                 color = "red",
                 label = "ground truth" if i == 0 else None
                 )
    
    axes[i].set_xlabel("normalized time")


axes[0].legend()
axes[0].set_ylabel("flux")
axes[5].set_ylabel("flux")
plt.tight_layout()
fig.show()
fig.savefig("LC_recon_AstroM3.pdf")
plt.close()


