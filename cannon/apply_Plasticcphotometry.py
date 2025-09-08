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
from daep.data_util import SpectraDatasetFromnp, collate_fn_stack, to_device, padding_collate_fun, PLAsTiCCfromcsvaug, to_np_cpu
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore
from daep.daep import unimodaldaep
import math 
import os
from tqdm import tqdm
import copy
from daep.plot_util import plot_lsst_lc


test_data = PLAsTiCCfromcsvaug("../data/PLAsTiCC/PLAsTiCC_test_42.csv")
torch.manual_seed(4)
test_loader = DataLoader(test_data, batch_size = 20, collate_fn = padding_collate_fun(supply=['flux', 'band', 'time'],
                                                           mask_by="flux", multimodal=False), shuffle = True)

x = to_device(next(iter(test_loader)))
x_ori = copy.deepcopy(x)
#breakpoint()


trained_daep = torch.load("../ckpt/Plasticcphotometry_daep_8-8-4-4-256_16_16_concatTrue_corrattnonlyFalse_fourierFalse_lr0.00025_epoch2000_batch256_reg0.0_aug5.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)

torch.manual_seed(42)

recon = trained_daep.reconstruct(x, ddim_steps = 200)
recon = to_np_cpu(recon)
x_ori = to_np_cpu(x_ori)

fig, axes = plt.subplots(3, 10, figsize=(30, 9))  # 4 rows, 5 columns

for i in range(10):
    oritime = x_ori['time'][i]
    sorttime = np.argsort(oritime)
    oritime = oritime[sorttime]
    oriflux =x_ori['flux'][i][sorttime] * 5.6163 + 1.9748 #np.sinh(x_ori['flux'][i][sorttime] * 5.6163 + 1.9748)
    oriband = x_ori['band'][i][sorttime]
    orimask = x_ori['mask'][i][sorttime]
    
    recflux = recon['flux'][i][sorttime] * 5.6163 + 1.9748 #np.sinh(recon['flux'][i][sorttime] * 5.6163 + 1.9748)
    plot_lsst_lc(oriband, oriflux, oritime, orimask, ax = axes[0, i], label = False, s = 15, lw = 2, flip = False)
    y_limits = axes[0, i].get_ylim()
    plot_lsst_lc(oriband, recflux, oritime, orimask, ax = axes[1, i], label = False, s = 15, lw = 2, flip = False)
    axes[1, i].set_ylim(y_limits)
    plot_lsst_lc(oriband, (recflux - oriflux)/oriflux, oritime, orimask, ax = axes[2, i], label = False, s = 15, lw = 2, flip = False)
    axes[2, i].set_ylim(-1, 1)
plt.tight_layout()
fig.show()
fig.savefig("LC_recon_plasticc.png")
plt.close()


