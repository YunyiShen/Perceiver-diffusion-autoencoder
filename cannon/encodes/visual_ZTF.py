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


which = ""
daep = np.load( f"ZTFspectra_daep_4-4-4-4-128_heads8-8_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_reg0.0_aug5{which}.npz")
mae = np.load(f"ZTFspectra_mae_4-4-4-2-128_heads8-8_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_mask0.75_aug5{which}.npz")
vae = np.load(f"ZTF_spectra_vaesne_4-4-128-4_heads8_0.00025_epoch2000_batch128_aug5_beta0.1{which}.npz")

figname = f"Figs/encoding_ZTFspectra{which}.pdf"

'''
which = "_test"
daep = np.load( f"ZTFphotometric_daep_2-2-4-4-128_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_reg0.0_aug5{which}.npz")
mae = np.load(f"ZTFphotometric_mae_2-2-4-2-128_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_mask0.75_aug5{which}.npz")
vae = np.load(f"ZTF_photometry_vaesne_2-2-128-4_heads4_0.00025_epoch2000_batch128_aug5_beta0.1{which}.npz")
figname = f"Figs/encoding_ZTFphotometry{which}.pdf"
'''

#breakpoint()

types = daep['types']

colors = []
for typ in types:
    if typ in [0, 3, 9, 12, 13, 14, 15]: # Ia
        colors.append("#444444")
    elif typ in [2, 6, 10, 11, 16, 19]: # Ib/c
        colors.append("#D62728")
    else:
        colors.append("#17BECF") # others


from sklearn.manifold import TSNE
print("running t-SNEs")
daep_tsne = TSNE(n_components=2, random_state=42)  # set perplexity or other params as needed
daep_encode = daep_tsne.fit_transform(daep['encode'])

mae_tsne = TSNE(n_components=2, random_state=42)
mae_encode = mae_tsne.fit_transform(mae['encode'])

vae_tsne = TSNE(n_components=2, random_state=42)
vae_encode = vae_tsne.fit_transform(vae['encode'])
print("Done")

import matplotlib.patches as mpatches

plt.rcParams["font.size"] = 16   # default is usually 10

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))  # 4 rows, 5 columns
axes[0].scatter(daep_encode[:, 0], daep_encode[:, 1], c = colors, s=1.5)
axes[1].scatter(mae_encode[:, 0], mae_encode[:, 1], c = colors, s=1.5)
axes[2].scatter(vae_encode[:, 0], vae_encode[:, 1], c = colors, s=1.5)

#axes[0].legend(handles=legend_handles, title='Type')
axes[0].set_title("daep")
axes[1].set_title("mae")
axes[2].set_title("vae")



for i in range(3):
    axes[i].set_xlabel("t-SNE 1")
axes[0].set_ylabel("t-SNE 2")
# Shared legend for all axes, placed below

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], linestyle="none", label="Type:"),  # acts as a prefix
    Line2D([0], [0], marker='o', color='w', label='Ia',
           markerfacecolor="#444444", markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Ib/c',
           markerfacecolor="#D62728", markersize=8),
    Line2D([0], [0], marker='o', color='w', label='other',
           markerfacecolor="#17BECF", markersize=8)
]

legend = fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,   # now 4 columns, since "Type" is included
    bbox_to_anchor=(0.5, 0.02)
)
#legend.get_texts()[0].set_weight("bold")

plt.tight_layout()
plt.subplots_adjust(bottom=0.23)  # make space at bottom for legend
fig.show()
fig.savefig(figname)
plt.close()
