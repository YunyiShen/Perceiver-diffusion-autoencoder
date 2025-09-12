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


which = "train"
daep = [np.load(f"AstroM3spectra/AstroM3spectra_daep_4-8-4-4-128_8_8_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch64_reg0.0_aug1_{which}_batch{i}.npz") for i in range(9 if which == "test" else 67)]
mae = [np.load(f"AstroM3spectra/AstroM3spectra_mae_4-8-4-2-128_8_8_hiddenlen256_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch64_mask0.75_aug1_{which}_batch{i}.npz") for i in range(9 if which == "test" else 67)]
vae = [np.load(f"AstroM3spectra/AstroM3_spectra_vaesne_4-8-128-6_heads8_hiddenlen256_0.00025_epoch200_batch128_aug1_beta0.1_{which}_batch{i}.npz") for i in range(9 if which == "test" else 67)]

daep = {"encode": np.concatenate([x['encode'] for x in daep]), "types": np.concatenate([x['types'] for x in daep])}
mae = {"encode": np.concatenate([x['encode'] for x in mae]), "types": np.concatenate([x['types'] for x in mae])}
vae = {"encode": np.concatenate([x['encode'] for x in vae]), "types": np.concatenate([x['types'] for x in vae])}

figname = f"Figs/encoding_AstroM3spectra_{which}.pdf"


#breakpoint()

types = daep['types']
unique_types, types_int = np.unique(types, return_inverse=True)
'''
cmap = distinct_colors_10 = [
    "#ffe119",  # yellow
    "#e6194b",  # red
    "#3cb44b",  # green
    
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#46f0f0",  # cyan
    "#f032e6",  # magenta
    "#bcf60c",  # lime
    "#fabebe",  # pink
] #
'''
cmap = plt.get_cmap("tab10")
type_to_color = {t: cmap(i) for i, t in enumerate(unique_types)}

# Vectorized mapping: build a color array for all points
colors = np.array([type_to_color[t] for t in types])
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
    plt.Line2D([0], [0], marker='o', color='w', label=t, markerfacecolor=type_to_color[t], markersize=8)
    for t in unique_types
]
legend = fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=5,   # now 4 columns, since "Type" is included
    bbox_to_anchor=(0.5, -0.02)
)
#legend.get_texts()[0].set_weight("bold")

plt.tight_layout()
plt.subplots_adjust(bottom=0.26)  # make space at bottom for legend
fig.show()
fig.savefig(figname)
plt.close()
