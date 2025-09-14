import torch
import glob
import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW
import pickle

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from daep.data_util import SpectraDatasetFromnp, collate_fn_stack, to_device, to_np_cpu
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore
from daep.daep import unimodaldaep
import math 
import os
from tqdm import tqdm
import copy
from daep.probing import make_fewshot_loaders, LinearProbe, MLPProbe, train_probe, evaluate_metrics
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import silhouette_score as silhoiette

which = "_test"
daep = dict( np.load( f"ZTFspectra_daep_4-4-4-4-128_heads8-8_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_reg0.0_aug5{which}.npz").items())
mae = dict( np.load(f"ZTFspectra_mae_4-4-4-2-128_heads8-8_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch128_mask0.75_aug5{which}.npz").items())
vae = dict( np.load(f"ZTF_spectra_vaesne_4-4-128-4_heads8_0.00025_epoch2000_batch128_aug5_beta0.1{which}.npz").items())

types = daep['types']

collapsed_type = []
for typ in types:
    if typ in [0, 3, 9, 12, 13, 14, 15]: # Ia
        collapsed_type.append(0)
    elif typ in [2, 6, 10, 11, 16, 19]: # Ib/c
        collapsed_type.append(1)
    else:
        collapsed_type.append(2)
collapsed_type = np.array(collapsed_type, dtype = int)
types_int = collapsed_type

daep['types'] = collapsed_type
mae['types'] = collapsed_type
vae['types'] = collapsed_type

kmeans_daep = KMeans(n_clusters=3).fit(daep['encode'])
ari_daep = ARI(kmeans_daep.labels_, types_int)



kmeans_mae = KMeans(n_clusters=3).fit(mae['encode'])
ari_mae = ARI(kmeans_mae.labels_, types_int)


kmeans_vae = KMeans(n_clusters=3).fit(vae['encode'])
ari_vae = ARI(kmeans_vae.labels_, types_int)


print(f"ari: daep: {ari_daep}, mae: {ari_mae}, vae: {ari_vae}")


silhoiette_daep = silhoiette(daep["encode"], types_int)
silhoiette_mae = silhoiette(mae['encode'], types_int)
silhoiette_vae = silhoiette(vae['encode'], types_int)

print(f"silhoiette: daep: {silhoiette_daep}, mae: {silhoiette_mae}, vae: {silhoiette_vae}")
