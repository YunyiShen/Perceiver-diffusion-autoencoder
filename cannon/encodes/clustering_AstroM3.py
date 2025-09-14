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

which = "test"
daep = [np.load(f"AstroM3spectra/AstroM3spectra_daep_4-8-4-4-128_8_8_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch64_reg0.0_aug1_{which}_batch{i}.npz") for i in range(9 if which == "test" else 67)]
mae = [np.load(f"AstroM3spectra/AstroM3spectra_mae_4-8-4-2-128_8_8_hiddenlen256_concatTrue_corrattnonlyFalse_lr0.00025_epoch2000_batch64_mask0.75_aug1_{which}_batch{i}.npz") for i in range(9 if which == "test" else 67)]
vae = [np.load(f"AstroM3spectra/AstroM3_spectra_vaesne_4-8-128-6_heads8_hiddenlen256_0.00025_epoch200_batch128_aug1_beta0.1_{which}_batch{i}.npz") for i in range(9 if which == "test" else 67)]

daep = {"encode": np.concatenate([x['encode'] for x in daep]), "types": np.concatenate([x['types'] for x in daep])}
mae = {"encode": np.concatenate([x['encode'] for x in mae]), "types": np.concatenate([x['types'] for x in mae])}
vae = {"encode": np.concatenate([x['encode'] for x in vae]), "types": np.concatenate([x['types'] for x in vae])}

unique_types, types_int = np.unique(daep["types"], return_inverse=True)

daep['types'] = types_int
mae['types'] = types_int
vae['types'] = types_int

kmeans_daep = KMeans(n_clusters=10).fit(daep['encode'])
ari_daep = ARI(kmeans_daep.labels_, types_int)



kmeans_mae = KMeans(n_clusters=10).fit(mae['encode'])
ari_mae = ARI(kmeans_mae.labels_, types_int)


kmeans_vae = KMeans(n_clusters=10).fit(vae['encode'])
ari_vae = ARI(kmeans_vae.labels_, types_int)


print(f"ari: daep: {ari_daep}, mae: {ari_mae}, vae: {ari_vae}")


silhoiette_daep = silhoiette(daep["encode"], types_int)
silhoiette_mae = silhoiette(mae['encode'], types_int)
silhoiette_vae = silhoiette(vae['encode'], types_int)

print(f"silhoiette: daep: {silhoiette_daep}, mae: {silhoiette_mae}, vae: {silhoiette_vae}")
