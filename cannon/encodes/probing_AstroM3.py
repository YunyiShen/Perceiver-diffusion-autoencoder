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

batch_size = 64
training_proportion = .30
epochs = 150
hidden = 64
lr = 1e-3

#breakpoint()
seeds = [1,2,3,4,5, 41, 42, 43, 44, 45]

for seed in tqdm(seeds):
    torch.manual_seed(seed)
    metric_name_base = f"classification_metric/AstroM3spectra/AstroM3spectra_fewshot_{training_proportion}_{lr}_{epochs}_{hidden}_{seed}"
    ckpt_name_base = f"classification_ckpt/AstroM3spectra/AstroM3spectra_fewshot_{training_proportion}_{lr}_{epochs}_{hidden}_{seed}"

    support_loader_daep, query_loader_daep, input_dim_daep, num_classes_daep = make_fewshot_loaders(daep, training_proportion, batch_size)
    support_loader_mae, query_loader_mae, input_dim_mae, num_classes_mae = make_fewshot_loaders(mae, training_proportion, batch_size)
    support_loader_vae, query_loader_vae, input_dim_vae, num_classes_vae = make_fewshot_loaders(vae, training_proportion, batch_size)
    
    daep_linear = LinearProbe(input_dim_daep, num_classes_daep)
    mae_linear = LinearProbe(input_dim_mae, num_classes_mae)
    vae_linear = LinearProbe(input_dim_vae, num_classes_vae)
    
    daep_mlp = MLPProbe(input_dim_daep, num_classes_daep, hidden)
    mae_mlp = MLPProbe(input_dim_mae, num_classes_mae, hidden)
    vae_mlp = MLPProbe(input_dim_vae, num_classes_vae, hidden)
    
    
    ##### daep ######
    ####### linear #######
    train_probe(daep_linear, support_loader_daep, query_loader_daep, 
                epochs=epochs, lr=lr, device=device)
    metric = evaluate_metrics(daep_linear, query_loader_daep, device = device)
    torch.save(daep_linear, f"{ckpt_name_base}_daep_linear.pth")
    print(f"daep, linear, seed{seed}:")
    print(metric['accuracy'], metric['f1_macro'])
    with open(f"{metric_name_base}_daep_linear.pkl", "wb") as f:
        pickle.dump(metric, f)
        
    ####### mlp #######
    train_probe(daep_mlp, support_loader_daep, query_loader_daep, 
                epochs=epochs, lr=lr, device=device)
    metric = evaluate_metrics(daep_mlp, query_loader_daep, device = device)
    torch.save(daep_mlp, f"{ckpt_name_base}_daep_mlp.pth")
    print(f"daep, mlp, seed{seed}:")
    print(metric['accuracy'], metric['f1_macro'])
    with open(f"{metric_name_base}_daep_mlp.pkl", "wb") as f:
        pickle.dump(metric, f)
        
        
    
    
    ##### mae ######
    ####### linear #######
    train_probe(mae_linear, support_loader_mae, query_loader_mae, 
                epochs=epochs, lr=lr, device=device)
    metric = evaluate_metrics(mae_linear, query_loader_mae,  device = device)
    torch.save(mae_linear, f"{ckpt_name_base}_mae_linear.pth")
    print(f"mae, linear, seed{seed}:")
    print(metric['accuracy'], metric['f1_macro'])
    with open(f"{metric_name_base}_mae_linear.pkl", "wb") as f:
        pickle.dump(metric, f)
        
    ####### mlp #######
    train_probe(mae_mlp, support_loader_mae, query_loader_mae, 
                epochs=epochs, lr=lr, device=device)
    metric = evaluate_metrics(mae_mlp, query_loader_mae, device = device)
    torch.save(mae_mlp, f"{ckpt_name_base}_mae_mlp.pth")
    print(f"mae, mlp, seed{seed}:")
    print(metric['accuracy'], metric['f1_macro'])
    with open(f"{metric_name_base}_mae_mlp.pkl", "wb") as f:
        pickle.dump(metric, f)
        
        
        
    
    ##### vae ######
    ####### linear #######
    train_probe(vae_linear, support_loader_vae, query_loader_vae, 
                epochs=epochs, lr=lr, device=device)
    metric = evaluate_metrics(vae_linear, query_loader_vae, device = device)
    torch.save(vae_linear, f"{ckpt_name_base}_vae_linear.pth")
    print(f"vae, linear, seed{seed}:")
    print(metric['accuracy'], metric['f1_macro'])
    with open(f"{metric_name_base}_vae_linear.pkl", "wb") as f:
        pickle.dump(metric, f)
        
    ####### mlp #######
    train_probe(vae_mlp, support_loader_vae, query_loader_vae, 
                epochs=epochs, lr=lr, device=device)
    metric = evaluate_metrics(vae_mlp, query_loader_vae, device = device)
    torch.save(vae_mlp, f"{ckpt_name_base}_vae_mlp.pth")
    print(f"vae, mlp, seed{seed}:")
    print(metric['accuracy'], metric['f1_macro'])
    with open(f"{metric_name_base}_vae_mlp.pkl", "wb") as f:
        pickle.dump(metric, f)
        
