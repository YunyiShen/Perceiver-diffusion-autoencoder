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
from daep.data_util import SpectraDatasetFromnp, collate_fn_stack, to_device, padding_collate_fun, PLAsTiCCfromcsvaug
from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore


from daep.daep import unimodaldaep
import math 
import os
from tqdm import tqdm

       


def train(epoch=1000, lr = 2.5e-4, bottlenecklen = 4, bottleneckdim = 4, 
          concat = True, cross_attn_only = False,
          model_dim = 256, encoder_heads = 4, decoder_heads = 4,
          encoder_layers = 4, 
          decoder_layers = 4,
          fourier = False, 
          regularize = 0.00, 
          batch = 16, aug = 1, save_every = 20):
    

    training_data = PLAsTiCCfromcsvaug("../data/PLAsTiCC/PLAsTiCC_train_42.csv",aug=aug)

    training_loader = DataLoader(training_data, batch_size = batch, collate_fn = padding_collate_fun(supply=['flux', 'band', 'time'],
                                                           mask_by="flux", multimodal=False))
    
    photometryEncoder = photometricTransceiverEncoder(
                    num_bands = 6,
                    bottleneck_length = bottlenecklen,
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    ff_dim = model_dim,
                    num_layers = encoder_layers,
                    num_heads = encoder_heads,
                    concat = concat,
                    fourier = fourier
                    ).to(device)

    photometryScore = photometricTransceiverScore(
                    num_bands = 6,
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    ff_dim = model_dim,
                    num_layers = decoder_layers,
                    num_heads = decoder_heads,
                    concat = concat,
                    cross_attn_only = cross_attn_only,
                    fourier = fourier
                    ).to(device)


    mydaep = unimodaldaep(photometryEncoder, photometryScore, regularize = regularize).to(device)
    
    mydaep.train()
    optimizer = AdamW(mydaep.parameters(), lr=lr)
    epoch_loss = []
    epoches = []
    target_save = None
    progress_bar = tqdm(range(epoch))
    for ep in progress_bar:
        losses = []
        for x in training_loader:
            x = to_device(x)
            #breakpoint()
            optimizer.zero_grad()
            loss = mydaep(x)
            #print(loss)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        this_epoch = np.array(losses).mean().item()
        epoch_loss.append(math.log(this_epoch))
        epoches.append(ep)
        if (ep+1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
            target_save = f"../ckpt/Plasticcphotometry_daep_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_{encoder_heads}_{decoder_heads}_concat{concat}_corrattnonly{cross_attn_only}_fourier{fourier}_lr{lr}_epoch{ep+1}_batch{batch}_reg{regularize}_aug{aug}.pth"
            torch.save(mydaep, target_save)
            plt.plot(epoches, epoch_loss)
            plt.show()
            plt.savefig(f"./logs/Plasticcphotometry_daep_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_{encoder_heads}_{decoder_heads}_concat{concat}_corrattnonly{cross_attn_only}_fourier{fourier}_lr{lr}_batch{batch}_reg{regularize}_aug{aug}.png")
            plt.close()
        progress_bar.set_postfix(loss=f"epochs:{ep}, {math.log(this_epoch):.4f}") 
    


import fire           

if __name__ == '__main__':
    fire.Fire(train)
