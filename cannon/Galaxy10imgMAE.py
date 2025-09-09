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
from daep.data_util import ImgH5DatasetAug, collate_fn_stack, to_device
from daep.ImgLayers import HostImgTransceiverEncoder, HostImgTransceiverDecoder
from daep.mae import unimodalmae
from tqdm import tqdm
import os
import fire
import math
import h5py


#breakpoint()

def train(epoch=200, lr = 2.5e-4, bottlenecklen = 4, bottleneckdim = 4, 
          model_dim = 64, sincosin = False, encoder_layers = 4, 
          decoder_layers = 4,mask_rate = 0.3, imgsize = 64, patch = 4, 
          batch = 256, aug = 5, save_every = 20):
    
    
    splits = np.load("../../Galaxy10/splits.npz")
    
    training_data = ImgH5DatasetAug("../../Galaxy10/Galaxy10_DECals.h5", 
                                    key="images", indices=splits["train"],
                                    size = imgsize,
                                    factor = aug, preload = True)
    training_loader = DataLoader(training_data, 
                                 
                                 batch_size = batch, 
                                 num_workers=1,  # adjust based on CPU cores
                                 pin_memory=True,  # speeds up transfer to GPU
                                 collate_fn = collate_fn_stack)

    img_encoder = HostImgTransceiverEncoder(img_size = imgsize,
                    bottleneck_length = bottlenecklen,
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    ff_dim = model_dim,
                    num_layers = encoder_layers,
                    sincosin = sincosin,
                    patch_size=patch).to(device)

    img_Decoder = HostImgTransceiverDecoder(
        img_size = imgsize,
        bottleneck_dim = bottleneckdim,
        model_dim = model_dim,
        ff_dim = model_dim,
        num_layers = decoder_layers,
        patch_size=patch,
        sincosin = sincosin
    ).to(device)

    mymae = unimodalmae(img_encoder, img_Decoder, mask_rate).to(device)
    mymae.train()
    optimizer = AdamW(mymae.parameters(), lr=lr)
    epoch_loss = []
    epoches = []
    target_save = None
    progress_bar = tqdm(range(epoch))
    for ep in progress_bar:
        losses = []
        for x in training_loader:
            x = to_device(x)
            optimizer.zero_grad()
            loss = mymae(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        this_epoch = np.array(losses).mean().item()
        epoch_loss.append(math.log(this_epoch))
        epoches.append(ep)
        if (ep+1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
            target_save = f"../ckpt/Galaxy10_mae_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_sincos{sincosin}_lr{lr}_epoch{ep+1}_batch{batch}_mask{mask_rate}_aug{aug}_imgsize{imgsize}.pth"
            torch.save(mymae, target_save)
            plt.plot(epoches, epoch_loss)
            plt.show()
            plt.savefig(f"./logs/Galaxy10_mae_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_sincos{sincosin}_lr{lr}_batch{batch}_mask{mask_rate}_aug{aug}_imgsize{imgsize}.png")
            plt.close()
        progress_bar.set_postfix(loss=f"epochs:{ep}, {math.log(this_epoch):.4f}") 
        
  
            

if __name__ == '__main__':
    fire.Fire(train)







