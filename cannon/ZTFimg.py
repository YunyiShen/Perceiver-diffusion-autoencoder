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
from daep.data_util import ImagePathDataset, collate_fn_stack, to_device, ImagePathDatasetAug
from daep.ImgLayers import HostImgTransceiverEncoder, HostImgTransceiverScore
from daep.daep import unimodaldaep
from tqdm import tqdm
import os
import fire
import math


#breakpoint()

def train(epoch=200, lr = 2.5e-4, bottlenecklen = 4, bottleneckdim = 4, 
          model_dim = 64, sincosin = True,encoder_layers = 4, 
          decoder_layers = 4,regularize = 0.0001, patch = 3, 
          batch = 256, aug = 5, save_every = 5):
    png_files = np.array(glob.glob("../data/ZTFBTS/hostImgs/*.png"))
    n_imgs = len(png_files)
    n_train = int(n_imgs * 0.8)
    training_list = png_files[:n_train]
    training_data = ImagePathDatasetAug(training_list.tolist(), factor = aug)
    training_loader = DataLoader(training_data, batch_size = batch, collate_fn = collate_fn_stack)


    img_encoder = HostImgTransceiverEncoder(img_size = 60,
                    bottleneck_length = bottlenecklen,
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    num_layers = encoder_layers,
                    sincosin = sincosin,
                    patch_size=patch).to(device)

    img_score = HostImgTransceiverScore(
        img_size = 60,
        bottleneck_dim = bottleneckdim,
        model_dim = model_dim,
        num_layers = decoder_layers,
        patch_size=patch,
        sincosin = sincosin
    ).to(device)

    mydaep = unimodaldaep(img_encoder, img_score, regularize = regularize).to(device)
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
            optimizer.zero_grad()
            loss = mydaep(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        this_epoch = np.array(losses).mean().item()
        epoch_loss.append(math.log(this_epoch))
        epoches.append(ep)
        if (ep+1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
            target_save = f"../ckpt/ZTF_daep_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_sincos{sincosin}_lr{lr}_epoch{ep+1}_batch{batch}_reg{regularize}_aug{aug}.pth"
            torch.save(mydaep, target_save)
            plt.plot(epoches, epoch_loss)
            plt.show()
            plt.savefig(f"./logs/ZTF_daep_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_sincos{sincosin}_lr{lr}_batch{batch}_reg{regularize}_aug{aug}.png")
            plt.close()
        progress_bar.set_postfix(loss=f"epochs:{ep}, {math.log(this_epoch):.4f}") 
        
  
            

if __name__ == '__main__':
    fire.Fire(train)







