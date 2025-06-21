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
from daep.data_util import ImagePathDataset, collate_fn_stack, to_device
from daep.ImgLayers import HostImgTransceiverEncoder, HostImgTransceiverScore
from daep.daep import unimodaldaep
from tqdm import tqdm
import os
import fire


#breakpoint()

def train(epoch=200, lr = 2.5e-4, bottlenecklen = 4, bottleneckdim = 4, regularize = 0.01, patch = 3, save_every = 5):

    png_files = np.array(glob.glob("../data/ZTFBTS/hostImgs/*.png"))
    n_imgs = len(png_files)
    n_train = int(n_imgs * 0.8)
    training_list = png_files[:n_train]
    training_data = ImagePathDataset(training_list.tolist())
    test_list = png_files[n_train:]
    test_data = ImagePathDataset(test_list)
    training_loader = DataLoader(training_data, batch_size = 64, collate_fn = collate_fn_stack)
    test_loader = DataLoader(test_data, batch_size = 8, collate_fn =  collate_fn_stack)


    img_encoder = HostImgTransceiverEncoder(img_size = 60,
                    bottleneck_length = bottlenecklen,
                    bottleneck_dim = bottleneckdim,
                    patch_size=patch).to(device)

    img_score = HostImgTransceiverScore(
        img_size = 60,
        bottleneck_dim = bottleneckdim,
        patch_size=patch
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
        epoch_loss.append(this_epoch)
        epoches.append(ep)
        if (ep+1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
            target_save = f"../ckpt/ZTF_daep_{bottlenecklen}-{bottleneckdim}_lr{lr}_epoch{ep+1}_reg{regularize}.pth"
            torch.save(mydaep, target_save)
            plt.plot(epoches, epoch_loss)
            plt.show()
            plt.savefig(f"./logs/ZTF_daep_{bottlenecklen}-{bottleneckdim}_lr{lr}_reg{regularize}.png")
            plt.close()
        progress_bar.set_postfix(loss=f"epochs:{ep}, {this_epoch:.4f}") 
        
  
            

if __name__ == '__main__':
    fire.Fire(train)







