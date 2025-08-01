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
from daep.data_util import SpectraDatasetFromnp, collate_fn_stack, to_device, padding_collate_fun
from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore


from daep.daep import unimodaldaep
import math 
import os
from tqdm import tqdm


def truncate_photometry(example):
    phot = example["photometry"]
    # Ensure it's a tensor
    if isinstance(phot, torch.Tensor):
        if phot.shape[0] > 800:
            phot = phot[:800, :]
    phot[:, 0] -= phot[:, 0].min()
    return {"photometry": phot}

class AstroM3Dataset(Dataset):
    def __init__(self, name = "full_42", which = "train", aug = None):
        # Load the default full dataset with seed 42
        assert aug is None or (aug >=1 and isinstance(aug, int)), "Augmentation has to be positive integer >=1 or None for not augmenting"
        self.dataset = load_dataset("../../AstroM3Dataset", name=name, trust_remote_code=True)[which]
        self.dataset.set_format(type="torch")
        self.aug = aug if aug is not None else 1
        self.which = which
        if which == "test" and self.aug > 1:
            print("We do not augment test")
            self.aug = 1
        #breakpoint()
        self.dataset = self.dataset.map(
            truncate_photometry,
            desc="Centering photometry",
            load_from_cache_file=False
            )
    def __len__(self):
        return self.aug * len(self.dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        
               
        
        return {
            "flux": (self.dataset[idx]['photometry'][:, 1] + (
                        (torch.randn_like(self.dataset[idx]['photometry'][:, 0]) * \
                        self.dataset[idx]['photometry'][:, 2]) if self.aug > 1 else 0.) - 22.6879)/27.7245 , # noise added only if augmentation and training
            "time": (self.dataset[idx]['photometry'][:, 0]- 788.0814)/475.3434 # [-3, 3] kinda aribitrary to match standardized range
        } 
        


def train(epoch=1000, lr = 2.5e-4, bottlenecklen = 4, bottleneckdim = 4, 
          concat = True, cross_attn_only = False,
          model_dim = 256, encoder_heads = 4, decoder_heads = 4,
          encoder_layers = 4, 
          decoder_layers = 4,regularize = 0.00, 
          batch = 16, aug = 1, save_every = 20):
    

    training_data = AstroM3Dataset(aug=aug)

    training_loader = DataLoader(training_data, batch_size = batch, collate_fn = padding_collate_fun(supply=['flux', 'wavelength', 'time'],
                                                           mask_by="flux", multimodal=False))
    
    photometryEncoder = photometricTransceiverEncoder(
                    num_bands = 1,
                    bottleneck_length = bottlenecklen,
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    ff_dim = model_dim,
                    num_layers = encoder_layers,
                    num_heads = encoder_heads,
                    concat = concat
                    ).to(device)

    photometryScore = photometricTransceiverScore(
                    num_bands = 1,
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    ff_dim = model_dim,
                    num_layers = decoder_layers,
                    num_heads = decoder_heads,
                    concat = concat,
                    cross_attn_only = cross_attn_only
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
            target_save = f"../ckpt/AstroM3photometry_daep_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_{encoder_heads}_{decoder_heads}_concat{concat}_corrattnonly{cross_attn_only}_lr{lr}_epoch{ep+1}_batch{batch}_reg{regularize}_aug{aug}.pth"
            torch.save(mydaep, target_save)
            plt.plot(epoches, epoch_loss)
            plt.show()
            plt.savefig(f"./logs/AstroM3photometry_daep_{bottlenecklen}-{bottleneckdim}-{encoder_layers}-{decoder_layers}-{model_dim}_{encoder_heads}_{decoder_heads}_concat{concat}_corrattnonly{cross_attn_only}_lr{lr}_batch{batch}_reg{regularize}_aug{aug}.png")
            plt.close()
        progress_bar.set_postfix(loss=f"epochs:{ep}, {math.log(this_epoch):.4f}") 
    


import fire           

if __name__ == '__main__':
    fire.Fire(train)
