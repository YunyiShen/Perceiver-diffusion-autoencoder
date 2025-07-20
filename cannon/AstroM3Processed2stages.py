from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torch
import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW
from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from daep.data_util import to_tensor, collate_fn_stack, to_device, padding_collate_fun
from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore
from daep.daep import multimodaldaep, modality_drop
from daep.tokenizers import photometryTokenizer, spectraTokenizer
from daep.Perceiver import PerceiverEncoder
from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore
from functools import partial

import math 
import os
from tqdm import tqdm


class AstroM3Procesed(Dataset):
    def __init__(self, name = "full_42", which = "train", aug = None):
        # Load the default full dataset with seed 42
        assert aug is None or (aug >=1 and isinstance(aug, int)), "Augmentation has to be positive integer >=1 or None for not augmenting"
        self.dataset = load_dataset("AstroMLCore/AstroM3Processed", name=name)[which]
        self.dataset.set_format(type="torch")
        self.aug = aug if aug is not None else 1
        self.which = which
        if which == "test" and aug > 1:
            print("We do not augment test")
            self.aug = 1

    def __len__(self):
        return self.aug * len(self.dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        
        res = {"flux": self.dataset[idx]['spectra'][1] + (
                        (torch.randn_like(self.dataset[idx]['spectra'][0]) * \
                        self.dataset[idx]['spectra'][2]) if self.aug > 1 else 0. )
                        , 
               "wavelength": self.dataset[idx]['spectra'][0], 
               "phase": torch.tensor(0.)}       
        
        photores = {"flux": self.dataset[idx]['photometry'][:, 1] + (
                        (torch.randn_like(self.dataset[idx]['photometry'][:, 0]) * \
                        self.dataset[idx]['photometry'][:, 2]) if self.aug > 1 else 0.) , # noise added only if augmentation and training
                    "time": self.dataset[idx]['photometry'][:, 0]
                    }
        #breakpoint()
        return {"spectra": res, "photometry": photores, 
                #"metadata": {"metadta": self.dataset[idx]['metadata'], 
                #             "label": self.dataset[idx]['label']}
                }
        
    

def train(epoch=1000, lr = 2.5e-4, bottlenecklen = 16, bottleneckdim = 16, 
          concat = True, 
          spectra_tokens = 128,
          photometry_tokens = 128,
          model_dim = 128, encoder_layers = 4, 
          decoder_layers = 4,regularize = 0.000, 
          dropping_prob = 0.3,
          batch = 16, aug = 3, save_every = 20):
    
    

    training_data = AstroM3Procesed(aug=aug)
    

    training_loader = DataLoader(training_data, batch_size = batch, 
                                 collate_fn = padding_collate_fun(supply = ['flux', 'wavelength', 'time'], 
                                                                  mask_by = "flux", 
                                                                  multimodal = True), 
                                 shuffle = True)
    
    #breakpoint()
    tokenizers = {
        "spectra": spectraTransceiverEncoder(
            bottleneck_length = spectra_tokens,
            bottleneck_dim = model_dim,
            model_dim = model_dim    
        ), 
        "photometry": photometricTransceiverEncoder(
            
            num_bands = 1, 
            bottleneck_length = photometry_tokens,
            bottleneck_dim = model_dim,
            model_dim = model_dim, 
        )
    }
    
    encoder = PerceiverEncoder(
                    bottleneck_length = bottlenecklen,
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    num_layers = encoder_layers,
                    ff_dim = model_dim,
                    num_heads = 4 
    )
    
    
    scores = {
        "spectra":spectraTransceiverScore(
                    bottleneck_dim = bottleneckdim,
                    model_dim = model_dim,
                    num_layers = decoder_layers,
                    concat = concat
                    ), 
        "photometry": photometricTransceiverScore(
            bottleneck_dim = bottleneckdim,
                 num_bands = 1,
                 model_dim = model_dim,
                 num_layers = decoder_layers,
                 concat = concat
        )
    }

    

    mydaep = multimodaldaep(tokenizers, encoder, scores, 
                            measurement_names = {"spectra":"flux", "photometry": "flux"}, 
                            modality_dropping_during_training = partial(modality_drop, p_drop=dropping_prob)).to(device)
    
    mydaep.train()
    optimizer = AdamW(mydaep.parameters(), lr=lr)
    epoch_loss = []
    epoches = []
    target_save = None
    progress_bar = tqdm(range(epoch))
    for ep in progress_bar:
        losses = []
        for x in tqdm(training_loader):
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
            target_save = f"../ckpt/AstroM3_daep2stages_{bottlenecklen}-{bottleneckdim}-{spectra_tokens}-{photometry_tokens}-{encoder_layers}-{decoder_layers}-{model_dim}_concat{concat}_lr{lr}_modaldropP{dropping_prob}_epoch{ep+1}_batch{batch}_reg{regularize}_aug{aug}.pth"
            torch.save(mydaep, target_save)
            plt.plot(epoches, epoch_loss)
            plt.show()
            plt.savefig(f"./logs/AstroM3_daep2stages_{bottlenecklen}-{bottleneckdim}-{spectra_tokens}-{photometry_tokens}-{encoder_layers}-{decoder_layers}-{model_dim}_concat{concat}_lr{lr}_modaldropP{dropping_prob}_batch{batch}_reg{regularize}_aug{aug}.png")
            plt.close()
        progress_bar.set_postfix(loss=f"epochs:{ep}, {math.log(this_epoch):.4f}") 
    


import fire           

if __name__ == '__main__':
    fire.Fire(train)
