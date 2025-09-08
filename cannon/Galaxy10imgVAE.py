import glob
import torch
from torch import nn
# dataloader
import numpy as np
# optimizer
from torch.optim import AdamW

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from VAESNe.ImageVAE import HostImgVAE
from VAESNe.training_util import training_step
from VAESNe.losses import elbo

from daep.data_util import  collate_fn_stack, to_device
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch
from torchvision import transforms
import h5py

class ImgH5DatasetAug(Dataset):
    def __init__(self, h5_path, size = 64, key = "images", indices=None,transform=None, factor = 1, preload = False):
        self.h5_path = h5_path
        self.key = key
        self.indices = indices
        self.preload = preload
        self._load_h5()
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(20),
            #transforms.RandomAffine(degrees = 15, translate = (0.05,0.05), scale = (0.75,1.25)),
            transforms.Resize((size, size)),   # Resize image to 128×128
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.factor = factor
        

    def _load_h5(self):
        with h5py.File(self.h5_path, 'r') as f:
            if self.preload:
                print("Preloading entire HDF5 dataset into memory...")
                self.data = f[self.key][:]
                print("Done")
            else:
                self.h5 = h5py.File(self.h5_path, 'r')
                self.data = self.h5[self.key]

    def __len__(self):
        return self.factor * (len(self.indices) if self.indices is not None else len(self.data))

    def __getitem__(self, idx):
        idx = idx % (len(self.indices) if self.indices is not None else len(self.data))
        real_idx = self.indices[idx] if self.indices is not None else idx
        image = Image.fromarray(self.data[real_idx])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([])
    
    def __del__(self):
        try:
            self.h5.close()
        except:
            pass 

def train(imgsize = 64, aug = 5, batch = 64,
          lr = 1e-3,
        epochs = 150,
        latent_len = 4,
        latent_dim = 4,
        beta = 0.5,
        patch_size = 2,
        model_dim = 32,
        num_layers = 4,
        hybrid = True,
        save_every = 20
        ):
    splits = np.load("../../Galaxy10/splits.npz")
    
    training_data = ImgH5DatasetAug("../../Galaxy10/Galaxy10_DECals.h5", 
                                    key="images", indices=splits["train"],
                                    size = imgsize,
                                    factor = aug, preload = True)
    training_loader = DataLoader(training_data, 
                                 
                                 batch_size = batch, 
                                 num_workers=1,  # adjust based on CPU cores
                                 pin_memory=True)

    #breakpoint()

    my_vaesne = HostImgVAE(
                    img_size = imgsize, 
                    latent_len = latent_len,
                    latent_dim = latent_dim,
                    
                    patch_size=patch_size, 
                    in_channels=3,
                    focal_loc = False,
                    model_dim = model_dim, 
                    num_heads = 4, 
                    ff_dim = model_dim, 
                    num_layers = num_layers,
                    dropout=0.1, 
                    selfattn=False, 
                    beta = beta,
                    hybrid = hybrid
    ).to(device)

    #breakpoint()

    optimizer = AdamW(my_vaesne.parameters(), lr=lr)
    all_losses = np.ones(epochs) + np.nan
    steps = np.arange(epochs)

    progress_bar = tqdm(range(epochs))
    target_save = None
    for i in progress_bar:
        loss = training_step(my_vaesne, optimizer, training_loader, elbo)
        all_losses[i] = loss
        if (i + 1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
            target_save = f'../ckpt/Galaxy10_vaesne_{latent_len}-{latent_dim}_{lr}_{i+1}_patch{patch_size}_beta{beta}_modeldim{model_dim}_numlayers{num_layers}_hybrid{hybrid}.pth'
            plt.plot(steps, all_losses)
            plt.show()
            plt.savefig(f"./logs/Galaxy10_vaesne_{latent_len}-{latent_dim}_{lr}_patch{patch_size}_beta{beta}_modeldim{model_dim}_numlayers{num_layers}_hybrid{hybrid}.png")
            plt.close()
            torch.save(my_vaesne, target_save)
        progress_bar.set_postfix(loss=f"epochs:{i}, {loss:.4f}")
        
        
import fire           

if __name__ == '__main__':
    fire.Fire(train)