import torch
from torch import nn
from datasets import load_dataset
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
from torch.optim import AdamW
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from matplotlib import pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset


from VAESNe.SpectraVAE import BrightSpectraVAE, SpectraVAE
from VAESNe.training_util import training_step
from VAESNe.losses import elbo, m_iwae, _m_iwae
from VAESNe.data_util import multimodalDataset
from VAESNe.mmVAE import photospecMMVAE
from tqdm import tqdm

torch.manual_seed(0)
from daep.data_util import SpectraDatasetFromnp, collate_fn_stack, to_device, padding_collate_fun

pd = padding_collate_fun(supply=['flux', 'wavelength', 'time'], mask_by="flux", multimodal=False)
def padding_(batch):
    tmp = pd(batch)
    return tmp['flux'], tmp['wavelength'], tmp['phase'], tmp['mask']
    
    
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

    def __len__(self):
        return self.aug * len(self.dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        
        res = {"flux": (torch.log10(self.dataset[idx]['spectra'][:, 1] + (
                        (torch.randn_like(self.dataset[idx]['spectra'][:, 2]) * \
                        self.dataset[idx]['spectra'][:, 2]) if self.aug > 1 else 0. ) ) - 2.8766)/0.7795  # this is the mean
                        , 
               "wavelength": (self.dataset[idx]['spectra'][:, 0] - 6000.1543)/1548.8627, 
               "phase": torch.tensor(0.)}       
        
        #breakpoint()
        return res



### dataset ###

def train(aug = 1, 
    batch_size=16,
    lr = 1e-3, #2.5e-4
    epochs = 200,
    latent_len = 4,
    latent_dim = 4,
    beta = 0.5,
    model_dim = 32,
    num_heads = 4,
    num_layers = 4,
    save_every = 20,
    hidden_len = 256
    ):

    training_data = AstroM3Dataset(aug=aug)

    train_loader = DataLoader(training_data, batch_size = batch_size, 
                                 collate_fn = padding_)
    

    my_spectravae = SpectraVAE(
        # model parameters
        latent_len = latent_len,
        latent_dim = latent_dim,
        model_dim = model_dim, 
        num_heads = num_heads, 
        ff_dim = model_dim, 
        num_layers = num_layers,
        dropout = 0.1,
        selfattn = False, #True
        beta = beta,
        twostages_decoding = True,
        hidden_len = hidden_len,
        ).to(device)


    optimizer = AdamW(my_spectravae.parameters(), lr=lr)
    all_losses = np.ones(epochs) + np.nan
    steps = np.arange(epochs)

    target_save = None
    progress_bar = tqdm(range(epochs))
    for i in progress_bar:
        loss = training_step(my_spectravae, 
                    optimizer,
                    train_loader, 
                    loss_fn = elbo, 
                    multimodal = False)
        all_losses[i] = loss
        if (i + 1) % save_every == 0:
            if target_save is not None:
                os.remove(target_save)
                
            target_save = f'../ckpt/AstroM3_spectra_vaesne_{latent_len}-{latent_dim}-{model_dim}-{num_layers}_heads{num_heads}_hiddenlen{hidden_len}_{lr}_epoch{i+1}_batch{batch_size}_aug{aug}_beta{beta}.pth'
            plt.plot(steps, all_losses)
            plt.xlabel("training epochs")
            plt.ylabel("loss")
            plt.show()
            plt.savefig(f"./logs/AstroM3_spectra_vaesne_{latent_len}-{latent_dim}-{model_dim}-{num_layers}_heads{num_heads}_hiddenlen{hidden_len}_{lr}_batch{batch_size}_aug{aug}_beta{beta}.png")
            plt.close()
            torch.save(my_spectravae, target_save)
        progress_bar.set_postfix(loss=f"epochs:{i}, {loss:.4f}")


import fire           

if __name__ == '__main__':
    fire.Fire(train)