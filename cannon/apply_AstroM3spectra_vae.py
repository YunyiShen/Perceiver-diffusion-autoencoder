import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
# optimizer
from datasets import load_dataset
from torch.optim import AdamW
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from matplotlib import pyplot as plt


from VAESNe.SpectraVAE import SpectraVAE
from VAESNe.PhotometricVAE import PhotometricVAE
from VAESNe.training_util import training_step
from VAESNe.losses import elbo, m_iwae, _m_iwae
from VAESNe.data_util import multimodalDataset
from VAESNe.mmVAE import photospecMMVAE
from daep.data_util import SpectraDatasetFromnp, collate_fn_stack, to_device, padding_collate_fun

pd = padding_collate_fun(supply=['flux', 'wavelength', 'time'], mask_by="flux", multimodal=False)
def padding_(batch):
    tmp = pd(batch)
    return tmp['flux'], tmp['wavelength'], tmp['phase'], tmp['mask']
class AstroM3Dataset(Dataset):
    def __init__(self, name = "full_42", which = "test", aug = None):
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


import copy

test_data = AstroM3Dataset(which = "test")
torch.manual_seed(42)
test_loader = DataLoader(test_data, batch_size = 64, 
                                 collate_fn = padding_, shuffle = False)
trained_vae = torch.load("../ckpt/AstroM3_spectra_vaesne_4-8-128-6_heads8_hiddenlen256_0.00025_epoch200_batch128_aug1_beta0.1.pth", # trained with K=1 on iwae
                         map_location=torch.device('cpu'), weights_only = False).to(device)
#test_loader = DataLoader(test_data, batch_size = 20, collate_fn = collate_fn_stack, shuffle = True)
#breakpoint()
### 
from tqdm import tqdm
for i, x in tqdm(enumerate(test_loader)):
    
    x = to_device(x)
    x_ori = copy.deepcopy(x)
    #breakpoint()

    rec = []

    #breakpoint()
    all_rec = []
    for j in range(5):
        with torch.no_grad():
            recon = trained_vae.reconstruct(x, K=10)
            all_rec.append(recon.detach().cpu().numpy())
#breakpoint()
    np.savez(f"./res/AstroM3spectra/AstroM3_spectra_vaesne_4-8-128-6_heads8_hiddenlen256_0.00025_epoch200_batch128_aug1_beta0.1_rec_batch{i}.npz",
         rec = np.concatenate(all_rec),
         wavelength = x_ori[1].detach().cpu().numpy(),
         flux = x_ori[0].detach().cpu().numpy(),
         mask = x_ori[3].detach().cpu().numpy()
         )

'''

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 5, figsize=(50, 12))
axes = axes.flatten()
for i in range(20):
    axes[i].plot(x_ori[1][i][~x_ori[3][i]] * 1548.8627 + 6000.1543, 
                 x_ori[0][i][~x_ori[3][i]] * 0.7795 + 2.8766,
                 color = "red",
                 label = "ground truth" if i == 0 else None
                 )
    axes[i].plot(x_ori[1][i][~x_ori[3][i]] * 1548.8627 + 6000.1543,
                 recon[0][i][~x_ori[3][i]] * 0.7795 + 2.8766,
                 color = "blue",
                 label = "vae" if i == 0 else None
                 )
    axes[i].set_xlabel("wavelength (A)")


axes[0].legend()
axes[0].set_ylabel("logFnu")
axes[5].set_ylabel("logFnu")
plt.tight_layout()
fig.show()
fig.savefig("AstroM3spectra_recon_vae.pdf")
plt.close()

'''