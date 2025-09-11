import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
# optimizer
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


import copy
which = "test"
### dataset ###
test_data = np.load(f"../data/{which}_data_align_with_simu_minimal.npz")

### spectra ###
flux, wavelength, mask = test_data['flux'], test_data['wavelength'], test_data['mask']
phase = test_data['phase']

wavelength_mean, wavelength_std = test_data['wavelength_mean'], test_data['wavelength_std']
flux_mean, flux_std = test_data['flux_mean'], test_data['flux_std']
phase_mean, phase_std = test_data['spectime_mean'], test_data['spectime_std']
phototime_mean, phototime_std = test_data['combined_time_mean'], test_data['combined_time_std']
photoflux_mean, photoflux_std = test_data['combined_mean'], test_data['combined_std']
#breakpoint()


flux_test = torch.tensor(flux, dtype=torch.float32)
wavelength_test = torch.tensor(wavelength, dtype=torch.float32)
mask_test = torch.tensor(mask == 0)
phase_test = torch.tensor(phase, dtype=torch.float32)

spectra_train_dataset = TensorDataset(flux_test, wavelength_test, phase_test, mask_test)
torch.manual_seed(42)
test_loader = DataLoader(spectra_train_dataset, batch_size=128, shuffle=False)
#test_loader = DataLoader(test_data, batch_size = 20, collate_fn = collate_fn_stack, shuffle = True)
#breakpoint()
### 
trained_vae = torch.load("../ckpt/ZTF_spectra_vaesne_4-4-128-4_heads8_0.00025_epoch2000_batch128_aug5_beta0.1.pth", # trained with K=1 on iwae
                         map_location=torch.device('cpu'), weights_only = False).to(device)



#x = next(iter(test_loader))

from tqdm import tqdm
for i, x in tqdm(enumerate(test_loader)):
    
    x = to_device(x)
    x_ori = copy.deepcopy(x)
    #breakpoint()

    rec = []

    #breakpoint()
    all_rec = []

    x_ori = copy.deepcopy(x)




    #breakpoint()
    all_rec = []
    for j in range(10):
        with torch.no_grad():
            recon = trained_vae.reconstruct(x, K=5)
            all_rec.append(recon.detach().cpu().numpy())
#breakpoint()
    np.savez(f"./res/ZTFspectra/ZTF_spectra_vaesne_4-4-128-4_heads8_0.00025_epoch2000_batch128_aug5_beta0.1_batch{i}.npz",
         rec = np.concatenate(all_rec)* flux_std + flux_mean,
         wavelength = x_ori[1].detach().cpu().numpy()* wavelength_std + wavelength_mean,
         flux = x_ori[0].detach().cpu().numpy()* flux_std + flux_mean,
         mask = x_ori[3].detach().cpu().numpy()
         )

'''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 5, figsize=(20, 12))  # 4 rows, 5 columns
axes = axes.flatten()
for i in range(20):
    axes[i].plot(x_ori[1][i][~x_ori[3][i]] * wavelength_std + wavelength_mean,
                 x_ori[0][i][~x_ori[3][i]] * flux_std + flux_mean,
                 color = "red",
                 label = "ground truth" if i == 0 else None
                 )
    axes[i].plot(x_ori[1][i][~x_ori[3][i]] * wavelength_std + wavelength_mean,
                 recon[0][i][~x_ori[3][i]] * flux_std + flux_mean,
                 color = "blue",
                 label = "vae" if i == 0 else None
                 )
    axes[i].set_xlabel("wavelength (A)")


axes[0].legend()
axes[0].set_ylabel("logFnu")
axes[5].set_ylabel("logFnu")
plt.tight_layout()
fig.show()
fig.savefig("spectra_recon_vae.pdf")
plt.close()

'''