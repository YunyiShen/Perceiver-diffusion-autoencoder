import torch 
from daep.daep import unimodaldaep
from daep.data_util import ImgH5DatasetAug, collate_fn_stack, to_device
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import glob
from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import copy


torch.manual_seed(42)
splits = np.load("../../Galaxy10/splits.npz")
    
test_data = ImgH5DatasetAug("../../Galaxy10/Galaxy10_DECals.h5", 
                                    key="images", indices=splits["test"],
                                    size = 64,
                                    factor = 1, preload = True)
test_loader = DataLoader(test_data, 
                                 shuffle = False,
                                 batch_size = 64, 
                                 num_workers=1,  # adjust based on CPU cores
                                 pin_memory=True,  # speeds up transfer to GPU
                                 collate_fn = collate_fn_stack)




x = to_device(next(iter(test_loader)))
x_ori = copy.deepcopy(x)


trained_vae = torch.load("../ckpt/Galaxy10_vaesne_8-8_0.00025_500_patch4_beta0.5_modeldim256_numlayers4_hybridTrue.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)
torch.manual_seed(42)
#recon = trained_daep.reconstruct((x['flux'], torch.tensor([])), K = 1)

#x_ori = x_ori['flux']
#recon = recon[0]


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
            recon = trained_vae.reconstruct((x['flux'], torch.tensor([])), K=10)* 0.5+0.5
            all_rec.append(recon.detach().cpu().numpy())
#breakpoint()
    np.savez(f"./res/Galaxy10/Galaxy10_vaesne_8-8_0.00025_500_patch4_beta0.5_modeldim256_numlayers4_hybridTrue_batch{i}.npz",
         rec = np.concatenate(all_rec),
         flux = x_ori['flux'].detach().cpu().numpy()* 0.5+0.5
         )




'''
breakpoint()
plt.rcParams.update({'font.size': 20})  # sets default font size
fig, axes = plt.subplots(2, 10, figsize=(20, 6))  # 4 rows, 5 columns
for i, idx in enumerate(x_ori):
    axes[0,i].imshow(x_ori[i].permute(1, 2, 0).detach().cpu().numpy()/2 + 0.5)
    axes[1,i].imshow(recon[i].permute(1, 2, 0).detach().cpu().numpy()/2 + 0.5)
    #axes[2,i].imshow((recon[i].permute(1, 2, 0).detach().cpu().numpy()-x_ori[i].permute(1, 2, 0).detach().cpu().numpy())/x_ori[i].permute(1, 2, 0).detach().cpu().numpy())
    axes[0, i].set_xticks([])         # removes ticks
    axes[1, i].set_xticklabels([])    # removes tick labels
    #axes[2, i].set_xticklabels([])    # removes tick labels
    axes[0, i].set_yticks([])         # removes ticks
    axes[1, i].set_yticklabels([])    # removes tick labels
    #axes[2, i].set_yticklabels([])    # removes tick labels

axes[0, 0].set_ylabel("original")
axes[1, 0].set_ylabel("reconstruction")
#axes[0, 2].set_ylabel("relative error")
plt.tight_layout()
plt.show()
plt.savefig("./Galaxy10_recon_vae.pdf")
plt.close()
'''