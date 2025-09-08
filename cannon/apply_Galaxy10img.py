import torch 
from daep.daep import unimodaldaep
from daep.data_util import ImgH5DatasetAug, collate_fn_stack, to_device, to_np_cpu, save_dictlist
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import glob
from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import copy
from tqdm import tqdm


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


size = 50
ckpt = "Galaxy10_daep_8-8-4-4-256_sincosFalse_lr0.00025_epoch500_batch64_reg0.0_aug3_imgsize64"
trained_daep = torch.load(f"../ckpt/{ckpt}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)

for i, x in tqdm(enumerate(test_loader)):


    x = to_device(x)
    x_ori = copy.deepcopy(x)


    #torch.manual_seed(12345)
    #x_ori = to_np_cpu(x_ori)
    #recon = to_np_cpu(recon)


    
    rec = []

    for j in range(size):
        x = copy.deepcopy(x_ori)
        recon = trained_daep.reconstruct(x, ddim_steps = 200)
        recon = to_np_cpu(recon)
        recon['flux'] = recon['flux'] * 0.5+0.5
    
        rec.append(recon)

    x_ori = to_np_cpu(x_ori)
    x_ori['flux'] = x_ori['flux'] * 0.5+0.5

    save_dictlist(f"./res/Galaxy10/{ckpt}_rec_batch{i}.npz", rec)
    np.savez(f"./res/Galaxy10/{ckpt}_gt_batch{i}.npz",
         **x_ori
         )



'''
x_ori = x_ori['flux']
recon = recon['flux']
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
plt.savefig("./Galaxy10_recon.pdf")
plt.close()

'''