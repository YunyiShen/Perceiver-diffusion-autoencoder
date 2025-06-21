import torch 
from daep.daep import unimodaldaep
from daep.data_util import ImagePathDataset, collate_fn_stack, to_device, to_np_cpu
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import glob
from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import copy


png_files = np.array(glob.glob("../data/ZTFBTS/hostImgs/*.png"))
n_imgs = len(png_files)
n_train = int(n_imgs * 0.8)
test_list = png_files[n_train:]
test_data = ImagePathDataset(test_list)
test_loader = DataLoader(test_data, batch_size = 10, collate_fn =  collate_fn_stack,shuffle=True)

x = to_device(next(iter(test_loader)))
x_ori = copy.deepcopy(x)


trained_daep = torch.load("../ckpt/ZTF_daep_16-8-64_lr0.00025_epoch200_batch256_reg0.0001.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)

recon = trained_daep.reconstruct(x)

x_ori = x_ori['flux']
recon = recon['flux']
fig, axes = plt.subplots(2, 10, figsize=(20, 6))  # 4 rows, 5 columns
for i, idx in enumerate(x_ori):
    axes[0,i].imshow(x_ori[i].permute(1, 2, 0).detach().cpu().numpy()/2 + 0.5)
    axes[1,i].imshow(recon[i].permute(1, 2, 0).detach().cpu().numpy()/2 + 0.5)

plt.tight_layout()
plt.show()
plt.savefig("./hostimg_recon.pdf")
plt.close()