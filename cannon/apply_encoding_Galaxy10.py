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
which = "test"
splits = np.load("../../Galaxy10/splits.npz")
    
test_data = ImgH5DatasetAug("../../Galaxy10/Galaxy10_DECals.h5", 
                                    key="images", indices=splits[which],
                                    size = 64,
                                    factor = 1, preload = True)

batch_size = 256

test_loader = DataLoader(test_data, 
                         shuffle = False,
                         batch_size = batch_size, 
                         num_workers=1,  # adjust based on CPU cores
                         pin_memory=True,  # speeds up transfer to GPU
                         collate_fn = collate_fn_stack)


#breakpoint()
import h5py
h5 = h5py.File(test_data.h5_path, "r")
types_ = np.array(h5['ans'])
redshift_ = np.array(h5['redshift'])


ckpt = "Galaxy10_daep_8-8-4-4-256_sincosFalse_lr0.00025_epoch500_batch64_reg0.0_aug3_imgsize64"
trained_daep = torch.load(f"../ckpt/{ckpt}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)
vae_ckpt = "Galaxy10_vaesne_8-8_0.00025_500_patch4_beta0.5_modeldim256_numlayers4_hybridTrue"
trained_vae = torch.load(f"../ckpt/{vae_ckpt}.pth",
                         map_location=torch.device('cpu'), weights_only = False).to(device)
#torch.manual_seed(4125)
#breakpoint()
from tqdm import tqdm
for i, x in tqdm(enumerate(test_loader)):

    types = types_[(i*batch_size):min((i+1)*batch_size, len(test_data))]
    redshift = redshift_[(i*batch_size):min((i+1)*batch_size, len(test_data))]

    x = to_device(x)




    torch.manual_seed(42)
    encode = trained_daep.encode(x)
    encode = to_np_cpu(encode)
    encode = encode.reshape(encode.shape[0], -1)



    #breakpoint()

    x_vae = (x['flux'], torch.tensor([]))



    encode_vae = trained_vae.encode(x_vae)
    encode_vae = to_np_cpu(encode_vae)
    encode_vae = encode_vae.reshape(encode_vae.shape[0], -1)


    np.savez(f"./encodes/Galaxy10/{ckpt}_{which}_batch{i}.npz",
         encode = encode,
         types = types,
         redshift = redshift
         )


    np.savez(f"./encodes/Galaxy10/{vae_ckpt}_{which}_batch{i}.npz",
         encode = encode_vae,
         types = types,
         redshift = redshift
         )



