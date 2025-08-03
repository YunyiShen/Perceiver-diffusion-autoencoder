from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import re
from collections import defaultdict

def to_device(data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Recursively moves all torch.Tensors in a nested structure to the given device.
    Handles arbitrary nesting of dicts and lists (or tuples).
    """
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    else:
        return data  # unchanged if not a tensor/list/dict/tuple


def to_np_cpu(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif isinstance(data, dict):
        return {k: to_np_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_np_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_np_cpu(v) for v in data)
    else:
        return data  # unchanged if not a tensor/list/dict/tuple
    
def to_tensor(data):
    if torch.is_tensor(data):
        return torch.tensor(data)
    elif isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_tensor(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_tensor(v) for v in data)
    else:
        return data  # unchanged if not a tensor/list/dict/tuple

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
        return {"flux": image} 

    def __del__(self):
        try:
            self.h5.close()
        except:
            pass

import pandas as pd
class PLAsTiCCfromcsvaug(Dataset):
    def __init__(self, csv_path, aug = 1):
        # Load CSV once
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["detected"] == 1]
        
        # Sort by object_id to make grouping easier
        self.df.sort_values("object_id", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        #breakpoint()

        # Convert object_id column to numpy array for fast indexing
        self.object_ids = self.df["object_id"].values

        # Get unique object IDs and their starting indices in the sorted array
        self.unique_ids, self.start_indices = np.unique(self.object_ids, return_index=True)

        # For convenience, append end index (last + 1)
        self.end_indices = np.append(self.start_indices[1:], len(self.df))
        self.aug = aug

    def __len__(self):
        return len(self.unique_ids) * self.aug

    def __getitem__(self, idx):
        # Get row indices for this object_id
        idx = idx % len(self.unique_ids)
        start = self.start_indices[idx]
        end = self.end_indices[idx]

        star_data = self.df.iloc[start:end]

        # Example: extract relevant columns as tensors
        time = torch.tensor(star_data["mjd"].values - star_data["mjd"].values.min(), dtype=torch.float32)
        band = torch.tensor(star_data['passband'].values, dtype = torch.long)
        if self.aug == 1:
            flux = torch.tensor(star_data["flux"].values, dtype=torch.float32)
        else:
            flux = torch.tensor(star_data["flux"].values, dtype=torch.float32) + torch.randn_like(time) * \
                                torch.tensor(star_data["flux_err"].values, dtype=torch.float32)
        

        return {
            
            "flux": (torch.arcsinh(flux) - 1.9748)/5.6163,
            "time": time-time.min(),
            "band": band
        }


class ImagePathDatasetAug(Dataset):
    def __init__(self, image_paths, transform=None, factor = 10):
        """
        Args:
            image_paths (list of str): List of image file paths.
            transform (callable, optional): Transform to apply to each image.
        """
        self.factor = factor
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(20),
            transforms.RandomAffine(degrees = 15, translate = (0.05,0.05), scale = (0.75,1.25)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) 

    def __len__(self):
        return len(self.image_paths) * self.factor

    def __getitem__(self, idx):
        idx = idx % len(self.image_paths)
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return {"flux": image}

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths (list of str): List of image file paths.
            transform (callable, optional): Transform to apply to each image.
        """
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
                                            transforms.ToTensor(), 
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
                                            ]) 

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return {"flux": image}

class SpectraDatasetFromnp(Dataset):
    def __init__(self, flux, wavelength, phase, mask = None):
        self.flux = torch.tensor(flux)
        self.wavelength = torch.tensor(wavelength)
        self.phase = torch.tensor(phase)
        self.mask = torch.tensor(mask) if mask is not None else mask
    
    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        
        res = {"flux": self.flux[idx], "wavelength": self.wavelength[idx], "phase": self.phase[idx]}
        if self.mask is not None:
            res['mask'] = self.mask[idx]
        return res



class PhotoSpectraDatasetFromnp(Dataset):
    def __init__(self, flux, wavelength, phase, 
                 photoflux, phototime, photoband
                 ,mask = None, photomask = None):
        self.flux = torch.tensor(flux)
        self.wavelength = torch.tensor(wavelength)
        self.phase = torch.tensor(phase)
        self.mask = torch.tensor(mask) if mask is not None else mask
        
        self.photoflux = torch.tensor(photoflux)
        self.phototime = torch.tensor(phototime)
        self.photoband = torch.tensor(photoband, dtype = torch.long)
        self.photomask = torch.tensor(photomask) if photomask is not None else photomask
    
    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        
        res = {"flux": self.flux[idx], "wavelength": self.wavelength[idx], "phase": self.phase[idx]}
        if self.mask is not None:
            res['mask'] = self.mask[idx]
        
        
        photores = {"flux": self.photoflux[idx], "time": self.phototime[idx], "band": self.photoband[idx]}
        if self.photomask is not None:
            photores['mask'] = self.photomask[idx]
        return {"spectra": res, "photometry": photores}



from torch.nn.utils.rnn import pad_sequence

def multimodal_padding(list_of_modal_dict, supply = ["flux", "wavelength", "time", "band"], 
                       mask_by = "flux", max_len = None, modalities_to_pad = None):
    modalities = [*list_of_modal_dict[0]] if modalities_to_pad is None else modalities_to_pad # img, spectra etc.
    res = {}
    for modal in modalities:
        tensor_keys = [*list_of_modal_dict[0][modal]] # e.g., flux, wavelength, phase etc
        this_modality = {}
        for tensor_key in tensor_keys:
            padded_tensor = [this_dict[modal][tensor_key] for this_dict in list_of_modal_dict]
            if tensor_key in supply and "mask" not in tensor_keys: # we do not pad if mask is given
                if tensor_key == mask_by:
                    length = torch.tensor([len(x) for x in padded_tensor])
                    max_len = length.max()
                    mask = torch.arange(max_len)[None, :] >= length[:, None]
                    this_modality['mask'] = mask
                padded_tensor = pad_sequence(padded_tensor, batch_first = True, padding_value = 0)
                nonfinite = ~torch.isfinite(padded_tensor)
                padded_tensor[nonfinite] = 0
                this_modality['mask'] = torch.logical_or(this_modality['mask'], nonfinite)
            else:
                padded_tensor = torch.stack(padded_tensor, dim=0)
                nonfinite = ~torch.isfinite(padded_tensor)
                padded_tensor[nonfinite] = 0 # sanitize
            this_modality[tensor_key] = padded_tensor
        res[modal] = this_modality
    return res


def unimodal_padding(list_of_modal_dict, supply = ['flux', 'wavelength', 'time', "band"], mask_by = "flux"):
    tensor_keys = [*list_of_modal_dict[0]] # e.g., flux, wavelength, phase etc
    assert mask_by in tensor_keys, "mask_by has to be in data"
    tensor_keys.remove(mask_by)
    tensor_keys.insert(0, mask_by)
    this_modality = {}
    for tensor_key in tensor_keys:
        padded_tensor = [this_dict[tensor_key] for this_dict in list_of_modal_dict]
        if tensor_key in supply and "mask" not in tensor_keys:
            if tensor_key == mask_by:
                length = torch.tensor([len(x) for x in padded_tensor])
                max_len = length.max()
                mask = torch.arange(max_len)[None, :] >= length[:, None]
                this_modality['mask'] = mask
            padded_tensor = pad_sequence(padded_tensor, batch_first = True, padding_value = 0)
            nonfinite = ~torch.isfinite(padded_tensor)
            padded_tensor[nonfinite] = 0
            #breakpoint()
            this_modality['mask'] = torch.logical_or(this_modality['mask'], nonfinite)
        else:
            padded_tensor = torch.stack(padded_tensor, axis = 0)
        this_modality[tensor_key] = padded_tensor
    return this_modality


class padding_collate_fun():
    def __init__(self, supply = ['flux', 'wavelength', 'time', "band"], mask_by = "flux", multimodal = True):
        self.supply = supply
        self.mask_by = mask_by
        self.multimodal = multimodal
    
    def __call__(self, batch):
        if self.multimodal:
            return multimodal_padding(batch, self.supply, self.mask_by)
        return unimodal_padding(batch, self.supply, self.mask_by)






def collate_fn_stack(batch):
    """
    Collate function to be used in DataLoader when dataset returns dictionaries.
    Concatenates each key's values along the first dimension.
    """


    collated = defaultdict(list)

    for sample in batch:
        for key, value in sample.items():
            collated[key].append(value)

    # Convert list of values to tensors and concatenate
    for key in collated:
        collated[key] = torch.stack(collated[key], dim=0)

    return dict(collated)

 

