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







from torch.nn.utils.rnn import pad_sequence

def multimodal_padding(list_of_modal_dict, supply = ["flux", "wavelength", "time", "band"], 
                       mask_by = "flux", max_len = None):
    modalities = [*list_of_modal_dict[0]] # img, spectra etc.
    res = {}
    for modal in modalities:
        tensor_keys = [*list_of_modal_dict[0][modal]] # e.g., flux, wavelength, phase etc
        this_modality = {}
        for tensor_key in tensor_keys:
            padded_tensor = [this_dict[tensor_key] for this_dict in list_of_modal_dict]
            if tensor_key in supply and "mask" not in tensor_keys: # we do not pad if mask is given
                if tensor_key == mask_by:
                    length = torch.tensor([len(x) for x in padded_tensor])
                    max_len = length.max()
                    mask = torch.arange(max_len)[None, :] >= length[:, None]
                    this_modality['mask'] = mask
                padded_tensor = pad_sequence(padded_tensor, batch_first = True, padding_value = 0)
            else:
                padded_tensor = torch.stack(padded_tensor, axis = 0)
            this_modality[tensor_key] = padded_tensor
        res[modal] = this_modality
    return res


def unimodal_padding(list_of_modal_dict, supply = ['flux', 'wavelength', 'time', "band"], mask_by = "flux"):
    tensor_keys = [*list_of_modal_dict[0]] # e.g., flux, wavelength, phase etc
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
        else:
            padded_tensor = torch.stack(padded_tensor, axis = 0)
        this_modality[tensor_key] = padded_tensor
    return this_modality


class padding_collate_fun():
    def __init__(self, supply = ['flux', 'wavelength', 'time'], mask_by = "flux", multimodal = False):
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

 

