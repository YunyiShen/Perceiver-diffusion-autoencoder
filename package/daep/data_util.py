from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import re
from collections import defaultdict


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



def collate_fn_concat(batch):
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
        collated[key] = torch.cat(collated[key], dim=0)

    return dict(collated)



def collate_fn_nested_concat(batch):
    """
    Collate function for nested dictionaries where each sample is:
    {
        'modal1': {'img': tensor},
        'modal2': {'flux': tensor, 'time': tensor},
        ...
    }
    Concatenates all leaf tensors across batch dimension (dim=0).
    """
    from collections import defaultdict
    import torch

    collated = defaultdict(lambda: defaultdict(list))

    for sample in batch:
        for modality, fields in sample.items():
            for key, value in fields.items():
                collated[modality][key].append(value)

    # Concatenate along batch dimension
    for modality in collated:
        for key in collated[modality]:
            collated[modality][key] = torch.cat(collated[modality][key], dim=0)

    return dict(collated)    

