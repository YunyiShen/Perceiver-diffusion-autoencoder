from pathlib import Path
from typing import Any, Dict
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from daep.data_util import padding_collate_fun

def create_dataloader(config, data_types, data_names, train=True):
    """
    Create dataset(s) and return DataLoader(s).

    Parameters
    ----------
    config : dict
        Configuration dict with keys under ``"data"`` and ``"training"``/``"testing"``.
    data_types : list[str]
        One or two items: ``["spectra"]``, ``["lightcurves"]`` or both.
    data_names : list[str]
        Dataset names aligned with ``data_types``.
    train : bool, default=True
        If True, returns train and validation loaders. If False, returns a test loader.

    Returns
    -------
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader] | torch.utils.data.DataLoader
        If ``train`` is True, returns ``(training_loader, validation_loader)``.
        Otherwise returns ``testing_loader``.
    """
    if len(data_types) == 1 and data_types[0] == "spectra":
        data_path = Path(config["data"]["data_path"]) / 'spectra'
        test_name = data_names[0]
        from daep.datasets.GALAHspectra_dataset import GALAHDatasetProcessed
        training_data = GALAHDatasetProcessed(
            data_dir=data_path / test_name, 
            train=train, 
        )
        collate_fn = padding_collate_fun(supply=['flux', 'wavelength', 'time'], mask_by="flux", multimodal=False)
        
    elif len(data_types) == 1 and data_types[0] == "lightcurves":
        data_path = Path(config["data"]["data_path"]) / 'lightcurves'
        test_name = data_names[0]
        from daep.datasets.TESSlightcurve_dataset import TESSDatasetProcessed
        training_data = TESSDatasetProcessed(
            data_dir=data_path / test_name, 
            train=train, 
        )
        collate_fn = padding_collate_fun(supply=['flux', 'time'], mask_by="flux", multimodal=False)
        
    elif len(data_types) == 2 and 'spectra' in data_types and 'lightcurves' in data_types:
        data_path = Path(config["data"]["data_path"])
        lightcurve_index = data_types.index('lightcurves')
        spectra_index = data_types.index('spectra')
        lightcurve_test_name = data_names[lightcurve_index]
        spectra_test_name = data_names[spectra_index]
        from daep.datasets.TESSGALAHspeclc_dataset import TESSGALAHDatasetProcessed
        from daep.datasets.TESSlightcurve_dataset import TESSDatasetProcessed
        from daep.datasets.GALAHspectra_dataset import GALAHDatasetProcessed
        dataset_lc = TESSDatasetProcessed(
            data_dir=data_path / 'lightcurves' / lightcurve_test_name, 
            train=train, 
        )
        dataset_spectra = GALAHDatasetProcessed(
            data_dir=data_path / 'spectra' / spectra_test_name, 
            train=train, 
        )
        training_data = TESSGALAHDatasetProcessed(
            dataset_lc, 
            dataset_spectra, 
        )
        collate_fn = padding_collate_fun(supply=['flux', 'wavelength', 'time'], mask_by="flux", multimodal=True)
    
    else:
        raise ValueError(f"Unsupported data_types: {data_types}")
    
    if train:
        val_split = 0.1  # 10% of the data for validation
        train_idx, validation_idx = train_test_split(
            np.arange(len(training_data)),
            test_size=val_split,
            random_state=42,
            shuffle=True
        )

        full_dataset = training_data
        training_data = Subset(full_dataset, train_idx)
        validation_data = Subset(full_dataset, validation_idx)

        training_loader = DataLoader(
            training_data,
            batch_size=config["training"]["batch"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )

        validation_loader = DataLoader(
            validation_data,
            batch_size=config["training"]["batch"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        return training_loader, validation_loader
    else:
        testing_loader = DataLoader(
            training_data, 
            batch_size=config["testing"]["batch"], 
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2, 
            pin_memory=True
        )
        return testing_loader
