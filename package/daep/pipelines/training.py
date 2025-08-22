import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, DistributedSampler, Subset
import numpy as np
# optimizer
from torch.optim import AdamW
# Multi-GPU support
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from daep.data_util import to_device, padding_collate_fun
from daep.LitWrapperAll import daepClassifierUnimodal, daepClassifierMultimodal, daepReconstructorUnimodal, daepReconstructorMultimodal
import math 
import os
import json
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from sklearn.model_selection import train_test_split

from daep.utils.general_utils import detect_env, load_config, update_config
from daep.utils.train_utils import LossLogger, AccuracyLogger
from daep.datasets.dataloaders import create_dataloader, get_weights_and_num_classes, update_config_with_num_classes
from daep.LitWrapperAll import PhotClassifierUnimodal

ENV = detect_env()
DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

def initialize_model(model_type, data_types, config, weights=None):
    if model_type == 'classifier':
        if len(data_types) == 1:
            model = daepClassifierUnimodal(config=config, class_weights=weights)
        else:
            model = daepClassifierMultimodal(config=config, class_weights=weights)
    elif model_type == 'reconstructor':
        if len(data_types) == 1:
            model = daepReconstructorUnimodal(config=config)
        else:
            model = daepReconstructorMultimodal(config=config)
    elif model_type == 'photclassifier':
        model = PhotClassifierUnimodal(config=config)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    return model


def train_worker(model, training_loader, validation_loader, config, output_dir, world_size):
    """
    Worker function for distributed training.
    
    Parameters
    ----------
    rank : int
        Rank of the current process.
    world_size : int
        Total number of processes.
    config : Dict[str, Any]
        Configuration dictionary containing all training parameters.
    data_type : str, optional
        "spectra" or "lightcurves" or "both" to specify the type of data to train on. Defaults to "spectra".
    """
    
    # Initialize logger
    logger = CSVLogger(output_dir, name=model.model_name(), version=model.model_instance_str())

    # Initialize callbacks
    eval_metric = 'val_loss'
    checkpoint_callback = L.callbacks.ModelCheckpoint(monitor=eval_metric, save_top_k=4, mode='min', filename='{epoch:02d}-{val_loss:.2f}') # type: ignore
    accuracy_logger = AccuracyLogger()
    loss_logger = LossLogger()
    if isinstance(model, daepClassifierUnimodal) or isinstance(model, daepClassifierMultimodal) or isinstance(model, PhotClassifierUnimodal):
        callbacks = [checkpoint_callback, loss_logger, accuracy_logger]
    else:
        callbacks = [checkpoint_callback, loss_logger]
    
    trainer = L.Trainer(max_epochs=config["training"]["epochs"],
                        min_epochs=max(10, config["training"]["epochs"] // 5),
                        accelerator=config["training"]["accelerator"],
                        strategy='ddp_find_unused_parameters_true',  # Use DistributedDataParallel for multi-GPU training
                        devices=world_size,  # Use all 4 GPUs
                        callbacks=callbacks,
                        logger=logger)
    
    # Start training
    if config['train_from_checkpoint']:
        trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=validation_loader,
                    ckpt_path=config['checkpoint_path'])
    else:
        trainer.fit(model=model, train_dataloaders=training_loader, val_dataloaders=validation_loader)
    
    print("Training complete")
    

def train(config_path: str, model_type: str, **kwargs):
    """
    Train a unimodal DAEP model on either spectra or lightcurves using configuration from file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration JSON file.
    model_type : str, optional
        "classifier" or "reconstructor" to specify the type of model to train.
    **kwargs : dict
        Optional keyword arguments to override config values.
        Useful for quick parameter adjustments without modifying the config file.
        
    Examples
    --------
    >>> train()  # Use default config_reconstruction_default.yaml
    >>> train("my_config.yaml")  # Use custom config file
    >>> train(epoch=100, lr=1e-4)  # Override specific parameters
    """
    # Set environment variables for debugging if needed
    import os
    if "TORCH_DISTRIBUTED_DEBUG" not in os.environ:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    
    L.seed_everything(42, workers=True)
    
    if ENV == "local":
        config_path_obj = Path(config_path)
        config_path = str(config_path_obj.with_name(config_path_obj.stem + '_local' + config_path_obj.suffix))
    
    # Load configuration
    print(f"Loading configuration from {config_path}")
    if model_type == 'reconstructor':
        default_config_path = DEFAULT_CONFIGS_DIR / "config_reconstruction_default.yaml"
    elif model_type == 'classifier' or model_type == 'photclassifier':
        default_config_path = DEFAULT_CONFIGS_DIR / "config_classifier_default.yaml"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    default_config = load_config(default_config_path)
    additional_config = load_config(config_path)
    config = update_config(default_config, additional_config)
    
    # Get data types and names from config
    test_name = config['test_name']
    data_types = config['data_types']
    data_names = config['data_names']
    if test_name is None:
        raise ValueError("test_name must be provided in the config")
    if data_types is None:
        raise ValueError("data_types must be provided in the config")
    if data_names is None:
        raise ValueError("data_names must be provided in the config")
    
    world_size = torch.cuda.device_count()
    print(f"Training with {world_size} GPUs")
    print(f"Configuration loaded from: {config_path}")
    
    # Create dataloader using the shared function
    training_loader, validation_loader = create_dataloader(config, data_types, data_names)
    if model_type == 'classifier':
        weights, num_classes = get_weights_and_num_classes(training_loader, weight=config['unimodal']['architecture']['classifier']['shape']['weight_by_class'])
        config = update_config_with_num_classes(config, num_classes)
    else:
        weights = None
        num_classes = None
    
    # Initialize model
    model = initialize_model(model_type, data_types, config, weights)
    
    # Create output directory
    output_dir = Path(config["data"]["models_path"]) / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_worker(model, training_loader, validation_loader, config, output_dir, world_size)
    
    # mp.spawn(train_worker, args=(world_size, config, spectra_or_lightcurves), nprocs=world_size, join=True)
