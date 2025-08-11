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
from daep.datasets.dataloaders import create_dataloader

ENV = detect_env()
DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

def initialize_model(model_type, data_types, config):
    if model_type == 'classifier':
        if len(data_types) == 1:
            model = daepClassifierUnimodal(config=config)
        else:
            model = daepClassifierMultimodal(config=config)
    elif model_type == 'reconstructor':
        if len(data_types) == 1:
            model = daepReconstructorUnimodal(config=config)
        else:
            model = daepReconstructorMultimodal(config=config)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    return model

class LossLogger(L.Callback):
    """
    Callback to record and plot epoch-level training/validation loss.

    Notes
    -----
    - Uses TensorBoard logger's directory as output path.
    - Plots are saved once per validation epoch on global rank 0.
    """

    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.output_dir = None

    def on_fit_start(self, trainer, pl_module):
        """Initialize output directory when training starts."""
        # Resolve TensorBoard log directory and ensure it exists
        self.output_dir = Path(trainer.logger.log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        After each validation epoch, read aggregated epoch metrics and update the plot.
        """
        # Skip Lightning's sanity validation
        if getattr(trainer, "sanity_checking", False):
            return
        # Only save from global rank 0 in DDP
        if getattr(trainer, "global_rank", 0) != 0:
            return

        metrics = trainer.callback_metrics
        train_epoch = metrics.get("train_loss")
        val_epoch = metrics.get("val_loss")
        self.train_loss.append(float(train_epoch))
        self.val_loss.append(float(val_epoch))

        self.plot_loss()

    def plot_loss(self):
        """Create and save the loss plot."""
        if not self.train_loss or not self.val_loss:
            return
        
        if np.all(np.array(self.train_loss) > 0) or np.all(np.array(self.val_loss) > 0):
            train_loss = [math.log(loss) for loss in self.train_loss]
            val_loss = [math.log(loss) for loss in self.val_loss]
            loss_label = 'Log-Loss'
        else:
            train_loss = self.train_loss
            val_loss = self.val_loss
            loss_label = 'Loss'
        
        epochs = range(1, len(self.train_loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'r-', linewidth=2, label=f'Training {loss_label}')
        plt.plot(epochs, val_loss, 'b-', linewidth=2, label=f'Validation {loss_label}')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(f'Training and Validation {loss_label}', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'loss.png', dpi=300, bbox_inches='tight')
        plt.close()

class AccuracyLogger(L.Callback):
    def __init__(self):
        self.train_acc = []
        self.val_acc = []
        self.output_dir = None

    def on_fit_start(self, trainer, pl_module):
        """Initialize output directory when training starts."""
        # Get the log directory from the trainer's logger and convert to Path
        self.output_dir = Path(trainer.logger.log_dir)

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        After each validation epoch, read aggregated epoch metrics and update the plot.
        """
        # Skip Lightning's sanity validation
        if getattr(trainer, "sanity_checking", False):
            return
        # Only save from global rank 0 in DDP
        if getattr(trainer, "global_rank", 0) != 0:
            return

        metrics = trainer.callback_metrics
        train_epoch = metrics.get("train_acc")
        val_epoch = metrics.get("val_acc")
        self.train_acc.append(float(train_epoch))
        self.val_acc.append(float(val_epoch))

        self.plot_acc()

    def plot_acc(self):
        """Create and save the accuracy plot."""
        if not self.train_acc or not self.val_acc:
            return
        
        epochs = range(1, len(self.train_acc) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_acc, 'r-', linewidth=2, label=f'Training Accuracy')
        plt.plot(epochs, self.val_acc, 'b-', linewidth=2, label=f'Validation Accuracy')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(f'Training and Validation Accuracy', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'acc.png', dpi=300, bbox_inches='tight')
        plt.close()

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
    if isinstance(model, daepClassifierUnimodal) or isinstance(model, daepClassifierMultimodal):
        callbacks = [checkpoint_callback, accuracy_logger]
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
    elif model_type == 'classifier':
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
    
    # Initialize model
    model = initialize_model(model_type, data_types, config)
    
    # Create dataloader using the shared function
    training_loader, validation_loader = create_dataloader(config, data_types, data_names)
    
    # Create output directory
    output_dir = Path(config["data"]["models_path"]) / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_worker(model, training_loader, validation_loader, config, output_dir, world_size)
    
    # mp.spawn(train_worker, args=(world_size, config, spectra_or_lightcurves), nprocs=world_size, join=True)
