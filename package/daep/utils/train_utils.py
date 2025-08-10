import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, 
                   device: torch.device = torch.device('cpu'), spectra_or_lightcurve: str = "spectra") -> Dict[str, Any]:
    """
    Load model checkpoint from file.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file to load.
    model : nn.Module
        The model to load the checkpoint into.
    optimizer : torch.optim.Optimizer, optional
        The optimizer to load state from checkpoint. If None, optimizer state is skipped.
    device : torch.device
        Device to load the checkpoint on.
    spectra_or_lightcurve : str, optional
        "spectra" or "lightcurve" or "both" to specify the type of data to train on. Defaults to "spectra".
    Returns
    -------
    Dict[str, Any]
        Dictionary containing checkpoint information including epoch, loss history, etc.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Handle PyTorch 2.6+ security changes for loading custom classes
    try:
        # First try with weights_only=False for compatibility with older saved models
        # This is safe since we're loading our own trained models
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint with weights_only=False: {e}")
        print("Trying with safe globals...")
        # Alternative approach: add our custom classes to safe globals
        from torch.serialization import add_safe_globals
        # Add our custom classes to the safe globals list
        if spectra_or_lightcurve == "spectra":
            from daep.daep import unimodaldaep
            from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore, spectraTransceiverScore2stages
            add_safe_globals([unimodaldaep, spectraTransceiverEncoder, spectraTransceiverScore, spectraTransceiverScore2stages])
        elif spectra_or_lightcurve == "lightcurve":
            from daep.daep import multimodaldaep
            from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore, photometricTransceiverScore2stages
            add_safe_globals([multimodaldaep, photometricTransceiverEncoder, photometricTransceiverScore, photometricTransceiverScore2stages])
        elif spectra_or_lightcurve == "both":
            from daep.daep import multimodaldaep
            from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore, spectraTransceiverScore2stages
            from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore, photometricTransceiverScore2stages
            add_safe_globals([multimodaldaep, spectraTransceiverEncoder, spectraTransceiverScore, spectraTransceiverScore2stages, photometricTransceiverEncoder, photometricTransceiverScore, photometricTransceiverScore2stages])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Checkpoint contains model state and metadata
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'].state_dict())
            
        start_epoch = checkpoint.get('epoch', 0)
        loss_history = checkpoint.get('loss_history', [])
        optimizer_state = checkpoint.get('optimizer_state_dict')
        
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            print(f"Loaded optimizer state from epoch {start_epoch}")
    else:
        # Checkpoint is the model itself (current format)
        model.load_state_dict(checkpoint.state_dict())
        start_epoch = 0
        loss_history = []
    
    print(f"Successfully loaded checkpoint. Starting from epoch {start_epoch}")
    return {
        'model': model,
        'start_epoch': start_epoch,
        'loss_history': loss_history
    }


def setup_ddp(rank, world_size, config):
    """
    Initialize distributed training process group.
    
    Parameters
    ----------
    rank : int
        Rank of the current process.
    world_size : int
        Total number of processes.
    config : Dict[str, Any]
        Configuration dictionary containing distributed training settings.
    """
    dist.init_process_group(config["distributed"]["backend"], 
                          init_method=config["distributed"]["init_method"],
                          rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up distributed training process group."""
    dist.destroy_process_group()

def loss_plot(epoches, epoch_loss, start_epoch, config, save_path):
    plt.plot(epoches, epoch_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # Add a horizontal dashed line at the loss value corresponding to start_epoch for documentation and visualization purposes
    if start_epoch > 0 and start_epoch < len(epoch_loss):
        plt.axvline(x=start_epoch, color='g', linestyle='--', label=f'Checkpoint Epoch ({start_epoch})')
        plt.legend()
    plt.title(f'Loss Plot for {config["data"]["test_name"]}')
    plt.savefig(save_path)
    plt.close()




