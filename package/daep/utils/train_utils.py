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
import numpy as np
import math
import pytorch_lightning as L

class LossLogger(L.Callback):
    """
    Callback to record and plot epoch-level training/validation loss.

    Notes
    -----
    - Uses TensorBoard logger's directory as output path.
    - Plots are saved once per validation epoch on global rank 0.
    """

    def __init__(self):
        self.epochs = []
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
        epoch = trainer.current_epoch
        train_epoch = metrics.get("train_loss_epoch", metrics.get("train_loss"))
        val_epoch = metrics.get("val_loss_epoch", metrics.get("val_loss"))
        self.epochs.append(epoch)
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
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, train_loss, 'r-', linewidth=2, label=f'Training {loss_label}')
        plt.plot(self.epochs, val_loss, 'b-', linewidth=2, label=f'Validation {loss_label}')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(f'Training and Validation {loss_label}', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'loss.png', dpi=300, bbox_inches='tight')
        plt.close()

class AccuracyLogger(L.Callback):
    def __init__(self):
        self.epochs = []
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
        epoch = trainer.current_epoch
        train_epoch = metrics.get("train_acc_epoch", metrics.get("train_acc"))
        val_epoch = metrics.get("val_acc_epoch", metrics.get("val_acc"))
        self.epochs.append(epoch)
        self.train_acc.append(float(train_epoch))
        self.val_acc.append(float(val_epoch))

        self.plot_acc()
        self.plot_confusion_matrix(pl_module)

    def plot_acc(self):
        """Create and save the accuracy plot."""
        if not self.train_acc or not self.val_acc:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_acc, 'r-', linewidth=2, label=f'Training Accuracy')
        plt.plot(self.epochs, self.val_acc, 'b-', linewidth=2, label=f'Validation Accuracy')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(f'Training and Validation Accuracy', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'acc.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, pl_module):
        """Create and save the confusion matrix plot."""
        """
        Create and save the confusion matrix plot using the model's `conf_matrix` attribute.

        Notes
        -----
        - Assumes the *model* (pl_module) has a `conf_matrix` attribute (e.g., a torchmetrics confusion matrix object).
        - The confusion matrix is plotted and saved as 'confusion_matrix.png' in the output directory.
        """
        model = pl_module
        # Check if the model has a confusion matrix
        if not hasattr(model, "conf_matrix") or model.conf_matrix is None:
            print("No confusion matrix found in the model. Skipping confusion matrix plot.")
            return

        # Plot and save the confusion matrix
        metric = model.conf_matrix
        fig, ax = metric.plot()
        save_path = self.output_dir / "confusion_matrix.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


# def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, 
#                    device: torch.device = torch.device('cpu'), spectra_or_lightcurve: str = "spectra") -> Dict[str, Any]:
#     """
#     Load model checkpoint from file.
    
#     Parameters
#     ----------
#     checkpoint_path : str
#         Path to the checkpoint file to load.
#     model : nn.Module
#         The model to load the checkpoint into.
#     optimizer : torch.optim.Optimizer, optional
#         The optimizer to load state from checkpoint. If None, optimizer state is skipped.
#     device : torch.device
#         Device to load the checkpoint on.
#     spectra_or_lightcurve : str, optional
#         "spectra" or "lightcurve" or "both" to specify the type of data to train on. Defaults to "spectra".
#     Returns
#     -------
#     Dict[str, Any]
#         Dictionary containing checkpoint information including epoch, loss history, etc.
#     """
#     if not os.path.exists(checkpoint_path):
#         raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
#     print(f"Loading checkpoint from: {checkpoint_path}")
    
#     # Handle PyTorch 2.6+ security changes for loading custom classes
#     try:
#         # First try with weights_only=False for compatibility with older saved models
#         # This is safe since we're loading our own trained models
#         checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#     except Exception as e:
#         print(f"Failed to load checkpoint with weights_only=False: {e}")
#         print("Trying with safe globals...")
#         # Alternative approach: add our custom classes to safe globals
#         from torch.serialization import add_safe_globals
#         # Add our custom classes to the safe globals list
#         if spectra_or_lightcurve == "spectra":
#             from daep.daep import unimodaldaep
#             from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore, spectraTransceiverScore2stages
#             add_safe_globals([unimodaldaep, spectraTransceiverEncoder, spectraTransceiverScore, spectraTransceiverScore2stages])
#         elif spectra_or_lightcurve == "lightcurve":
#             from daep.daep import multimodaldaep
#             from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore, photometricTransceiverScore2stages
#             add_safe_globals([multimodaldaep, photometricTransceiverEncoder, photometricTransceiverScore, photometricTransceiverScore2stages])
#         elif spectra_or_lightcurve == "both":
#             from daep.daep import multimodaldaep
#             from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore, spectraTransceiverScore2stages
#             from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore, photometricTransceiverScore2stages
#             add_safe_globals([multimodaldaep, spectraTransceiverEncoder, spectraTransceiverScore, spectraTransceiverScore2stages, photometricTransceiverEncoder, photometricTransceiverScore, photometricTransceiverScore2stages])
#         checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
#     # Handle different checkpoint formats
#     if isinstance(checkpoint, dict):
#         # Checkpoint contains model state and metadata
#         if 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#         elif 'model' in checkpoint:
#             model.load_state_dict(checkpoint['model'].state_dict())
            
#         start_epoch = checkpoint.get('epoch', 0)
#         loss_history = checkpoint.get('loss_history', [])
#         optimizer_state = checkpoint.get('optimizer_state_dict')
        
#         if optimizer is not None and optimizer_state is not None:
#             optimizer.load_state_dict(optimizer_state)
#             print(f"Loaded optimizer state from epoch {start_epoch}")
#     else:
#         # Checkpoint is the model itself (current format)
#         model.load_state_dict(checkpoint.state_dict())
#         start_epoch = 0
#         loss_history = []
    
#     print(f"Successfully loaded checkpoint. Starting from epoch {start_epoch}")
#     return {
#         'model': model,
#         'start_epoch': start_epoch,
#         'loss_history': loss_history
#     }


# def setup_ddp(rank, world_size, config):
#     """
#     Initialize distributed training process group.
    
#     Parameters
#     ----------
#     rank : int
#         Rank of the current process.
#     world_size : int
#         Total number of processes.
#     config : Dict[str, Any]
#         Configuration dictionary containing distributed training settings.
#     """
#     dist.init_process_group(config["distributed"]["backend"], 
#                           init_method=config["distributed"]["init_method"],
#                           rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)

# def cleanup_ddp():
#     """Clean up distributed training process group."""
#     dist.destroy_process_group()

# def loss_plot(epoches, epoch_loss, start_epoch, config, save_path):
#     plt.plot(epoches, epoch_loss)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     # Add a horizontal dashed line at the loss value corresponding to start_epoch for documentation and visualization purposes
#     if start_epoch > 0 and start_epoch < len(epoch_loss):
#         plt.axvline(x=start_epoch, color='g', linestyle='--', label=f'Checkpoint Epoch ({start_epoch})')
#         plt.legend()
#     plt.title(f'Loss Plot for {config["data"]["test_name"]}')
#     plt.savefig(save_path)
#     plt.close()




