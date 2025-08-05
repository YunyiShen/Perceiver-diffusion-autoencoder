import torch
from torch import nn
# dataloader
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
# optimizer
from torch.optim import AdamW
# Multi-GPU support
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from matplotlib import pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from daep.data_util import to_device, padding_collate_fun
from daep.daep import unimodaldaepclassifier, multimodaldaepclassifier
from daep.Classifier import LCC
import math 
import os
import json
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from daep.utils.train_utils import setup_ddp, cleanup_ddp, load_checkpoint, loss_plot, load_and_update_config
from daep.utils.general_utils import detect_env, create_model_str_classifier
from daep.pipelines.training import create_dataloader

ENV = detect_env()

def load_pretrained_model(model_path: str, device: torch.device):
    """
    Load a pretrained model from a checkpoint file and optionally extract specific components.
    
    Parameters
    ----------
    model_path : str
        Path to the pretrained model checkpoint
    device : torch.device
        Device to load the model on
    
    Returns
    -------
    torch.nn.Module
        Loaded pretrained model or extracted component
    """
    print(f"Loading pretrained model from: {model_path}")
    
    if model_path.endswith('.pth'):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # This would need to be handled based on the specific model architecture
                raise NotImplementedError("state_dict loading not implemented yet")
            else:
                model = checkpoint
        else:
            model = checkpoint
            
    else:
        raise ValueError(f"Unsupported model file format: {model_path}")
    
    return model

def freeze_model_parameters(model: nn.Module):
    """
    Freeze model parameters for frozen encoder/tokenizer training.
    
    Parameters
    ----------
    model : nn.Module
        Model whose parameters to freeze
    """
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Frozen {sum(p.numel() for p in model.parameters())} parameters")

def initialize_classifier_model(device, spectra_or_lightcurves, config):
    """
    Initialize the classifier model with frozen encoder/tokenizers.
    
    Parameters
    ----------
    device : torch.device
        Device to place the model on.
    spectra_or_lightcurves : str
        "spectra", "lightcurves", or "both" to specify the type of data to train on.
    config : Dict[str, Any]
        Configuration dictionary containing all model parameters.
        
    Returns
    -------
    torch.nn.Module
        Initialized classifier model with frozen encoder/tokenizers.
    """
    # Build classifier
    classifier = LCC(
        emb_d=config["model"]["bottleneckdim"],
        dropout_p=config["model"]["classifier_dropout"],
        num_classes=config["model"]["num_classes"]
    )
    
    model = load_pretrained_model(
            config["model"]["pretrained_encoder_path"], 
            device
        )
    
    # Initialize model based on mode
    if spectra_or_lightcurves in ["spectra", "lightcurves"]:
        # Load and freeze encoder
        if hasattr(model, 'encoder'):
            encoder = model.encoder
        else:
            encoder = model
        freeze_model_parameters(model)
        
        # Create unimodal classifier
        model = unimodaldaepclassifier(
            encoder=encoder,
            classifier=classifier,
            MMD=None,  # Can be added if needed
            regularize=config["model"]["regularize"]
        ).to(device)
        
    elif spectra_or_lightcurves == "both":
        from daep.daep import modality_drop
        from functools import partial
        
        # Assume model is a multimodaldaep model
        spectra_tokenizer = model.tokenizers["spectra"]
        photometry_tokenizer = model.tokenizers["photometry"]
        encoder = model.encoder
        
        freeze_model_parameters(spectra_tokenizer)
        freeze_model_parameters(photometry_tokenizer)
        freeze_model_parameters(encoder)

        # # Load pretrained tokenizers
        # tokenizers = {}
        
        # # Load spectra tokenizer
        # spectra_model = load_pretrained_model(
        #     config["model"]["pretrained_spectra_tokenizer_path"],
        #     device
        # )
        # if hasattr(spectra_model, 'tokenizers'):
        #     tokenizers["spectra"] = spectra_model.tokenizers["spectra"]
        # else:
        #     tokenizers["spectra"] = spectra_model
            
        # # Load photometry tokenizer
        # photometry_model = load_pretrained_model(
        #     config["model"]["pretrained_photometry_tokenizer_path"], 
        #     device
        # )
        # if hasattr(photometry_model, 'tokenizers'):
        #     tokenizers["photometry"] = photometry_model.tokenizers["photometry"]
        # else:
        #     tokenizers["photometry"] = photometry_model

        # # Freeze tokenizer parameters
        # for modality, tokenizer in tokenizers.items():
        #     print(f"Freezing {modality} tokenizer")
        #     freeze_model_parameters(tokenizer)

        # # Load and freeze encoder
        # model = load_pretrained_model(
        #     config["model"]["pretrained_encoder_path"], 
        #     device
        # )
        # if hasattr(model, 'encoder'):
        #     encoder = model.encoder
        # else:
        #     encoder = model
        # freeze_model_parameters(encoder)

        model = multimodaldaepclassifier(
            tokenizers={"spectra": spectra_tokenizer, "photometry": photometry_tokenizer},
            encoder=encoder,
            classifier=classifier,
            measurement_names={"spectra": "flux", "photometry": "flux"},
            modality_dropping_during_training=partial(modality_drop, p_drop=config["model"]["dropping_prob"])
        ).to(device)
    
    return model

def extract_targets_from_batch(batch, device):
    """
    Extract targets from batch and prepare input for model.
    
    Parameters
    ----------
    batch : Any
        Input batch
    device : torch.device
        Device to move data to
        
    Returns
    -------
    tuple
        (model_input, targets) or (None, None) if no targets found
    """
    x = to_device(batch, device)
    
    if isinstance(x, dict) and 'starclass' in x:
        targets = x['starclass']
        # Remove targets from x to avoid passing to model
        model_input = {k: v for k, v in x.items() if k != 'starclass'}
        return model_input, targets
    else:
        return None, None

def save_checkpoint_and_plot(model, optimizer, epoch, epoch_loss, config, ckpt_dir, logs_dir, target_save, loss_plot_path):
    """
    Save checkpoint and loss plot.
    
    Returns
    -------
    tuple
        (new_checkpoint_path, new_loss_plot_path)
    """
    # Delete old checkpoints, except for every 10th checkpoint
    if target_save is not None and ((epoch+1) % (10*config["training"]["save_every"]) != 1):
        os.remove(target_save)
    if loss_plot_path is not None:
        os.remove(loss_plot_path)
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Unwrap DDP model before saving
    model_to_save = model.module
    
    config_w_date = config.copy()
    config_w_date["datetime"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config_w_date["trained_to_epoch"] = epoch+1
    
    # Save checkpoint with metadata
    checkpoint_data = {
        'model': model_to_save,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch + 1,
        'loss_history': epoch_loss,
        'config': config_w_date
    }
    
    # Save checkpoint
    new_checkpoint_path = ckpt_dir / f"epoch_{epoch+1}_date_{str(datetime.now().strftime('%Y-%m-%d_%H-%M'))}.pth"
    torch.save(checkpoint_data, new_checkpoint_path)
    
    # Save loss plot
    new_loss_plot_path = logs_dir / f"epoch_{epoch+1}_date_{str(datetime.now().strftime('%Y-%m-%d_%H-%M'))}.png"
    loss_plot(list(range(len(epoch_loss))), epoch_loss, 0, config, new_loss_plot_path)
    
    return new_checkpoint_path, new_loss_plot_path

def train_classifier_worker(rank, world_size, config, spectra_or_lightcurves):
    """
    Worker function for distributed classifier training with frozen encoder/tokenizers.
    
    Parameters
    ----------
    rank : int
        Rank of the current process.
    world_size : int
        Total number of processes.
    config : Dict[str, Any]
        Configuration dictionary containing all training parameters.
    spectra_or_lightcurves : str
        "spectra", "lightcurves", or "both" to specify the type of data to train on.
    """
    setup_ddp(rank, world_size, config)
    
    # Check if checkpoint loading is specified in config
    if "checkpoint" in config and config["checkpoint"].get("load_checkpoint", False):
        print(f"Will load checkpoint from: {config['checkpoint']['checkpoint_path']}")
    else:
        print("No checkpoint specified, starting training from scratch")
    
    # Create dataloader using the shared function
    training_loader, data_name, sampler = create_dataloader(config, spectra_or_lightcurves, rank, world_size)
    data_name += "_classifier"
    
    # Get models path based on model mode
    base_path = Path(config["data"]["models_path"])
    if spectra_or_lightcurves == "spectra":
        models_path = base_path / 'spectra_classifier'
    elif spectra_or_lightcurves == "lightcurves":
        models_path = base_path / 'lightcurves_classifier'
    elif spectra_or_lightcurves == "both":
        models_path = base_path / 'speclc_classifier'
    else:
        raise ValueError(f"Unknown spectra_or_lightcurves: {spectra_or_lightcurves}")
    
    models_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(f"cuda:{rank}")
    
    # Initialize the model
    model = initialize_classifier_model(device, spectra_or_lightcurves, config)
    
    # Print # of parameters
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {total_params} total parameters")
        print(f"Model has {trainable_params} trainable parameters")
        print(f"Frozen parameters: {total_params - trainable_params}")
    
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    model.train()
    optimizer = AdamW(model.parameters(), lr=config["training"]["lr"])
    
    # Initialize training state
    start_epoch = 0
    epoch_loss = []
    target_save = None
    loss_plot_path = None
    
    model_str = create_model_str_classifier(config, data_name)
    model_parent_str = Path(config["model"]["pretrained_encoder_path"]).parent.parent.name
    
    # Load checkpoint if specified
    if config.get("checkpoint", {}).get("load_checkpoint", False) and rank == 0:
        try:
            checkpoint_path = config["checkpoint"]["checkpoint_path"]
            checkpoint_info = load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model.module,  # Unwrap DDP model for loading
                optimizer=optimizer,
                device=device,
                spectra_or_lightcurves=spectra_or_lightcurves
            )
            start_epoch = checkpoint_info['start_epoch']
            epoch_loss = checkpoint_info['loss_history']
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(f"Resuming training from epoch {start_epoch}")
            model_dir = checkpoint_path.parent
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("Starting training from scratch")
    
    # Synchronize checkpoint loading across processes
    if world_size > 1:
        start_epoch_tensor = torch.tensor(start_epoch, device=device)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = start_epoch_tensor.item()
    
    # Create directories if they don't exist
    test_name = config['data']['test_name']
    model_dir = models_path / test_name / model_str / model_parent_str
    print(f"Saving model to directory: {model_dir}")
    ckpt_dir = model_dir / "ckpt"
    logs_dir = model_dir / "loss_plots"
    
    # Use tqdm for progress tracking (only on main process to avoid duplicate bars)
    progress_bar = tqdm(range(start_epoch, config["training"]["epoch"]), desc="Training", unit="epoch", disable=(rank != 0))
    
    for ep in progress_bar:
        sampler.set_epoch(ep)
        losses = []

        for batch_idx, batch in enumerate(training_loader):
            model_input, targets = extract_targets_from_batch(batch, device)
            
            if model_input is None:
                print(f"Warning: No targets found in batch {batch_idx}")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass with targets
            # Without an MMD provided in model initialization, the total loss is just the classification loss
            classification_loss, mmd_loss, total_loss = model(model_input, targets=targets)
            
            # Add NaN check
            if torch.isnan(total_loss):
                print(f"NaN loss detected: {total_loss.item()}, skipping batch")
                continue
            elif torch.isinf(total_loss):
                print(f"Inf loss detected: {total_loss.item()}, skipping batch")
                continue
                
            total_loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            losses.append(total_loss.item())
        
        if rank == 0:  # Only save and log on main process
            this_epoch = np.array(losses).mean().item()
            if this_epoch > 0:
                logloss = np.log(this_epoch)
            else:
                logloss = float('nan')
            epoch_loss.append(logloss)


            # Update epoch progress bar with both losses
            progress_bar.set_postfix({
                'avg_loss': f'{this_epoch:.4f}',
                'log_loss': f'{logloss:.4f}',
            })
            
            if (ep+1) % config["training"]["save_every"] == 0:
                target_save, loss_plot_path = save_checkpoint_and_plot(
                    model, optimizer, ep, epoch_loss, config, ckpt_dir, logs_dir, target_save, loss_plot_path
                )
                
            print(f"[GPU {rank}] Epoch {ep+1} loss: {this_epoch:.4f}, logloss: {logloss:.4f}")

    # Comment: Added logloss (classification loss) tracking and reporting for each epoch.
    
    # Close progress bar
    if rank == 0:
        progress_bar.close()
    
    cleanup_ddp()
    
    print(f"{config['training']['epoch']} Epochs of training complete for {data_name} classifier on {spectra_or_lightcurves} data")

def train_classifier(config_path: str = "config_train_classifier.json", spectra_or_lightcuves: str = "spectra", **kwargs):
    """
    Train a classifier model with frozen encoder/tokenizers using configuration from file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration JSON file. Defaults to "config_train_classifier.json".
    model_mode : str, optional
        "spectra", "lightcurves", or "both" to specify the type of data to train on. Defaults to "spectra".
    **kwargs : dict
        Optional keyword arguments to override config values.
        Useful for quick parameter adjustments without modifying the config file.
        
    Examples
    --------
    >>> train_classifier()  # Use default config_train_classifier.json
    >>> train_classifier("my_config.json")  # Use custom config file
    >>> train_classifier(epoch=100, lr=1e-4)  # Override specific parameters
    """
    # Set environment variables for debugging if needed
    import os
    if "TORCH_DISTRIBUTED_DEBUG" not in os.environ:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    
    if ENV == "local":
        config_path_obj = Path(config_path)
        config_path = str(config_path_obj.with_name(config_path_obj.stem + '_local' + config_path_obj.suffix))
    
    # Load configuration
    config = load_and_update_config(config_path, **kwargs)
    
    world_size = torch.cuda.device_count()
    print(f"Training classifier with {world_size} GPUs")
    print(f"Configuration loaded from: {config_path}")
    
    mp.spawn(train_classifier_worker, args=(world_size, config, spectra_or_lightcuves), nprocs=world_size, join=True) 