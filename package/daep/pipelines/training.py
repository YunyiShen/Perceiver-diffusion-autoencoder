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
from daep.daep import unimodaldaep
import math 
import os
import json
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from daep.utils.train_utils import setup_ddp, cleanup_ddp, load_checkpoint, loss_plot, load_and_update_config
from daep.utils.general_utils import detect_env, create_model_str

ENV = detect_env()

def initialize_model(device, model_mode, config):
    """
    Initialize the DAEP model based on configuration and model mode.
    
    Parameters
    ----------
    device : torch.device
        Device to place the model on.
    model_mode : str
        Mode of the model: "spectra", "lightcurves", "both", or "both_from_pretrained_encoders".
    config : Dict[str, Any]
        Configuration dictionary containing all model parameters.
        
    Returns
    -------
    torch.nn.Module
        Initialized DAEP model.
    """
    # Build model
    if model_mode == "spectra":
        from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore2stages
        encoder = spectraTransceiverEncoder(
            bottleneck_length=config["model"]["bottlenecklen"],
            bottleneck_dim=config["model"]["bottleneckdim"],
            model_dim=config["model"]["model_dim"],
            num_heads=config["model"]["encoder_heads"],
            ff_dim=config["model"]["model_dim"],
            num_layers=config["model"]["encoder_layers"],
            concat=config["model"]["concat"]
        )
        score = spectraTransceiverScore2stages(
            bottleneck_dim=config["model"]["bottleneckdim"],
            model_dim=config["model"]["model_dim"],
            num_heads=config["model"]["decoder_heads"],
            ff_dim=config["model"]["model_dim"],
            num_layers=config["model"]["decoder_layers"],
            concat=config["model"]["concat"],
            cross_attn_only=config["model"]["cross_attn_only"]
        )
        mydaep = unimodaldaep(encoder, score, regularize=config["model"]["regularize"]).to(device)
    elif model_mode == "lightcurves":
        from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore2stages
        encoder = photometricTransceiverEncoder(
            num_bands=1,
            bottleneck_length=config["model"]["bottlenecklen"],
            bottleneck_dim=config["model"]["bottleneckdim"],
            model_dim=config["model"]["model_dim"],
            num_heads=config["model"]["encoder_heads"],
            ff_dim=config["model"]["model_dim"],
            num_layers=config["model"]["encoder_layers"],
            concat=config["model"]["concat"],
            fourier=config["model"]["fourier_embed"]
        )
        score = photometricTransceiverScore2stages(
            num_bands=1,
            bottleneck_dim=config["model"]["bottleneckdim"],
            model_dim=config["model"]["model_dim"],
            num_heads=config["model"]["decoder_heads"],
            ff_dim=config["model"]["model_dim"],
            num_layers=config["model"]["decoder_layers"],
            concat=config["model"]["concat"],
            cross_attn_only=config["model"]["cross_attn_only"],
            fourier=config["model"]["fourier_embed"]
        )
        mydaep = unimodaldaep(encoder, score, regularize=config["model"]["regularize"]).to(device)
    elif model_mode == "both":
        from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore2stages
        from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore2stages
        from daep.Perceiver import PerceiverEncoder
        from daep.daep import multimodaldaep, modality_drop
        from functools import partial
        
        tokenizers = {
            "spectra": spectraTransceiverEncoder(
                bottleneck_length = config["model"]["bottlenecklen"],
                bottleneck_dim = config["model"]["spectra_tokens"],
                model_dim = config["model"]["model_dim"],
                ff_dim = config["model"]["model_dim"],
                num_layers = config["model"]["encoder_layers"],
                num_heads = config["model"]["encoder_heads"],
            ), 
            "photometry": photometricTransceiverEncoder(
                num_bands = 1, 
                bottleneck_length = config["model"]["bottlenecklen"],
                bottleneck_dim = config["model"]["photometry_tokens"],
                model_dim = config["model"]["model_dim"], 
                ff_dim = config["model"]["model_dim"],
                num_layers = config["model"]["encoder_layers"],
                num_heads = config["model"]["encoder_heads"],
                fourier=config["model"]["fourier_embed"]
            )
        }
        encoder = PerceiverEncoder(
                        bottleneck_length = config["model"]["bottlenecklen"],
                        bottleneck_dim = config["model"]["bottleneckdim"],
                        model_dim = config["model"]["model_dim"],
                        ff_dim = config["model"]["model_dim"],
                        num_layers = config["model"]["encoder_layers"],
                        num_heads = config["model"]["encoder_heads"],
                        selfattn = config["model"]["mixer_selfattn"]
        )
        scores = {
            "spectra":spectraTransceiverScore2stages(
                        bottleneck_dim = config["model"]["bottleneckdim"],
                        model_dim = config["model"]["model_dim"],
                        ff_dim = config["model"]["model_dim"],
                        num_heads = config["model"]["decoder_heads"],
                        num_layers = config["model"]["decoder_layers"],
                        concat = config["model"]["concat"]
                        ), 
            "photometry": photometricTransceiverScore2stages(
                bottleneck_dim = config["model"]["bottleneckdim"],
                    num_bands = 1,
                    model_dim = config["model"]["model_dim"],
                    ff_dim = config["model"]["model_dim"],
                    num_heads = config["model"]["decoder_heads"],
                    num_layers = config["model"]["decoder_layers"],
                    concat = config["model"]["concat"],
                    fourier=config["model"]["fourier_embed"]
            )
        }
        mydaep = multimodaldaep(
            tokenizers, encoder, scores,
            measurement_names={"spectra": "flux", "photometry": "flux"},
            modality_dropping_during_training=partial(modality_drop, p_drop=config["model"]["dropping_prob"])
        ).to(device)
    
    elif model_mode == "both_from_pretrained_encoders":
        pretrained_spectra_encoder_path = config["model"]["pretrained_spectra_encoder_path"]
        pretrained_photometry_encoder_path = config["model"]["pretrained_photometry_encoder_path"]
        
        from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore2stages
        from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore2stages
        from daep.Perceiver import PerceiverEncoder
        from daep.daep import multimodaldaep, modality_drop
        from functools import partial
        
        # Load pretrained tokenizers and freeze their weights
        print(f"Loading pretrained tokenizers from {pretrained_spectra_encoder_path} and {pretrained_photometry_encoder_path}")
        tokenizers = {
            "spectra": torch.load(pretrained_spectra_encoder_path, map_location=device, weights_only=False),
            "photometry": torch.load(pretrained_photometry_encoder_path, map_location=device, weights_only=False)
        }
        
        # Freeze tokenizer weights to prevent them from being updated during training
        for modality, tokenizer in tokenizers.items():
            print(f"Freezing {len(tokenizer.parameters())} parameters in {modality} tokenizer")
            for param in tokenizer.parameters():
                param.requires_grad = False
        
        encoder = PerceiverEncoder(
                        bottleneck_length = config["model"]["bottlenecklen"],
                        bottleneck_dim = config["model"]["bottleneckdim"],
                        model_dim = config["model"]["model_dim"],
                        ff_dim = config["model"]["model_dim"],
                        num_layers = config["model"]["encoder_layers"],
                        num_heads = config["model"]["encoder_heads"],
                        selfattn = config["model"]["mixer_selfattn"]
        )
        scores = {
            "spectra":spectraTransceiverScore2stages(
                        bottleneck_dim = config["model"]["bottleneckdim"],
                        model_dim = config["model"]["model_dim"],
                        ff_dim = config["model"]["model_dim"],
                        num_heads = config["model"]["decoder_heads"],
                        num_layers = config["model"]["decoder_layers"],
                        concat = config["model"]["concat"]
                        ), 
            "photometry": photometricTransceiverScore2stages(
                bottleneck_dim = config["model"]["bottleneckdim"],
                    num_bands = 1,
                    model_dim = config["model"]["model_dim"],
                    ff_dim = config["model"]["model_dim"],
                    num_heads = config["model"]["decoder_heads"],
                    num_layers = config["model"]["decoder_layers"],
                    concat = config["model"]["concat"]
            )
        }
        mydaep = multimodaldaep(
            tokenizers, encoder, scores,
            measurement_names={"spectra": "flux", "photometry": "flux"},
            modality_dropping_during_training=partial(modality_drop, p_drop=config["model"]["dropping_prob"])
        ).to(device)
        
    return mydaep

def train_worker(rank, world_size, config, spectra_or_lightcurves):
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
    spectra_or_lightcurves : str, optional
        "spectra" or "lightcurves" or "both" to specify the type of data to train on. Defaults to "spectra".
    """
    setup_ddp(rank, world_size, config)
    
    # Check if checkpoint loading is specified in config
    if "checkpoint" in config and config["checkpoint"].get("load_checkpoint", False):
        print(f"Will load checkpoint from: {config['checkpoint']['checkpoint_path']}")
    else:
        print("No checkpoint specified, starting training from scratch")
    
    # Extract paths from config
    test_name = config["data"]["test_name"]
    if spectra_or_lightcurves == "spectra":
        data_name = 'GALAHspectra'
        data_path = Path(config["data"]["data_path"]) / 'spectra'
        models_path = Path(config["data"]["models_path"]) / 'spectra_daep'
        models_path.mkdir(parents=True, exist_ok=True)
        from daep.datasets.GALAHspectra_dataset import GALAHDatasetProcessed
        training_data = GALAHDatasetProcessed(data_dir=data_path / test_name, train=True)
        collate_fn = padding_collate_fun(supply=['flux', 'wavelength', 'time'], mask_by="flux", multimodal=False)
    elif spectra_or_lightcurves == "lightcurves":
        data_name = 'TESSlightcurve'
        data_path = Path(config["data"]["data_path"]) / 'lightcurves'
        models_path = Path(config["data"]["models_path"]) / 'lightcurves_daep'
        models_path.mkdir(parents=True, exist_ok=True)
        from daep.datasets.TESSlightcurve_dataset import TESSDatasetProcessed
        training_data = TESSDatasetProcessed(data_dir=data_path / test_name, train=True)
        collate_fn = padding_collate_fun(supply=['flux', 'time'], mask_by="flux", multimodal=False)
    elif spectra_or_lightcurves == "both":
        data_name = 'TESSGALAHspeclc'
        data_path = Path(config["data"]["data_path"])
        models_path = Path(config["data"]["models_path"]) / 'speclc_daep'
        models_path.mkdir(parents=True, exist_ok=True)
        from daep.datasets.TESSGALAHspeclc_dataset import TESSGALAHDatasetProcessed
        from daep.datasets.TESSlightcurve_dataset import TESSDatasetProcessed
        from daep.datasets.GALAHspectra_dataset import GALAHDatasetProcessed
        lightcurve_test_name = config["data"]["lightcurve_test_name"]
        spectra_test_name = config["data"]["spectra_test_name"]
        dataset_lc = TESSDatasetProcessed(data_dir=data_path / 'lightcurves' / lightcurve_test_name, train=True)
        dataset_spectra = GALAHDatasetProcessed(data_dir=data_path / 'spectra' / spectra_test_name, train=True)
        training_data = TESSGALAHDatasetProcessed(dataset_lc, dataset_spectra)
        collate_fn = padding_collate_fun(supply=['flux', 'wavelength', 'time'], mask_by="flux", multimodal=True)
        
    # Dataset and loader with distributed sampler
    sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank, shuffle=True)
    training_loader = DataLoader(training_data, batch_size=config["training"]["batch"], sampler=sampler,
                                collate_fn=collate_fn,
                                num_workers=config["data_processing"]["num_workers"], 
                                pin_memory=config["data_processing"]["pin_memory"])
    
    device = torch.device(f"cuda:{rank}")
    
    mydaep = initialize_model(device, model_mode=spectra_or_lightcurves, config=config)
    
    # Print # of parameters
    if rank == 0:
        print(f"Model has {sum(p.numel() for p in mydaep.parameters())} total parameters")
        print(f"Model has {sum(p.numel() for p in mydaep.parameters() if p.requires_grad)} trainable parameters")
    
    mydaep = DDP(mydaep, device_ids=[rank], find_unused_parameters=True)
    
    mydaep.train()
    optimizer = AdamW(mydaep.parameters(), lr=config["training"]["lr"])
    
    # Initialize training state
    start_epoch = 0
    epoch_loss = []
    epoches = []
    target_save = None
    loss_plot_path = None
    
    model_str = create_model_str(config, data_name)
    
    # Load checkpoint if specified
    if config.get("checkpoint", {}).get("load_checkpoint", False) and rank == 0:
        try:
            checkpoint_path = config["checkpoint"]["checkpoint_path"]
            checkpoint_info = load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=mydaep.module,  # Unwrap DDP model for loading
                optimizer=optimizer,
                device=device,
                spectra_or_lightcurves=spectra_or_lightcurves
            )
            # mydaep = checkpoint_info['model']
            start_epoch = checkpoint_info['start_epoch']
            epoch_loss = checkpoint_info['loss_history']
            epoches = list(range(len(epoch_loss)))
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
    
    # Use tqdm for progress tracking (only on main process to avoid duplicate bars)
    progress_bar = tqdm(range(start_epoch, config["training"]["epoch"]), desc="Training", unit="epoch", disable=(rank != 0))
    
    for ep in progress_bar:
        sampler.set_epoch(ep)
        losses = []
        
        for x in training_loader:
            x = to_device(x, device)
            optimizer.zero_grad()
            loss = mydaep(x)
            
            # Add NaN check
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss detected: {loss.item()}, skipping batch")
                continue
                
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(mydaep.parameters(), max_norm=1.0)
            
            optimizer.step()
            losses.append(loss.item())
        
        if rank == 0:  # Only save and log on main process
            this_epoch = np.array(losses).mean().item()
            epoch_loss.append(math.log(this_epoch))
            epoches.append(ep)
            
            # Update epoch progress bar
            progress_bar.set_postfix({
                'avg_loss': f'{this_epoch:.4f}',
                'log_loss': f'{math.log(this_epoch):.4f}'
            })
            
            if (ep+1) % config["training"]["save_every"] == 0:
                # Delete old checkpoints, except for every 10th checkpoint
                if target_save is not None and ((ep+1) % (10*config["training"]["save_every"]) != 1):
                    os.remove(target_save)
                if loss_plot_path is not None:
                    os.remove(loss_plot_path)
                # Create directories if they don't exist
                model_dir = models_path / test_name / model_str
                ckpt_dir = model_dir / "ckpt"
                logs_dir = model_dir / "loss_plots"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                logs_dir.mkdir(parents=True, exist_ok=True)
                
                # Unwrap DDP model before saving
                model_to_save = mydaep.module
                
                config_w_date = config.copy()
                config_w_date["datetime"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                config_w_date["trained_to_epoch"] = ep+1
                
                # Save checkpoint with metadata
                checkpoint_data = {
                    'model': model_to_save,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': ep + 1,
                    'loss_history': epoch_loss,
                    'config': config_w_date
                }
                
                # if config["model"]["model_mode"] != "both_from_pretrained_encoders":
                #     model_str = f"{config['model']['bottlenecklen']}-{config['model']['bottleneckdim']}-{config['model']['encoder_layers']}-{config['model']['decoder_layers']}-{config['model']['encoder_heads']}-{config['model']['decoder_heads']}-{config['model']['model_dim']}_concat{config['model']['concat']}_crossattnonly{config['model']['cross_attn_only']}_lr{config['training']['lr']}_epoch{ep+1}_batch{config['training']['batch']}_world{world_size}_reg{config['model']['regularize']}_aug{config['training']['aug']}_date{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
                # else:
                #     model_str = f"{config['model']['bottlenecklen']}-{config['model']['bottleneckdim']}-{config['model']['spectra_tokens']}-{config['model']['photometry_tokens']}-{config['model']['encoder_layers']}-{config['model']['decoder_layers']}-{config['model']['encoder_heads']}-{config['model']['decoder_heads']}-{config['model']['model_dim']}_concat{config['model']['concat']}_crossattnonly{config['model']['cross_attn_only']}_lr{config['training']['lr']}_modaldropP{config['model']['dropping_prob']}_epoch{ep+1}_batch{config['training']['batch']}_world{world_size}_reg{config['model']['regularize']}_aug{config['training']['aug']}_date{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
                
                # # Save checkpoint
                # target_save = ckpt_dir / f"{data_name}_daep_{model_str}.pth"
                
                
                # Save checkpoint
                target_save = ckpt_dir / f"epoch_{ep+1}_date_{str(datetime.now().strftime('%Y-%m-%d_%H-%M'))}.pth"
                
                torch.save(checkpoint_data, target_save)
                
                # Save loss plot
                loss_plot_path = logs_dir / f"epoch_{ep+1}_date_{str(datetime.now().strftime('%Y-%m-%d_%H-%M'))}.png"
                loss_plot(epoches, epoch_loss, start_epoch, config, loss_plot_path)
                
            print(f"[GPU {rank}] Epoch {ep+1} log-loss: {math.log(this_epoch):.4f}")
    
    # Close progress bar
    if rank == 0:
        progress_bar.close()
    
    cleanup_ddp()
    
    print(f"{config['training']['epoch']} Epochs of training complete for {data_name} on {spectra_or_lightcurves} data")


def train(config_path: str = "config_train_galah.json", spectra_or_lightcurves: str = "spectra", **kwargs):
    """
    Train a unimodal DAEP model on either spectra or lightcurves using configuration from file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration JSON file. Defaults to "config_train_galah.json".
    spectra_or_lightcurves : str, optional
        "spectra" or "lightcurves" or "both" to specify the type of data to train on. Defaults to "spectra".
    **kwargs : dict
        Optional keyword arguments to override config values.
        Useful for quick parameter adjustments without modifying the config file.
        
    Examples
    --------
    >>> train()  # Use default config_train_galah.json
    >>> train("my_config.json")  # Use custom config file
    >>> train(epoch=100, lr=1e-4)  # Override specific parameters
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
    print(f"Training with {world_size} GPUs")
    print(f"Configuration loaded from: {config_path}")
    
    mp.spawn(train_worker, args=(world_size, config, spectra_or_lightcurves), nprocs=world_size, join=True)
