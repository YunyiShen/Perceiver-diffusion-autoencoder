from typing import Tuple
from typing import Dict, Any
from datetime import datetime
import json
from copy import deepcopy
from pathlib import Path

def detect_env() -> str:
    """
    Detects whether the environment is 'local' or 'remote'.

    Returns
    -------
    env : str
        'local' if running on a local machine, 'remote' if running on a remote server.

    Notes
    -----
    This function checks for common environment variables and file paths that are typically present
    on remote servers (e.g., SLURM job environment, specific user directories). If none are found,
    it defaults to 'local'.

    Examples
    --------
    >>> env = detect_env()
    >>> print(env)
    'local'
    """
    import os

    # Check for SLURM environment variable (common on HPC clusters)
    if "SLURM_JOB_ID" in os.environ:
        return "remote"
    # Check for known remote base path
    remote_base = "/nobackup/users/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder"
    if Path(remote_base).exists():
        return "remote"
    # Check for other possible remote indicators
    if "JUPYTERHUB_USER" in os.environ or "PBS_JOBID" in os.environ:
        return "remote"
    # Default to local
    return "local"

def set_paths(env: str, spectra_or_lightcurve: str) -> Tuple[str, str, str, str]:
    if spectra_or_lightcurve == 'lightcurve':
        spectra_or_lightcurve = 'lightcurves'
    if env == 'remote':
        base_path = "/nobackup/users/allisone/UROP_2025_Summer/Perceiver-diffusion-autoencoder"
        model_path = base_path + f"/models/{spectra_or_lightcurve}"
        data_path = base_path + f"/data/{spectra_or_lightcurve}"
        raw_data_path = f"/nobackup/users/allisone/UROP_2025_Summer/vastclammm/data/{spectra_or_lightcurve}_raw" #base_path + f"/data/{spectra_or_lightcurve}_raw"
    elif env == 'local':
        base_path = "/home/altair/Documents/UROP/2025_Summer/Perceiver-diffusion-autoencoder"
        model_path = base_path + f"/models/{spectra_or_lightcurve}"
        data_path = base_path + f"/data/{spectra_or_lightcurve}"
        raw_data_path = base_path + f"/data/{spectra_or_lightcurve}_raw"
    
    return base_path, model_path, data_path, raw_data_path

def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the configuration YAML file.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary containing all training parameters.

    Notes
    -----
    This function loads a YAML configuration file and sets the appropriate data/model paths
    based on the 'env' key in the config.

    Examples
    --------
    >>> config = load_config("config.yaml")
    >>> print(config["training"]["epochs"])
    """
    import yaml

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set data/model paths based on environment
        if "data" in config:
            if config["env"] == "local":
                config["data"]["data_path"] = config["data"]["data_path_local"]
                config["data"]["models_path"] = config["data"]["models_path_local"]
            elif config["env"] == "remote":
                config["data"]["data_path"] = config["data"]["data_path_remote"]
                config["data"]["models_path"] = config["data"]["models_path_remote"]
            else:
                raise ValueError(f"Invalid environment: {config['env']}; environment must be either 'local' or 'remote'")

        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Invalid YAML in configuration file: {e}")

    # Added: switched from JSON to YAML for config loading, using yaml.safe_load for security.
    # Added: updated docstring to reflect YAML usage.

def update_config(base_config: Dict[str, Any], additional_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge ``additional_config`` into ``base_config`` (recursive for nested dicts).
    Values that are ``None`` in ``additional_config`` are ignored and do not overwrite.

    Parameters
    ----------
    base_config : Dict[str, Any]
        Base configuration.
    additional_config : Dict[str, Any]
        Updates to apply.

    Returns
    -------
    Dict[str, Any]
        New merged configuration. Inputs are not mutated.
    """

    def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = deepcopy(a)
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)  # type: ignore[index]
            elif isinstance(v, dict):
                # Base is not a dict (or missing); accept the new nested dict
                out[k] = deepcopy(v)
            elif v is not None:
                # Only overwrite when update value is not None
                out[k] = deepcopy(v)
            # else: v is None → skip, keep base value
        return out

    new_config = merge(base_config, additional_config)
    
    # Set data/model paths based on environment
    if "data" in new_config:
            if new_config["env"] == "local":
                new_config["data"]["data_path"] = new_config["data"]["data_path_local"]
                new_config["data"]["models_path"] = new_config["data"]["models_path_local"]
            elif new_config["env"] == "remote":
                new_config["data"]["data_path"] = new_config["data"]["data_path_remote"]
                new_config["data"]["models_path"] = new_config["data"]["models_path_remote"]
            else:
                raise ValueError(f"Invalid environment: {new_config['env']}; environment must be either 'local' or 'remote'")
    return new_config


def convert_to_native_byte_order(df):
    for col in df.columns:
        col_data = df[col]
        if hasattr(col_data.values, 'dtype') and hasattr(col_data.values.dtype, 'byteorder'):
            if col_data.values.dtype.byteorder == '>':
                try:    
                    df[col] = col_data.values.byteswap().newbyteorder()
                except AttributeError:
                    # Fix for NumPy 2.0: use .view(dtype.newbyteorder()) instead of .newbyteorder()
                    df[col] = col_data.values.byteswap().view(col_data.values.dtype.newbyteorder('='))
    return df

def create_model_str(config: Dict[str, Any], data_name: str) -> str:
    # Use a base format string and insert extra fields only if needed for compactness
    base_fmt = (
        f"{data_name}_"
        f"{config['model']['bottlenecklen']}-{config['model']['bottleneckdim']}-"
    )
    if 'spectra_tokens' in config['model'] and 'photometry_tokens' in config['model']:
        base_fmt += (
            f"{config['model']['spectra_tokens']}-{config['model']['photometry_tokens']}-"
        )
    base_fmt += (
        f"{config['model']['encoder_layers']}-{config['model']['decoder_layers']}-"
        f"{config['model']['encoder_heads']}-{config['model']['decoder_heads']}-"
        f"{config['model']['model_dim']}_"
    )
    if 'use_uncertainty' in config['model']:
        base_fmt += f"useUncertainty{config['model']['use_uncertainty']}_"
    if 'sinpos_embed' in config['model']:
        base_fmt += f"sinpos{config['model']['sinpos_embed']}_"
    if 'fourier_embed' in config['model']:
        base_fmt += f"fourier{config['model']['fourier_embed']}_"
    base_fmt += (
        f"concat{config['model']['concat']}_crossattnonly{config['model']['cross_attn_only']}_"
        f"lr{config['training']['lr']}_"
    )
    if 'dropping_prob' in config['model']:
        base_fmt += f"modaldropP{config['model']['dropping_prob']}_"
    base_fmt += (
        f"batch{config['training']['batch']}_reg{config['model']['regularize']}_"
        f"aug{config['training']['aug']}_date{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    model_str = str(base_fmt)
    return model_str

# def create_model_str_classifier(config: Dict[str, Any], data_name: str) -> str:
#     base_fmt = f"{data_name}_"
#     if 'bottlenecklen' in config['model']:
#         base_fmt += f"{config['model']['bottlenecklen']}-{config['model']['bottleneckdim']}-"
#     else:
#         base_fmt += f"{config['model_new_encoder']['bottlenecklen']}-{config['model_new_encoder']['bottleneckdim']}-"
#     if 'classifier_dropout' in config['model']:
#         base_fmt += f"classifierdropP{config['model']['classifier_dropout']}_"
#     if 'dropping_prob' in config['model']:
#         base_fmt += f"modaldropP{config['model']['dropping_prob']}_"
#     base_fmt += (
#         f"lr{config['training']['lr']}_batch{config['training']['batch']}_reg{config['model']['regularize']}_"
#         f"aug{config['training']['aug']}_date{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
#     )
#     model_str = str(base_fmt)
#     return model_str

def create_model_str_classifier(config: Dict[str, Any], data_name: str) -> str:
    # Use a base format string and insert extra fields only if needed for compactness
    
    if 'pretrained_encoder_config' in config:
        bottleneck_len = config['pretrained_encoder_config']['model']['bottlenecklen']
        bottleneck_dim = config['pretrained_encoder_config']['model']['bottleneckdim']
    else:
        bottleneck_len = config['model_new_encoder']['bottlenecklen']
        bottleneck_dim = config['model_new_encoder']['bottleneckdim']
    
    base_fmt = f"{data_name}_{bottleneck_len}-{bottleneck_dim}-"
    
    # Include spectra and photometry token numbers if they are present
    if config['data']["lightcurve_test_name"] is not None and config['data']["spectra_test_name"] is not None:
        if 'pretrained_encoder_config' in config:
            if 'spectra_tokens' in config['pretrained_encoder_config']['model'] and 'photometry_tokens' in config['pretrained_encoder_config']['model']:
                spectra_tokens = config['pretrained_encoder_config']['model']['spectra_tokens']
                photometry_tokens = config['pretrained_encoder_config']['model']['photometry_tokens']
                base_fmt += f"{spectra_tokens}-{photometry_tokens}-"
        else:
            if 'model_new_encoder_multimodal' in config:
                spectra_tokens = config['model_new_encoder_multimodal']['spectra_tokens']
                photometry_tokens = config['model_new_encoder_multimodal']['photometry_tokens']
                base_fmt += f"{spectra_tokens}-{photometry_tokens}-"
    
    if 'pretrained_encoder_config' in config:
        encoder_layers = config['pretrained_encoder_config']['model']['encoder_layers']
        encoder_heads = config['pretrained_encoder_config']['model']['encoder_heads']
        model_dim = config['pretrained_encoder_config']['model']['model_dim']
    else:
        encoder_layers = config['model_new_encoder']['encoder_layers']
        encoder_heads = config['model_new_encoder']['encoder_heads']
        model_dim = config['model_new_encoder']['model_dim']
    
    base_fmt += f"{encoder_layers}-{encoder_heads}-{model_dim}_"
    
    base_fmt += f"useUncertainty{config['model_new_encoder']['use_uncertainty']}_"
    base_fmt += f"sinpos{config['model_new_encoder']['sinpos_embed']}_"
    base_fmt += f"fourier{config['model_new_encoder']['fourier_embed']}_"
    
    if 'pretrained_encoder_config' in config:
        concat = config['pretrained_encoder_config']['model']['concat']
        cross_attn_only = config['pretrained_encoder_config']['model']['cross_attn_only']
    else:
        concat = config['model_new_encoder']['concat']
        cross_attn_only = config['model_new_encoder']['cross_attn_only']
    
    base_fmt += (
        f"concat{concat}_crossattnonly{cross_attn_only}_"
        f"lr{config['training']['lr']}_"
    )
    
    if 'pretrained_encoder_config' in config:
        if 'dropping_prob' in config['pretrained_encoder_config']['model']:
            base_fmt += f"modaldropP{config['pretrained_encoder_config']['model']['dropping_prob']}_"
    else:
        if 'model_new_encoder_multimodal' in config:
            if 'dropping_prob' in config['model_new_encoder_multimodal']:
                base_fmt += f"modaldropP{config['model_new_encoder_multimodal']['dropping_prob']}_"
    
    base_fmt += (
        f"batch{config['training']['batch']}_reg{config['model']['regularize']}_"
        f"aug{config['training']['aug']}_date{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    model_str = str(base_fmt)
    return model_str