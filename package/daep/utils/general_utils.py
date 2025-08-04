from typing import Tuple
from typing import Dict, Any
from datetime import datetime

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
    from pathlib import Path

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
        raw_data_path = base_path + f"/data/{spectra_or_lightcurve}_raw"
    elif env == 'local':
        base_path = "/home/altair/Documents/UROP/2025_Summer/Perceiver-diffusion-autoencoder"
        model_path = base_path + f"/models/{spectra_or_lightcurve}"
        data_path = base_path + f"/data/{spectra_or_lightcurve}"
        raw_data_path = base_path + f"/data/{spectra_or_lightcurve}_raw"
    
    return base_path, model_path, data_path, raw_data_path
    
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