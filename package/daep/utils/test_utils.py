import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Sequence, List
import pytorch_lightning as L
import json
import re
from pytorch_lightning.callbacks import BasePredictionWriter

# Import the necessary modules
from daep.daep import unimodaldaep, multimodaldaep
from daep.datasets.GALAHspectra_dataset import GALAHDatasetProcessedSubset
from daep.datasets.TESSlightcurve_dataset import TESSDatasetProcessedSubset
from daep.utils.general_utils import create_model_str, create_model_str_classifier
from daep.utils.plot_utils import plot_spectra_simple  # type: ignore

def extract_epoch_from_model_path(model_path: str) -> str:
    """
    Extract epoch number from model path.
    
    Parameters
    ----------
    model_path : str
        Path to the model file
        
    Returns
    -------
    str
        Epoch number as string, or 'Unknown' if not found
    """
    epoch_match = re.search(r'epoch_(\d+)', str(model_path))
    if epoch_match:
        epoch_number = int(epoch_match.group(1))
        print(f"Detected epoch number from model path: {epoch_number}")
        return str(epoch_number)
    else:
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            epoch_number = checkpoint['epoch']
            print(f"Detected epoch number from checkpoint: {epoch_number}")
            return str(epoch_number)
        except Exception as e:
            print("Epoch number not found in model path.")
            return 'Unknown'

def create_analysis_directory(config: Dict[str, Any], models_path: Path, 
                           test_name: str, epoch_number: str, data_name: str) -> Path:
    """
    Create analysis directory path based on configuration and model parameters.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    models_path : Path
        Path to models directory
    test_name : str
        Name of the test dataset
    epoch_number : str
        Epoch number from model path
        
    Returns
    -------
    Path
        Path to analysis directory
    """
    model_str = create_model_str(config, data_name)
    analysis_dir = models_path / test_name / model_str / f"epoch_{epoch_number}" / "analysis_results"
    return analysis_dir

def print_evaluation_metrics(metrics: Dict[str, float], input_modalities: Optional[list] = None, output_modalities: Optional[list] = None):
    """
    Print evaluation metrics in a formatted way.
    
    Parameters
    ----------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    def primary(metrics):
        print("\n=== Model Evaluation Results ===")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"Mean Relative Error: {metrics['mean_relative_error']:.6f}")
        print(f"68% Coverage: {metrics['coverage_68']:.3f} (expected: 0.68)")
        print(f"95% Coverage: {metrics['coverage_95']:.3f} (expected: 0.95)")
        print(f"Mean Uncertainty Width: {metrics['mean_uncertainty_width']:.6f}")
        print(f"Uncertainty-Error Correlation: {metrics['uncertainty_error_correlation']:.3f}")
    
    if len(input_modalities) <= 1 and len(output_modalities) <= 1:
        primary(metrics)
    else:
        for input_modality in input_modalities:
            for output_modality in output_modalities:
                print(f"=== Input: {input_modality} to Output: {output_modality} ===")
                primary(metrics[input_modality][output_modality])

def auto_detect_model_path(config: Dict[str, Any], models_path: Path, test_name: str) -> str:
    """
    Auto-detect the most recent model checkpoint based on configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and training parameters
    models_path : Path
        Path to the models directory
    test_name : str
        Name of the test dataset
        
    Returns
    -------
    str
        Path to the most recent model checkpoint
    """
    # Look for models matching the current configuration
    try:
        model_str = create_model_str(config, '*')
    except KeyError:
        model_str = create_model_str_classifier(config, '*')
    
    # Remove 'date' and all characters after it in model_str for pattern matching
    import re
    model_str = re.sub(r'date.*', '', model_str)
    model_pattern = f"{model_str}*"

    ckpt_dir = models_path / test_name / "ckpt"
    if ckpt_dir.exists():
        available_models = list(ckpt_dir.glob(model_pattern))
        if not available_models:
            raise FileNotFoundError(f"No models matching model pattern {model_pattern} found in {ckpt_dir}")
        
        # Use the most recent matching model
        model_path = sorted(available_models, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        print(f"Found matching model: {model_path}")
        return model_path
    else:
        models_path = models_path / test_name
        if not models_path.exists():
            raise FileNotFoundError(f"Models path not found: {models_path}")
        
        # Only include directories matching the model pattern
        available_model_dirs = [p for p in models_path.glob(model_pattern) if p.is_dir()]
        if not available_model_dirs:
            raise FileNotFoundError(f"No models matching model pattern {model_pattern} found in {models_path}")
        
        # Use the most recent matching model
        model_dir = sorted(available_model_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        print(f"Found matching model: {model_dir}")
        # Take the highest epoch checkpoint within model_dir
        checkpoint_files = list((model_dir / "ckpt").glob("*.pth"))
        if not checkpoint_files:
            # If no checkpoint files found in model_dir/ckpt, search inside the most recently created subfolder within model_dir
            subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
            if not subdirs:
                raise FileNotFoundError(f"No model subdirectories found in {model_dir / 'ckpt'}")
            # Sort subdirectories by modification time (most recent first)
            most_recent_subdir = sorted(subdirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            checkpoint_files = list((most_recent_subdir / 'ckpt').glob("*.pth"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {most_recent_subdir}")
        # Extract epoch numbers from filenames and select the highest
        import re
        def extract_epoch(filename):
            match = re.search(r"epoch_(\d+)", filename.stem)
            return int(match.group(1)) if match else -1
        checkpoint_files_sorted = sorted(checkpoint_files, key=lambda x: extract_epoch(x), reverse=True)
        last_ckpt = checkpoint_files_sorted[0]
        return last_ckpt        
        

def load_trained_model(model_path: Path, device: torch.device, config: Dict[str, Any], spectra_or_lightcurve: str = "spectra") -> unimodaldaep:
    """
    Load a trained model.
    
    Parameters
    ----------
    model_path : Path
        Path to the saved model
    model_params : dict
        Model parameters used during training
    spectra_or_lightcurve : str, optional
        "spectra" or "lightcurve" to specify the type of data to train on. Defaults to "spectra".
    Returns
    -------
    unimodaldaep
        Loaded model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load the model with weights_only=False for compatibility with older saved models
    # This is safe since we're loading our own trained models
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load model with weights_only=False: {e}")
        print("Trying with safe globals...")
        # Alternative approach: add our custom class to safe globals
        from torch.serialization import add_safe_globals
        add_safe_globals([unimodaldaep, multimodaldaep])
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Check if the checkpoint is a model or a dict containing more information
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            # If the checkpoint contains a model, we can just load the model
            model = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            # If the checkpoint is a state dict, we need to instantiate the model and load the state dict
            if spectra_or_lightcurve == "spectra":
                from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore2stages
                encoder_type = spectraTransceiverEncoder
                score_type = spectraTransceiverScore2stages
            elif spectra_or_lightcurve == "lightcurve":
                from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore2stages
                encoder_type = photometricTransceiverEncoder
                score_type = photometricTransceiverScore2stages
            encoder = encoder_type(
                bottleneck_length=config["model"]["bottlenecklen"],
                bottleneck_dim=config["model"]["bottleneckdim"],
                model_dim=config["model"]["model_dim"],
                num_heads=config["model"]["encoder_heads"],
                ff_dim=config["model"]["model_dim"],
                num_layers=config["model"]["encoder_layers"],
                concat=config["model"]["concat"]
            ).to(device)
            score = score_type(
                bottleneck_dim=config["model"]["bottleneckdim"],
                model_dim=config["model"]["model_dim"],
                num_heads=config["model"]["decoder_heads"],
                ff_dim=config["model"]["model_dim"],
                num_layers=config["model"]["decoder_layers"],
                concat=config["model"]["concat"],
                cross_attn_only=config["model"]["cross_attn_only"]
            ).to(device)
            model = unimodaldaep(encoder, score, regularize=config["model"]["regularize"]).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If the checkpoint is a model, we can just load the model
        model = checkpoint
    
    if 'use_uncertainty' in config["model"]:
        model.output_uncertainty = config["model"]["use_uncertainty"]
    else:
        model.output_uncertainty = False
    
    model.eval()
    return model


def calculate_metrics(results: Dict[str, np.ndarray], input_modalities: Optional[list] = None, output_modalities: Optional[list] = None) -> Dict[str, float]:
    """
    Calculate evaluation metrics for the model predictions.
    
    Parameters
    ----------
    results : dict
        Dictionary containing:
        - predictions : np.ndarray
            Model predictions
        - ground_truth : np.ndarray
            Ground truth values
        - ground_truth_uncertainties : np.ndarray
            Ground truth measurement uncertainties
        - uncertainties : np.ndarray
            Model prediction uncertainties
    input_modalities : list, optional
        List of input modalities for multimodal models
    output_modalities : list, optional
        List of output modalities for multimodal models
        
    Returns
    -------
    dict
        Dictionary containing various metrics with native Python types:
        - mse: Mean squared error
        - mae: Mean absolute error
        - rmse: Root mean squared error
        - mean_relative_error: Mean relative error
        - coverage_68: Fraction of predictions where ground truth falls within ±1σ (68% CI)
        - coverage_95: Fraction of predictions where ground truth falls within ±1.96σ (95% CI)
        - coverage_68_gt: Fraction where prediction error ≤ ground truth uncertainty
        - coverage_95_gt: Fraction where prediction error ≤ 1.96 × ground truth uncertainty
        - mean_uncertainty_width: Mean width of prediction uncertainty intervals
        - uncertainty_error_correlation: Correlation between prediction uncertainties and errors
        
    Notes
    -----
    The coverage metrics assess uncertainty calibration:
    - coverage_68/95: Check if model uncertainties are well-calibrated (should be close to 0.68/0.95)
    - coverage_68_gt/95_gt: Assess if prediction errors are comparable to ground truth uncertainties
    """
    
    def primary(results):
        predictions = results['predictions']
        ground_truth = results['ground_truth']
        ground_truth_uncertainties = results['ground_truth_uncertainties']
        uncertainties = results['uncertainties']
        
        # Mask outliers where the absolute z-score is greater than 10 (i.e., > 10 sigma)
        z_scores = (predictions - ground_truth) / (uncertainties + 1e-8)
        mask = np.abs(z_scores) <= 10.0
        # Apply the mask to all relevant arrays to exclude outliers from metric calculations
        predictions = predictions[mask]
        ground_truth = ground_truth[mask]
        ground_truth_uncertainties = ground_truth_uncertainties[mask]
        uncertainties = uncertainties[mask]
        
        # Calculate basic error metrics
        mse = float(np.nanmean((predictions - ground_truth) ** 2))
        mae = float(np.nanmean(np.abs(predictions - ground_truth)))
        rmse = float(np.sqrt(mse))
        
        # Calculate relative errors
        relative_error = np.abs(predictions - ground_truth) / (np.abs(ground_truth) + 1e-8)
        mean_relative_error = float(np.nanmean(relative_error))
        
        # Calculate uncertainty calibration metrics
        # Check if uncertainties are well-calibrated (coverage)
        # For 68% confidence interval: check if ground truth falls within ±1σ of predictions
        # For 95% confidence interval: check if ground truth falls within ±1.96σ of predictions
        z_scores = (predictions - ground_truth) / (uncertainties + 1e-8)
        coverage_68 = float(np.nanmean(np.abs(z_scores) <= 1.0))  # 68% confidence interval
        coverage_95 = float(np.nanmean(np.abs(z_scores) <= 1.96))  # 95% confidence interval
        
        # Calculate ground truth uncertainty quality metrics
        # This assesses the relationship between ground truth uncertainties and prediction errors
        # A well-calibrated model should have prediction errors comparable to ground truth uncertainties
        prediction_errors = np.abs(predictions - ground_truth)
        uncertainty_ratio = prediction_errors / (ground_truth_uncertainties + 1e-8)
        coverage_68_gt = float(np.nanmean(uncertainty_ratio <= 1.0))  # Fraction where prediction error ≤ ground truth uncertainty
        coverage_95_gt = float(np.nanmean(uncertainty_ratio <= 1.96))  # Fraction where prediction error ≤ 1.96 × ground truth uncertainty
        
        # Calculate mean uncertainty width
        mean_uncertainty_width = float(np.nanmean(uncertainties))
        mean_uncertainty_width_gt = float(np.nanmean(ground_truth_uncertainties))
        
        # Calculate correlation between uncertainty and error
        errors = np.abs(predictions - ground_truth)
        uncertainty_error_correlation = float(np.corrcoef(uncertainties.flatten(), errors.flatten())[0, 1])
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mean_relative_error': mean_relative_error,
            'coverage_68': coverage_68,
            'coverage_95': coverage_95,
            'coverage_68_gt': coverage_68_gt,
            'coverage_95_gt': coverage_95_gt,
            'mean_uncertainty_width': mean_uncertainty_width,
            'mean_uncertainty_width_gt': mean_uncertainty_width_gt,
            'uncertainty_error_correlation': uncertainty_error_correlation
        }
    
    if len(input_modalities) <= 1 and len(output_modalities) <= 1:
        return primary(results)
    else:
        all_metrics = {input_modality: {output_modality: None for output_modality in output_modalities} for input_modality in input_modalities}
        for input_modality in input_modalities:
            for output_modality in output_modalities:
                print(f"=== Calculating metrics for input modality: {input_modality} to output modality: {output_modality} ===")
                all_metrics[input_modality][output_modality] = primary(results[input_modality][output_modality])
        return all_metrics


def plot_example_spectra(results: Dict[str, np.ndarray], test_dataset: GALAHDatasetProcessedSubset, 
                        num_examples: int, save_dir: str):
    """
    Plot example spectra showing predictions vs ground truth with residuals using plot_spectra_simple.
    
    Parameters
    ----------
    results : dict
        Results from evaluate_model containing 'predictions', 'ground_truth', 
        'uncertainties', and 'wavelengths' arrays
    test_dataset : GALAHDataset
        Test dataset containing object IDs
    num_examples : int, default=3
        Number of example spectra to plot
    save_dir : str, default='test_results'
        Directory to save plots
        
    Notes
    -----
    This function creates two plots per example: an overlay comparison plot showing ground truth
    and predictions with uncertainty bands, and a residuals plot. Both use plot_spectra_simple
    as the base for consistent formatting with spectral line annotations.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    predictions = results['predictions']
    ground_truth = results['ground_truth']
    ground_truth_uncertainties = results['ground_truth_uncertainties']
    uncertainties = results['uncertainties']
    wavelengths = results['wavelengths']
    test_instance_idxs = results['test_instance_idxs']
    sobject_ids = results['sobject_ids']
    
    # # Select random examples
    # indices = np.random.choice(test_instance_idxs, num_examples, replace=False)
    indices = np.random.choice(len(predictions), num_examples, replace=False)
    
    for i, idx in enumerate(indices):
        # Get the sobject_id for this example
        sobject_id = int(test_dataset.ids[idx, 2])  # Assuming sobject_id is in column 2
        
        # Create overlay comparison plot
        print(f"Creating overlay comparison plot for object {sobject_id}")
        
        # First, create the base plot with ground truth
        fig, axes = plot_spectra_simple(
            sobject_id=sobject_id,
            fluxes=ground_truth[idx],
            wavelengths=wavelengths[idx],
            uncertainties=ground_truth_uncertainties[idx],
            plot_elements=True,
            savefig=False,
            showfig=False,
            save_dir=''
        )
        
        # Then overlay predictions on the same axes
        plot_spectra_simple(
            sobject_id=sobject_id,
            fluxes=predictions[idx],
            wavelengths=wavelengths[idx],
            uncertainties=uncertainties[idx],
            plot_elements=False,  # Don't add spectral lines again
            savefig=False,
            showfig=False,
            save_dir='',
            fig=fig,
            axes=axes
        )
        
        # Update titles and legends for the overlay plot
        for ccd in range(4):
            ax = axes[ccd]
            ax.set_title(f'CCD {ccd+1} - Object {sobject_id} (Ground Truth + Prediction)')
        axes[-1].legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(save_path / f'overlay_comparison_{sobject_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create residuals plot
        print(f"Creating residuals plot for object {sobject_id}")
        residuals = predictions[idx] - ground_truth[idx]
        
        fig_res, axes_res = plot_spectra_simple(
            sobject_id=sobject_id,
            fluxes=residuals,
            wavelengths=wavelengths[idx],
            uncertainties=None,
            plot_elements=True,
            savefig=False,
            showfig=False,
            save_dir=''
        )
        
        # Add horizontal reference line at zero for residuals
        for ccd in range(4):
            ax = axes_res[ccd]
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, lw=0.5)
            ax.set_ylabel('Residual (Pred - Actual)')
            ax.set_title(f'CCD {ccd+1} - Object {sobject_id} (Residuals)', fontsize=10)
            ax.set_ylim(-0.4, 0.4)  # Adjust y-axis limits for residuals
        
        # Increase horizontal and vertical padding to add more space between CCD subplots
        plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=1.0)
        plt.savefig(save_path / f'residuals_{sobject_id}.png', dpi=300)  #, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plots for object {sobject_id}: overlay comparison and residuals")


def plot_lightcurve_simple(ticid: int, fluxes: np.ndarray, times: np.ndarray, uncertainties: np.ndarray = None,
                        plot_elements=True, savefig=False, showfig=True, save_dir='', starclass=None,
                        fig=None, axes=None):
    """
    Plot the lightcurve for a given object.

    Parameters
    ----------
    ticid : int
        Identifier for the lightcurve object to be plotted.
    fluxes : np.ndarray
        Array of flux values for the lightcurve
    times : np.ndarray
        Array of time values corresponding to the fluxes
    uncertainties : np.ndarray, optional
        Array of uncertainty values for the fluxes. If None, uncertainties are not shown.
    plot_elements : bool, default=True
        Whether to overlay lightcurve markers and element labels.
    savefig : bool, default=False
        Whether to save the figure to disk.
    showfig : bool, default=True
        Whether to display the figure interactively.
    save_dir : str, default=''
        Directory path to save the figure if `savefig` is True.
    fig : matplotlib.figure.Figure, optional
        Existing figure object to plot on. If provided, `axes` must also be provided.
    axes : np.ndarray, optional
        Existing axes array to plot on. If provided, `fig` must also be provided.

    Returns
    -------
    f : matplotlib.figure.Figure or None
        The matplotlib Figure object if `showfig` is False, otherwise None.
    ccds : np.ndarray or None
        Array of Axes objects for each CCD panel if `showfig` is False, otherwise None.

    Notes
    -----
    If `fig` and `axes` are provided, the function will plot on the existing axes instead of creating new ones.
    """
    # Check if using existing figure/axes or creating new ones
    if fig is not None and axes is not None:
        f, ax = fig, axes
        use_existing = True
    elif fig is None and axes is None:
        f, ax = plt.subplots(1, 1, figsize=(12, 8))
        use_existing = False
    else:
        raise ValueError("Both `fig` and `axes` must be provided together, or both must be None")
    
    if not use_existing:
        kwargs_sob = dict(c = 'k', lw=0.5, label='Flux', rasterized=True)
        kwargs_error_spectrum = dict(color = 'grey', label='Flux error', rasterized=True)
    else:
        kwargs_sob = dict(color='cyan', label='Predicted Flux', lw=0.5, rasterized=True)
        kwargs_error_spectrum = dict(color='blue', alpha=0.2, label='Predicted Flux Error', rasterized=True)
        
    # Create plots
    
    
    # Plot the uncertainty if provided
    if uncertainties is not None:
        ax.fill_between(
            times,
            fluxes - uncertainties,
            fluxes + uncertainties,
            **kwargs_error_spectrum
            )
    
    # Overplot observed light curve
    ax.plot(
        times,
        fluxes,
        **kwargs_sob
        )
    
    # Set title, labels, and limits
    ax.set_xlabel('Time')
    ax.set_ylabel('Flux')
    ax.set_title(f'Lightcurve - {ticid} - {starclass}')
    ax.grid(True, alpha=0.3)
    
    # Only call tight_layout if not using existing figure
    if not use_existing:
        plt.tight_layout()
    
    if savefig:
        if len(save_dir) > 0:
            plt.savefig(Path(save_dir) / f'{ticid}.png',bbox_inches='tight',dpi=200)
        else:
            print('No save directory provided, so not saving figure')
    if showfig:
        plt.show()
        plt.close()
        return None, None
    else:
        return f, ax

def plot_example_lightcurves(results: Dict[str, np.ndarray], test_dataset: TESSDatasetProcessedSubset, 
                             num_examples: int, save_dir: str):
    """
    Plot example lightcurves showing predictions vs ground truth with residuals using plot_lightcurve_simple.
    
    Parameters
    ----------
    results : dict
        Results from evaluate_model containing 'predictions', 'ground_truth', 
        'uncertainties', and 'times' arrays
    test_dataset : TESSDataset
        Test dataset containing ticids
    num_examples : int
        Number of example lightcurves to plot
    save_dir : str, default='test_results'
        Directory to save plots
        
    Notes
    -----
    This function creates two plots per example: an overlay comparison plot showing ground truth
    and predictions with uncertainty bands, and a residuals plot. Both use plot_spectra_simple
    as the base for consistent formatting with spectral line annotations.
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    predictions = results['predictions']
    ground_truth = results['ground_truth']
    ground_truth_uncertainties = results['ground_truth_uncertainties']
    uncertainties = results['uncertainties']
    times = results['times']
    test_instance_idxs = results['test_instance_idxs']
    ticids = results['ticids']
    
    # Select random examples
    # indices = np.random.choice(test_instance_idxs, num_examples, replace=False)
    indices = np.random.choice(len(predictions), num_examples, replace=False)
    
    for i, idx in enumerate(indices):
        ticid = ticids[idx]
        # Create overlay comparison plot
        print(f"Creating overlay comparison plot for object {ticid}")
        
        # First, create the base plot with ground truth
        actual_lightcurve = test_dataset.get_actual_lightcurve(idx)
        starclass = actual_lightcurve['starclass']
        
        fig, axes = plot_lightcurve_simple(
            ticid=ticid,
            fluxes=ground_truth[idx],
            times=times[idx],
            uncertainties=ground_truth_uncertainties[idx],
            plot_elements=True,
            savefig=False,
            showfig=False,
            save_dir='',
            starclass=starclass
        )
        
        # Then overlay predictions on the same axes
        plot_lightcurve_simple(
            ticid=ticid,
            fluxes=predictions[idx],
            times=times[idx],
            # uncertainties=uncertainties[idx],
            plot_elements=False,  # Don't add spectral lines again
            savefig=False,
            showfig=False,
            save_dir='',
            fig=fig,
            axes=axes,
            starclass=starclass
        )
        
        # Update titles and legends for the overlay plot
        axes.set_title(f'Object {ticid} (Ground Truth + Prediction)')
        axes.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(save_path / f'overlay_comparison_{ticid}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create residuals plot
        print(f"Creating residuals plot for object {ticid}")
        residuals = predictions[idx] - ground_truth[idx]
        
        fig_res, ax_res = plot_lightcurve_simple(
            ticid=ticid,
            fluxes=residuals,
            times=times[idx],
            uncertainties=None,
            plot_elements=True,
            savefig=False,
            showfig=False,
            save_dir='',
            starclass=starclass
        )
        
        # Add horizontal reference line at zero for residuals
        ax_res.axhline(y=0, color='k', linestyle='--', alpha=0.5, lw=0.5)
        ax_res.set_ylabel('Residual (Pred - Actual)')
        ax_res.set_title(f'Object {ticid} (Residuals)', fontsize=10)
        ax_res.set_ylim(-0.4, 0.4)  # Adjust y-axis limits for residuals
        
        # Increase horizontal and vertical padding to add more space between CCD subplots
        plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=1.0)
        plt.savefig(save_path / f'residuals_{ticid}.png', dpi=300)  #, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plots for object {ticid}: overlay comparison and residuals")

def plot_metrics_summary(metrics: Dict[str, float], save_dir: str = 'test_results',
                         input_modalities: Optional[list] = None, output_modalities: Optional[list] = None):
    """
    Create a summary plot of the evaluation metrics.
    
    Parameters
    ----------
    metrics : dict
        Metrics from calculate_metrics
    save_dir : str
        Directory to save the plot
    """
    
    def primary(metrics, save_dir):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Create a summary figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Error metrics
        error_metrics = ['mse', 'mae', 'rmse', 'mean_relative_error']
        error_values = [metrics[m] for m in error_metrics]
        axes[0, 0].bar(error_metrics, error_values, color='skyblue')
        axes[0, 0].set_title('Error Metrics')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Coverage metrics
        coverage_metrics = ['coverage_68', 'coverage_95']
        coverage_gt_metrics = ['coverage_68_gt', 'coverage_95_gt']
        coverage_values = [metrics[m] for m in coverage_metrics]
        coverage_gt_values = [metrics[m] for m in coverage_gt_metrics]
        expected_coverage = [0.68, 0.95]
        x = np.arange(len(coverage_metrics))
        
        total_width = 0.7  # Total width allotted for all bars at each x-tick
        n_bars = 3
        bar_width = total_width / n_bars
        offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, n_bars)

        axes[0, 1].bar(x + offsets[0], coverage_values, bar_width, label='Actual', color='lightcoral')
        axes[0, 1].bar(x + offsets[1], coverage_gt_values, bar_width, label='Ground Truth', color='skyblue')
        axes[0, 1].bar(x + offsets[2], expected_coverage, bar_width, label='Expected', color='lightgreen')
        # Added comment: Adjusted bar positions and widths for clearer grouped bar visualization.
        axes[0, 1].set_title('Uncertainty Coverage')
        axes[0, 1].set_ylabel('Coverage')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(coverage_metrics)
        axes[0, 1].legend()
        
        # Plot 3: Uncertainty width
        axes[1, 0].bar(0, [abs(metrics['mean_uncertainty_width'])], 0.5, color='lightcoral')
        axes[1, 0].bar(1, [abs(metrics['mean_uncertainty_width_gt'])], 0.5, color='skyblue')
        axes[1, 0].set_title('Mean Uncertainty Width')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_xticks(np.arange(2))
        axes[1, 0].set_xticklabels(['Mean Uncertainty Width', 'Ground Truth Mean Uncertainty Width'])
        
        # Plot 4: Correlation
        axes[1, 1].bar(['Uncertainty-Error\nCorrelation'], [metrics['uncertainty_error_correlation']], color='gold')
        axes[1, 1].set_title('Uncertainty-Error Correlation')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].set_ylim(-1, 1)
        
        plt.tight_layout()
        plt.savefig(save_path / 'metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved metrics summary to {save_path / 'metrics_summary.png'}")
    
    if len(input_modalities) <= 1 and len(output_modalities) <= 1:
        primary(metrics, save_dir)
    else:
        for input_modality in input_modalities:
            for output_modality in output_modalities:
                primary(metrics[input_modality][output_modality], save_dir / f"input_{input_modality}_output_{output_modality}")


def save_results(results: Dict[str, np.ndarray], metrics: Dict[str, float], save_dir: Path,
                 spectra_or_lightcurve: str = "spectra", input_modalities: Optional[list] = None,
                 output_modalities: Optional[list] = None):
    """
    Save prediction results and metrics to files.
    
    Parameters
    ----------
    results : dict
        Results from evaluate_model
    metrics : dict
        Metrics from calculate_metrics
    save_dir : Path
        Directory to save results
    spectra_or_lightcurve : str
        "spectra" or "lightcurve" to specify the type of data to train on. Defaults to "spectra".
    """
    
    def primary(results, metrics, save_dir, spectra_or_lightcurve):
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results as numpy arrays
        np.save(save_dir / "predictions.npy", results['predictions'])
        np.save(save_dir / "ground_truth.npy", results['ground_truth'])
        np.save(save_dir / "ground_truth_uncertainties.npy", results['ground_truth_uncertainties'])
        np.save(save_dir / "uncertainties.npy", results['uncertainties'])
        if spectra_or_lightcurve == "spectra":
            np.save(save_dir / "wavelengths.npy", results['wavelengths'])
        elif spectra_or_lightcurve == "lightcurve":
            np.save(save_dir / "times.npy", results['times'])
        np.save(save_dir / "test_instance_idxs.npy", results['test_instance_idxs'])
        
        # Save sobject_ids as text file (since it's a list)
        if spectra_or_lightcurve == "spectra":
            with open(save_dir / "sobject_ids.txt", 'w') as f:
                for sobject_id in results['sobject_ids']:
                    f.write(f"{sobject_id}\n")
        elif spectra_or_lightcurve == "lightcurve":
            with open(save_dir / "ticids.txt", 'w') as f:
                for ticid in results['ticids']:
                    f.write(f"{ticid}\n")
        
        # Save metrics as JSON
        with open(save_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Results saved to: {save_dir}")
    
    if len(input_modalities) <= 1 and len(output_modalities) <= 1:
        primary(results, metrics, save_dir, spectra_or_lightcurve)
    else:
        for input_modality in input_modalities:
            for output_modality in output_modalities:
                primary(results[input_modality][output_modality], metrics[input_modality][output_modality],
                             save_dir / f"input_{input_modality}_output_{output_modality}", spectra_or_lightcurve)


def load_results(save_dir: Path, spectra_or_lightcurve: str = "spectra") -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Load prediction results and metrics from files.
    
    Parameters
    ----------
    save_dir : Path
        Directory containing saved results
    spectra_or_lightcurve : str
        "spectra" or "lightcurve" to specify the type of data to train on. Defaults to "spectra".
    Returns
    -------
    tuple
        (results_dict, metrics_dict)
    """
    # Load numpy arrays
    results = {
        'predictions': np.load(save_dir / "predictions.npy"),
        'ground_truth': np.load(save_dir / "ground_truth.npy"),
        'ground_truth_uncertainties': np.load(save_dir / "ground_truth_uncertainties.npy"),
        'uncertainties': np.load(save_dir / "uncertainties.npy"),
        'test_instance_idxs': np.load(save_dir / "test_instance_idxs.npy")
    }
    if spectra_or_lightcurve == "spectra":
        results['wavelengths'] = np.load(save_dir / "wavelengths.npy")
    elif spectra_or_lightcurve == "lightcurve":
        results['times'] = np.load(save_dir / "times.npy")
    
    # Load sobject_ids
    if spectra_or_lightcurve == "spectra":
        with open(save_dir / "sobject_ids.txt", 'r') as f:
            results['sobject_ids'] = [int(line.strip()) for line in f]
    elif spectra_or_lightcurve == "lightcurve":
        with open(save_dir / "ticids.txt", 'r') as f:
            results['ticids'] = [int(line.strip()) for line in f]
    
    # Load metrics
    with open(save_dir / "metrics.json", 'r') as f:
        metrics = json.load(f)
    
    return results, metrics


def plot_results_from_saved(results_dir: str, test_dataset: GALAHDatasetProcessedSubset | TESSDatasetProcessedSubset, 
                           num_examples: int = 3, save_dir: str = 'analysis_results',
                           spectra_or_lightcurve: str = "spectra", input_modalities: Optional[list] = None,
                           output_modalities: Optional[list] = None):
    """
    Load saved results and create plots.
    
    Parameters
    ----------
    results_dir : str
        Directory containing saved results
    test_dataset : GALAHDataset or TESSDataset
        Test dataset for getting object IDs or ticids
    num_examples : int, default=3
        Number of example spectra or lightcurves to plot
    save_dir : str, default='analysis_results'
        Directory to save plots
    spectra_or_lightcurve : str
        "spectra" or "lightcurve" to specify the type of data to train on. Defaults to "spectra".
    """
    def primary(results_dir, test_dataset, num_examples, save_dir, spectra_or_lightcurve):
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        print(f"Loading results from: {results_dir}")
        results, metrics = load_results(results_path / 'saved_results', spectra_or_lightcurve)
        
        # Print metrics
        print_evaluation_metrics(metrics)
        
        # Create plots
        print("Creating example plots...")
        if spectra_or_lightcurve == "spectra":
            plot_example_spectra(results, test_dataset, num_examples, save_dir)
        elif spectra_or_lightcurve == "lightcurve":
            plot_example_lightcurves(results, test_dataset, num_examples, save_dir)
        
        print("Creating metrics summary...")
        plot_metrics_summary(metrics, save_dir)
        
        print(f"Plots saved to: {save_dir}")

    if len(input_modalities) <= 1 and len(output_modalities) <= 1:
        primary(results_dir, test_dataset, num_examples, save_dir, spectra_or_lightcurve)
    else:
        for input_modality in input_modalities:
            for output_modality in output_modalities:
                if output_modality == "spectra":
                    primary(results_dir, test_dataset.spectra_dataset, num_examples, save_dir / f"input_{input_modality}_output_{output_modality}", spectra_or_lightcurve)
                elif output_modality == "lightcurves":
                    primary(results_dir, test_dataset.lightcurve_dataset, num_examples, save_dir / f"input_{input_modality}_output_{output_modality}", spectra_or_lightcurve)




class UnprocessPredictionWriter(BasePredictionWriter):
    """
    Unprocesses model predictions back to original data units during prediction.

    This callback reads the sample indices from each batch (expects `batch['idx']`)
    and uses the underlying dataset's unprocessing helpers (e.g.,
    `unprocess_lightcurves` or `unprocess_spectra`) to invert preprocessing.

    Parameters
    ----------
    save_dir : str or pathlib.Path, optional
        If provided, unprocessed results will be saved under this directory per-batch.
        If None, results are only stored in-memory on `self.results`.
    write_interval : {"batch", "epoch"}
        When to write. Defaults to "batch".

    Notes
    -----
    - Works with datasets that expose `unprocess_lightcurves(idx, time, flux, flux_err=None)`
      or `unprocess_spectra(flux, idx)`.
    - Assumes batches include `idx` (base dataset indices) and, for lightcurves,
      a `time` tensor corresponding to the normalized time axis.
    - For distributed (DDP), this runs on each rank independently. Avoid file-name
      collisions by including rank/epoch/batch in filenames.
    """

    def __init__(self, write_interval: str = "batch") -> None:
        super().__init__(write_interval=write_interval)
        self.save_dir = None
        # Per-sample records and batch-aggregated arrays mirroring testing.py
        self.results: List[Dict[str, Any]] = []
        self._predictions: List[np.ndarray] = []
        self._ground_truth: List[np.ndarray] = []
        self._ground_truth_uncertainties: List[np.ndarray] = []
        self._uncertainties: List[np.ndarray] = []
        self._wavelengths_or_times: List[np.ndarray] = []
        self._indices: List[np.ndarray] = []
        self._star_ids: List[Any] = []
        self._dataset: Any = None
        self._task: Optional[str] = None  # "lightcurves" or "spectra"
        self._logger_dir: Optional[Path] = None

    def on_predict_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Cache the base dataset reference and detect task type.

        Attempts to unwrap nested Subset datasets to obtain the base dataset
        that holds the unprocessing helpers and normalization buffers.
        """
        # Resolve logger directory for default saving location
        log_dir = getattr(trainer.logger, "log_dir", None)
        self._logger_dir = Path(log_dir) if log_dir is not None else Path.cwd()

        # Determine dataset from predict dataloader 0
        ds = None
        try:
            vloaders = trainer.predict_dataloaders
            if isinstance(vloaders, (list, tuple)) and len(vloaders) > 0:
                ds = getattr(vloaders[0], "dataset", None)
        except Exception:
            ds = None

        # Unwrap torch.utils.data.Subset chains to reach base dataset
        unwrap_budget = 5
        while hasattr(ds, "dataset") and unwrap_budget > 0:
            ds = ds.dataset
            unwrap_budget -= 1
        self._dataset = ds

        # Detect task type by available unprocessing functions
        if hasattr(ds, "unprocess_lightcurves"):
            self._task = "lightcurves"
        elif hasattr(ds, "unprocess_spectra"):
            self._task = "spectra"
        else:
            raise ValueError("Dataset does not have unprocess_lightcurves or unprocess_spectra functions")

        # Prepare save directory
        self.save_dir = self._logger_dir / "predictions_unprocessed"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        predictions: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Dict[str, Any],
        dataloader_idx: int,
    ) -> None:
        """
        Unprocess predictions for a batch using dataset stats and per-sample indices.

        Parameters
        ----------
        predictions : Any
            Output from `predict_step`. Can be a tensor, dict, list or tuple.
        batch_indices : sequence of int or None
            Indices as tracked by Lightning. Prefer `batch['idx']` when present
            to ensure alignment with the base dataset.
        batch : dict
            Must contain `idx` and, for lightcurves, `time` in normalized units.
        """

        # Indices of current batch
        idxs = self._to_numpy(batch["idx"])  # matches testing.py using batch['idx']

        # Extract predictions. Handle MC samples or single prediction.
        flux_pred, flux_err_pred = self._extract_flux_predictions(predictions)
        # If predictions contain a leading MC-sample dimension, collapse with mean/std
        # Expect shapes: (num_samples, batch, length) or (batch, length)
        if flux_pred.ndim == 3:
            pred_mean = flux_pred.mean(axis=0)
            pred_std = flux_pred.std(axis=0)
        else:
            pred_mean = flux_pred
            # Placeholder std; will compute from dataset stats later
            pred_std = torch.zeros_like(flux_pred)

        # Collect actuals depending on task, then stack into numpy arrays for efficiency
        if self._task == "lightcurves":
            actuals = [self._dataset.get_actual_lightcurve(int(idx)) for idx in idxs]
            axis_values = np.stack([a["time"] for a in actuals])
        else:  # spectra
            actuals = [self._dataset.get_actual_spectrum(int(idx)) for idx in idxs]
            axis_values = np.stack([a["wavelength"] for a in actuals])
        ground_truth = np.stack([a["flux"] for a in actuals])
        ground_truth_uncertainties = np.stack([a["flux_errs"] for a in actuals])
        star_ids_batch = np.array([a["ids"][2] for a in actuals])

        # Convert prediction mean to numpy
        pred_mean_np = self._to_numpy(pred_mean)
        # Unprocess predictions to original units using dataset helper
        if self._task == "lightcurves":
            unprocessed = self._dataset.unprocess_lightcurves(idx=idxs, time=axis_values, flux=pred_mean_np)
            pred_mean_np = unprocessed["flux"]
            wavelengths_or_times = unprocessed["time"]
        else:  # spectra
            pred_mean_np = self._dataset.unprocess_spectra(flux=pred_mean_np, idx=idxs)
            wavelengths_or_times = axis_values

        # Handle uncertainties: use flux_err_pred mean if provided; else scale pred_std
        if flux_err_pred is not None:
            pred_uncertainty_np = self._to_numpy(flux_err_pred)
            # Scale by dataset stds
            pred_uncertainty_np = pred_uncertainty_np * np.repeat(
                self._dataset._fluxes_stds[idxs][:, None], pred_uncertainty_np.shape[1], axis=1
            )
            uncertainties = pred_uncertainty_np
        else:
            pred_std_np = self._to_numpy(pred_std)
            if np.all(pred_std_np <= 1e-9):
                pred_std_np = np.zeros_like(pred_mean_np)
            else:
                pred_std_np = pred_std_np * np.repeat(
                    self._dataset._fluxes_stds[idxs][:, None], pred_std_np.shape[1], axis=1
                )
            uncertainties = pred_std_np

        # Append batch results mirroring testing.py
        self._indices.append(idxs)
        self._predictions.append(pred_mean_np)
        self._ground_truth.append(ground_truth)
        self._ground_truth_uncertainties.append(ground_truth_uncertainties)
        self._uncertainties.append(uncertainties)
        self._wavelengths_or_times.append(wavelengths_or_times)
        self._star_ids.extend(star_ids_batch)

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Concatenate all accumulated batches and save a single results file mirroring testing.py outputs.
        """
        if len(self._predictions) == 0:
            return
        predictions = np.concatenate(self._predictions, axis=0)
        ground_truth = np.concatenate(self._ground_truth, axis=0)
        ground_truth_uncertainties = np.concatenate(self._ground_truth_uncertainties, axis=0)
        uncertainties = np.concatenate(self._uncertainties, axis=0)
        wavelengths_or_times = np.concatenate(self._wavelengths_or_times, axis=0)
        test_instance_idxs = np.concatenate(self._indices, axis=0)

        out = {
            "predictions": predictions,
            "ground_truth": ground_truth,
            "ground_truth_uncertainties": ground_truth_uncertainties,
            "uncertainties": uncertainties,
            "test_instance_idxs": test_instance_idxs,
        }
        # Field names follow testing.py
        if self._task == "lightcurves":
            out["times"] = wavelengths_or_times
            out["ticids"] = self._star_ids
        elif self._task == "spectra":
            out["wavelengths"] = wavelengths_or_times
            out["sobject_ids"] = self._star_ids

        # Save one NPZ for convenience
        if self.save_dir is not None:
            np.savez_compressed(self.save_dir / "prediction_results.npz", **out)

    # --------- helpers ---------
    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if x is None:
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _extract_flux_predictions(predictions: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract predicted flux (and optional flux uncertainty) following testing.py logic.

        Supported structures
        --------------------
        - Tuple(pred_flux_like, pred_flux_uncertainty_like)
        - Dict with one of:
          - 'flux'
          - 'photometry': {'flux', ...}
          - 'spectra': {'flux', ...}
        - Raw torch.Tensor
        """
        def extract_flux_like(x: Any) -> torch.Tensor:
            if torch.is_tensor(x):
                return x
            if isinstance(x, dict):
                if 'flux' in x:
                    return x['flux']
                if 'photometry' in x and isinstance(x['photometry'], dict) and 'flux' in x['photometry']:
                    return x['photometry']['flux']
                if 'spectra' in x and isinstance(x['spectra'], dict) and 'flux' in x['spectra']:
                    return x['spectra']['flux']
            raise ValueError("Unsupported prediction structure for flux extraction.")
        
        def extract_flux_uncertainty_like(x: Any) -> torch.Tensor:
            if torch.is_tensor(x):
                return x
            if isinstance(x, dict):
                if 'flux_err' in x:
                    return x['flux_err']
                elif 'photometry' in x and isinstance(x['photometry'], dict) and 'flux_err' in x['photometry']:
                    return x['photometry']['flux_err']
                elif 'spectra' in x and isinstance(x['spectra'], dict) and 'flux_err' in x['spectra']:
                    return x['spectra']['flux_err']
            raise ValueError("Unsupported prediction structure for flux uncertainty extraction.")

        # Case 1: (flux, flux_uncertainty) tuple when use_uncertainty=True
        if isinstance(predictions, (tuple, list)) and len(predictions) == 2:
            flux_like, flux_err_like = predictions
            flux = extract_flux_like(flux_like)
            flux_err = extract_flux_uncertainty_like(flux_err_like)
            return flux, flux_err
        
        # Case 2: dict with flux (no flux_err) or nested photometry/spectra
        if isinstance(predictions, dict):
            flux = extract_flux_like(predictions)
            flux_err = None
            return flux, flux_err
        
        # Case 3: raw tensor
        if torch.is_tensor(predictions):
            return predictions, None

        raise ValueError("Could not extract flux predictions from predict_step output; structure not supported.")

