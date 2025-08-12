import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Sequence, List
import pytorch_lightning as L
import json
import re
from pytorch_lightning.callbacks import BasePredictionWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.utils.data import Dataset
import seaborn as sns

# Import the necessary modules
from daep.daep import unimodaldaep, multimodaldaep
from daep.datasets.GALAHspectra_dataset import GALAHDatasetProcessedSubset
from daep.datasets.TESSlightcurve_dataset import TESSDatasetProcessedSubset
from daep.utils.general_utils import create_model_str, create_model_str_classifier
from daep.utils.plot_utils import plot_spectra_simple, plot_lightcurve_simple

def calculate_metrics(results: Dict[str, np.ndarray], input_modalities: list, output_modalities: list) -> Dict[str, float]:
    """
    Calculate evaluation metrics for the model predictions for all input-output modality combinations.
    
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
    input_modalities : list
        List of input modalities
    output_modalities : list
        List of output modalities
        
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
    
    def calc_metrics_for_modality(results):
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
    
    all_metrics = {input_modality: {output_modality: None for output_modality in output_modalities} for input_modality in input_modalities}
    for input_modality in input_modalities:
        for output_modality in output_modalities:
            print(f"=== Calculating metrics for input modality: {input_modality} to output modality: {output_modality} ===")
            all_metrics[input_modality][output_modality] = calc_metrics_for_modality(results[input_modality][output_modality])
    return all_metrics


def plot_example_spectra(results: Dict[str, np.ndarray],
                        num_examples: int, save_dir: str):
    """
    Plot example spectra showing predictions vs ground truth with residuals using plot_spectra_simple.
    
    Parameters
    ----------
    results : dict
        Results from evaluate_model containing 'predictions', 'ground_truth', 
        'uncertainties', and 'wavelengths' arrays
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
        sobject_id = int(sobject_ids[idx])  # Assuming sobject_id is in column 2
        
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


def plot_example_lightcurves(results: Dict[str, np.ndarray], 
                             num_examples: int, save_dir: str):
    """
    Plot example lightcurves showing predictions vs ground truth with residuals using plot_lightcurve_simple.
    
    Parameters
    ----------
    results : dict
        Results from evaluate_model containing 'predictions', 'ground_truth', 
        'uncertainties', and 'times' arrays
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
    starclass_arr = results.get('starclass', None)
    
    # Select random examples
    # indices = np.random.choice(test_instance_idxs, num_examples, replace=False)
    indices = np.random.choice(len(predictions), num_examples, replace=False)
    
    for i, idx in enumerate(indices):
        ticid = ticids[idx]
        # Create overlay comparison plot
        print(f"Creating overlay comparison plot for object {ticid}")
        
        # First, create the base plot with ground truth
        # Prefer saved starclass, fallback to dataset lookup
        if starclass_arr is not None:
            starclass = starclass_arr[idx]
        else:
            starclass = 'Not Provided'
        
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
    
    for input_modality in input_modalities:
        for output_modality in output_modalities:
            primary(metrics[input_modality][output_modality], save_dir / f"input_{'-'.join(input_modality)}_output_{output_modality}")


def save_results(results: Dict[str, np.ndarray], metrics: Dict[str, float], save_dir: Path,
                 input_modalities: list, output_modalities: list):
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
    """
    
    def save_results_for_modality(results, metrics, save_dir, output_modality):
        save_dir = save_dir / 'saved_results'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all results in a single compressed .npz file
        save_dict = {
            "predictions": results['predictions'],
            "ground_truth": results['ground_truth'],
            "ground_truth_uncertainties": results['ground_truth_uncertainties'],
            "uncertainties": results['uncertainties'],
            "test_instance_idxs": results['test_instance_idxs'],
        }
        if output_modality == "spectra":
            save_dict["wavelengths"] = results['wavelengths']
        elif output_modality == "lightcurves":
            save_dict["times"] = results['times']
            # Save starclass alongside lightcurves if available
            if 'starclass' in results:
                save_dict["starclass"] = results['starclass']
        np.savez_compressed(save_dir / f"predictions_and_ground_truth.npz", **save_dict)
        
        # Save sobject_ids as text file (since it's a list)
        if output_modality == "spectra":
            with open(save_dir / "sobject_ids.txt", 'w') as f:
                for sobject_id in results['sobject_ids']:
                    f.write(f"{sobject_id}\n")
        elif output_modality == "lightcurves":
            with open(save_dir / "ticids.txt", 'w') as f:
                for ticid in results['ticids']:
                    f.write(f"{ticid}\n")
        
        # Save metrics as JSON
        with open(save_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Results saved to: {save_dir}")
    
    for input_modality in input_modalities:
        for output_modality in output_modalities:
            save_results_for_modality(results[input_modality][output_modality], metrics[input_modality][output_modality],
                            save_dir / f"input_{'-'.join(input_modality)}_output_{output_modality}", output_modality)


def load_results(save_dir: Path, output_modality: str = "spectra") -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Load prediction results and metrics saved by `save_results`.

    Parameters
    ----------
    save_dir : Path
        Directory that contains the saved results for a specific input-output
        modality combination (e.g., `.../input_photometry_output_spectra`).
    output_modality : str, default="spectra"
        One of {"spectra", "lightcurves"}. Used to determine axis-name fields
        and ID filename.

    Returns
    -------
    tuple
        (results_dict, metrics_dict)

    Notes
    -----
    - This now reads from a single compressed NPZ file named
      `predictions_and_ground_truth.npz`, which contains all arrays saved by
      `save_results`.
    - Star IDs are still stored as text files (`sobject_ids.txt` or
      `ticids.txt`) alongside `metrics.json` in the same directory.
    """
    npz_path = save_dir / "predictions_and_ground_truth.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Expected results file not found: {npz_path}")

    data = np.load(npz_path)

    # Gather common fields from the NPZ container
    results: Dict[str, Any] = {
        "predictions": data["predictions"],
        "ground_truth": data["ground_truth"],
        "ground_truth_uncertainties": data["ground_truth_uncertainties"],
        "uncertainties": data["uncertainties"],
        "test_instance_idxs": data["test_instance_idxs"],
    }

    # Axis values key depends on the output modality
    if output_modality == "spectra":
        if "wavelengths" in data:
            results["wavelengths"] = data["wavelengths"]
        id_file = save_dir / "sobject_ids.txt"
        with open(id_file, "r") as f:
            results["sobject_ids"] = [int(line.strip()) for line in f]
    elif output_modality == "lightcurves":
        if "times" in data:
            results["times"] = data["times"]
        id_file = save_dir / "ticids.txt"
        with open(id_file, "r") as f:
            results["ticids"] = [int(line.strip()) for line in f]
        # Optional: starclass saved for lightcurves
        if "starclass" in data:
            results["starclass"] = data["starclass"]
    else:
        raise ValueError("output_modality must be one of {'spectra', 'lightcurves'}")

    # Load metrics JSON
    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return results, metrics

def plot_results_from_scratch(results, testing_loader, num_examples, analysis_dir, input_modalities, output_modalities):
    """
    Create plots directly from in-memory results structure returned by get_predictions.

    Expects `results` to be a nested dict: results[input_modality][output_modality] -> results dict.
    Writes plots into `analysis_dir/input_<in-mods>_output_<out-mod>`.
    """
    for input_modality_combo in input_modalities:
        for output_modality in output_modalities:
            subdir = analysis_dir / f"input_{'-'.join(input_modality_combo)}_output_{output_modality}"
            modality_results = results[input_modality_combo][output_modality]
            if output_modality == "spectra":
                plot_example_spectra(modality_results, num_examples, subdir)
            else:
                plot_example_lightcurves(modality_results, num_examples, subdir)
    

def plot_results_from_saved(
    results_dir: str,
    test_dataset: GALAHDatasetProcessedSubset | TESSDatasetProcessedSubset,
    num_examples: int = 3,
    save_dir: str = "analysis_results",
    input_modalities: Optional[list] = None,
    output_modalities: Optional[list] = None,
):
    """
    Load saved results (written by `save_results`) and create plots per
    input-output modality combination.

    Parameters
    ----------
    results_dir : str
        Root directory containing subfolders per modality combination, i.e.,
        `input_<in-mods>_output_<out-mod>` where arrays are saved in a single
        `predictions_and_ground_truth.npz` with associated `metrics.json` and
        star ID text files.
    test_dataset : GALAHDataset or TESSDataset
        Test dataset for getting object IDs or TIC/Sobject IDs.
    num_examples : int, default=3
        Number of example spectra or lightcurves to plot.
    save_dir : str, default="analysis_results"
        Directory where plots will be written, mirrored by modality folders.
    """

    base_results_dir = Path(results_dir)
    base_save_dir = Path(save_dir)

    def primary(modality_results_dir: Path, ds, num_examples: int, out_dir: Path, output_modality: str):
        if not modality_results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {modality_results_dir}")

        print(f"Loading results from: {modality_results_dir}")
        results, metrics = load_results(modality_results_dir, output_modality)

        # Create plots
        print("Creating example plots...")
        if output_modality == "spectra":
            plot_example_spectra(results, num_examples, out_dir)
        elif output_modality == "lightcurves":
            plot_example_lightcurves(results, num_examples, out_dir)

        print("Creating metrics summary...")
        plot_metrics_summary(metrics, out_dir)

        print(f"Plots saved to: {out_dir}")

    for input_modality in input_modalities:
        for output_modality in output_modalities:
            results_subdir = base_results_dir / f"input_{'-'.join(input_modality)}_output_{output_modality}"
            save_subdir = base_save_dir / f"input_{'-'.join(input_modality)}_output_{output_modality}"
            if output_modality == "spectra":
                primary(results_subdir, getattr(test_dataset, "spectra_dataset", test_dataset), num_examples, save_subdir, output_modality)
            elif output_modality == "lightcurves":
                primary(results_subdir, getattr(test_dataset, "lightcurve_dataset", test_dataset), num_examples, save_subdir, output_modality)

def make_confusion_matrix(model, class_names, device, output_dir):
    """
    Plot and save the confusion matrix for the given model.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model with a confusion matrix attribute.
    class_names : list
        List of class names for labeling the confusion matrix.
    device : torch.device or str
        Device to move the confusion matrix to.
    output_dir : str or Path
        Directory where the confusion matrix image will be saved.

    Notes
    -----
    Uses pathlib for path handling.
    """

    metric = model.conf_matrix.to(device)
    fig_, ax_ = metric.plot(labels=class_names)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    save_path = output_dir / f'confusion_matrix.png'
    plt.savefig(save_path)
    plt.close()


def calculate_classification_metrics(results: Dict[str, np.ndarray], dataset: Dataset,
                                   input_modalities: Optional[list] = None, 
                                   output_modalities: Optional[list] = None) -> Dict[str, float]:
    """
    Calculate classification evaluation metrics for the model predictions.
    
    Parameters
    ----------
    results : dict
        Dictionary containing:
        - predictions : np.ndarray
            Model predicted class labels
        - ground_truth : np.ndarray
            Ground truth class labels
        - prediction_probs : np.ndarray
            Model prediction probabilities
    input_modalities : list, optional
        List of input modalities for multimodal models
    output_modalities : list, optional
        List of output modalities for multimodal models
        
    Returns
    -------
    dict
        Dictionary containing classification metrics in the format:
        - confusion_matrix: confusion matrix array
        - class_metrics: dict with per-class accuracy, recall, precision, F1
        - total_metrics: dict with total accuracy, recall, precision, F1
    """
    
    def primary(results):
        predictions = results['predictions']
        ground_truth = results['ground_truth']
        
        # Ensure both predictions and ground_truth are in single class label format
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Predictions are one-hot encoded, convert to single labels
            predictions = np.argmax(predictions, axis=1)
        
        if len(ground_truth.shape) > 1 and ground_truth.shape[1] > 1:
            # Ground truth are one-hot encoded, convert to single labels
            ground_truth = np.argmax(ground_truth, axis=1)
        
        # Ensure both are 1D arrays
        predictions = predictions.flatten()
        print(predictions)
        ground_truth = ground_truth.flatten()
        print(ground_truth)
        
        # Get class names if available, otherwise use integers
        if hasattr(dataset, 'starclass_name_to_int'):
            # Use actual class names from the dataset
            class_names = list(dataset.starclass_name_to_int.keys())
            labels = list(range(len(class_names)))  # Use integer labels for sklearn functions
        else:
            # Fallback to integer labels if no class names available
            unique_classes = np.unique(np.concatenate([predictions, ground_truth]))
            class_names = [f'class_{i}' for i in unique_classes]
            labels = unique_classes
        
        # Calculate confusion matrix
        cm = confusion_matrix(ground_truth, predictions, labels=labels)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, average=None, zero_division=0, labels=labels
        )
        
        # Calculate per-class accuracy (diagonal of confusion matrix / row sums)
        class_accuracy = np.diag(cm) / (np.sum(cm, axis=1) + 1e-8)
        
        # Calculate total metrics
        total_accuracy = float(accuracy_score(ground_truth, predictions))
        total_precision, total_recall, total_f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='macro', zero_division=0, labels=labels
        )
        
        # Create per-class metrics dictionary using class names
        num_classes = len(cm)
        class_metrics = {}
        for i in range(num_classes):
            class_metrics[class_names[i]] = {
                'accuracy': float(class_accuracy[i]),
                'recall': float(recall[i]),
                'precision': float(precision[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        # Create total metrics dictionary
        total_metrics = {
            'accuracy': total_accuracy,
            'recall': float(total_recall),
            'precision': float(total_precision),
            'f1': float(total_f1)
        }
        
        return {
            'class_metrics': class_metrics,
            'total_metrics': total_metrics,
            'num_classes': num_classes,
            'class_names': class_names
        }
    
    if len(input_modalities) <= 1 and len(output_modalities) <= 1:
        return primary(results)
    else:
        all_metrics = {input_modality: {output_modality: None for output_modality in output_modalities} for input_modality in input_modalities}
        for input_modality in input_modalities:
            for output_modality in output_modalities:
                all_metrics[input_modality][output_modality] = primary(results[input_modality][output_modality])
        return all_metrics


def print_classification_metrics(metrics: Dict[str, float], 
                                input_modalities: Optional[list] = None, 
                                output_modalities: Optional[list] = None):
    """
    Print classification evaluation metrics in a formatted way.
    
    Parameters
    ----------
    metrics : dict
        Dictionary containing classification evaluation metrics
    input_modalities : list, optional
        List of input modalities for multimodal models
    output_modalities : list, optional
        List of output modalities for multimodal models
    """
    def primary(metrics):
        print("\n=== Classification Model Evaluation Results ===")
        
        # Print total metrics
        total = metrics['total_metrics']
        print(f"Total Accuracy: {total['accuracy']:.2f}")
        print(f"Total Recall: {total['recall']:.2f}")
        print(f"Total Precision: {total['precision']:.2f}")
        print(f"Total F1: {total['f1']:.2f}")
        
        # Print per-class metrics in table format
        print("\n=== Per-Class Performance ===")
        print("Class\t\tAccuracy\tRecall\t\tPrecision\tF1\t\tSupport")
        print("-" * 80)
        
        class_metrics = metrics['class_metrics']
        for class_name, class_data in class_metrics.items():
            print(f"{class_name}\t\t{class_data['accuracy']:.3f}\t\t{class_data['recall']:.3f}\t\t"
                  f"{class_data['precision']:.3f}\t\t{class_data['f1']:.3f}\t\t{class_data['support']}")
    
    if len(input_modalities) <= 1 and len(output_modalities) <= 1:
        primary(metrics)
    else:
        for input_modality in input_modalities:
            for output_modality in output_modalities:
                print(f"=== Input: {input_modality} to Output: {output_modality} ===")
                primary(metrics[input_modality][output_modality])


def plot_confusion_matrix(results: Dict[str, np.ndarray], save_dir: Path, 
                         input_modalities: Optional[list] = None, 
                         output_modalities: Optional[list] = None,
                         starclass_names: Optional[list] = None):
    """
    Plot confusion matrix for classification results.
    
    Parameters
    ----------
    results : dict
        Dictionary containing predictions and ground truth
    save_dir : Path
        Directory to save the plot
    input_modalities : list, optional
        List of input modalities for multimodal models
    output_modalities : list, optional
        List of output modalities for multimodal models
    dataset : Dataset, optional
        Dataset object to get class names from
    """
    def primary(results, save_dir, dataset=None):
        predictions = results['predictions']
        ground_truth = results['ground_truth']
        
        # Get class names if available
        if starclass_names is not None:
            class_names = starclass_names
            labels = list(range(len(class_names)))
        else:
            # Fallback to integer labels
            unique_classes = np.unique(np.concatenate([predictions, ground_truth]))
            class_names = [f'Class {i}' for i in unique_classes]
            labels = unique_classes
        
        # Calculate confusion matrix
        cm = confusion_matrix(ground_truth, predictions, labels=labels)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        save_path = save_dir / f'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {save_path}")
    
    
    for input_modality in input_modalities:
        for output_modality in output_modalities:
            subdir = save_dir / f"input_{input_modality}_output_{output_modality}"
            primary(results[input_modality][output_modality], subdir, dataset=dataset)


def plot_classification_metrics_summary(metrics: Dict[str, float], save_dir: Path,
                                       input_modalities: Optional[list] = None, 
                                       output_modalities: Optional[list] = None,
                                       dataset: Optional[Dataset] = None):
    """
    Create a table of the classification evaluation metrics as an image.
    
    Parameters
    ----------
    metrics : dict
        Metrics from calculate_classification_metrics
    save_dir : Path
        Directory to save the plot
    input_modalities : list, optional
        List of input modalities for multimodal models
    output_modalities : list, optional
        List of output modalities for multimodal models
    dataset : Dataset, optional
        Dataset object to get class names from
    """
    def primary(metrics, save_dir, modality_name="", dataset=None):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Create figure for table
        fig, ax = plt.subplots(figsize=(14, 10))  # Increased figure size for better readability
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        class_metrics = metrics['class_metrics']
        total_metrics = metrics['total_metrics']
        
        # Create table data
        table_data = []
        headers = ['Class', 'Accuracy', 'Recall', 'Precision', 'F1', 'Support']
        
        # Add per-class rows
        for class_name, class_data in class_metrics.items():
            table_data.append([
                class_name,
                f"{class_data['accuracy']:.3f}",
                f"{class_data['recall']:.3f}",
                f"{class_data['precision']:.3f}",
                f"{class_data['f1']:.3f}",
                str(class_data['support'])
            ])
        
        # Add total row
        table_data.append([
            'Total',
            f"{total_metrics['accuracy']:.3f}",
            f"{total_metrics['recall']:.3f}",
            f"{total_metrics['precision']:.3f}",
            f"{total_metrics['f1']:.3f}",
            str(sum(class_metrics[cls]['support'] for cls in class_metrics))
        ])
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)  # Slightly smaller font to fit more content
        table.scale(1.2, 1.8)  # Increased height scaling for better spacing
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight the total row
        for i in range(len(headers)):
            table[(len(table_data), i)].set_facecolor('#2196F3')
            table[(len(table_data), i)].set_text_props(weight='bold', color='white')
        
        plt.title(f'Classification Performance Table{modality_name}', fontsize=14, fontweight='bold', pad=20)
        
        # Save plot
        save_path = save_dir / f'performance_table{modality_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance table saved to: {save_path}")
    
    if len(input_modalities) <= 1 and len(output_modalities) <= 1:
        primary(metrics, save_dir, dataset=dataset)
    else:
        for input_modality in input_modalities:
            for output_modality in output_modalities:
                modality_name = f"_input_{input_modality}_output_{output_modality}"
                primary(metrics[input_modality][output_modality], save_dir, modality_name, dataset=dataset)


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
        self._starclass: List[Any] = []
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
        log_dir = getattr(trainer.logger, "log_dir")
        self._logger_dir = Path(log_dir)

        # Determine dataset from predict dataloader 0
        vloaders = trainer.predict_dataloaders
        if isinstance(vloaders, (list, tuple)):
            ds = getattr(vloaders[0], "dataset")
        else:
            ds = getattr(vloaders, "dataset")

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
        batch_idx: int,
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
        flux_pred = predictions['flux']
        flux_err_pred = predictions['flux_uncertainty']
        # flux_pred, flux_err_pred = self._extract_flux_predictions(predictions)
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
        if self._task == "lightcurves":
            starclass_batch = [a["starclass"] for a in actuals]

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
        if self._task == "lightcurves":
            self._starclass.extend(starclass_batch)

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
            out["starclass"] = np.array(self._starclass)
        elif self._task == "spectra":
            out["wavelengths"] = wavelengths_or_times
            out["sobject_ids"] = self._star_ids

        # Save one NPZ for convenience
        input_modalities = pl_module.predict_input_modalities
        output_modalities = pl_module.predict_output_modality
        if self.save_dir is not None:
            np.savez_compressed(self.save_dir / f"prediction_results_input_{'-'.join(input_modalities)}_output_{output_modalities}.npz", **out)

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

