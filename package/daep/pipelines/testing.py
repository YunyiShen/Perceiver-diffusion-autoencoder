import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, Union
import os
import json
import re
from astropy.stats import mad_std

# Import the necessary modules
from daep.data_util import to_device, padding_collate_fun
from daep.daep import unimodaldaep, multimodaldaep
from torch.utils.data import DataLoader, Dataset

from daep.datasets.GALAHspectra_dataset import GALAHDatasetProcessedSubset
from daep.datasets.TESSlightcurve_dataset import TESSDatasetProcessedSubset
from daep.datasets.TESSGALAHspeclc_dataset import TESSGALAHDatasetProcessedSubset

from daep.utils.train_utils import load_and_update_config
from daep.utils.test_utils import (load_trained_model, auto_detect_model_path, extract_epoch_from_model_path,
                        create_analysis_directory, plot_results_from_saved, print_evaluation_metrics,
                        calculate_metrics, plot_example_spectra, plot_metrics_summary, save_results,
                        plot_example_lightcurves)

# Global device variable
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def setup_test_data_and_loader(config: Dict[str, Any], data_path: Path, 
                              test_name: str, batch_size: int, spectra_or_lightcurves: str) -> Tuple[Dataset, DataLoader]:
    """
    Set up test dataset and data loader.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    data_path : Path
        Path to data directory
    test_name : str
        Name of the test dataset
    batch_size : int
        Batch size for data loader
        
    Returns
    -------
    tuple
        (test_dataset, test_loader)
    """
    print("Loading test dataset...")
    if 'use_uncertainty' in config["model"]:
        use_uncertainty = config["model"]["use_uncertainty"]
    else:
        use_uncertainty = False
    
    if spectra_or_lightcurves == "spectra":
        
        test_data = GALAHDatasetProcessedSubset(
            num_spectra=config["testing"]["num_spectra"], 
            data_dir=data_path / 'spectra' / test_name, 
            train=False, 
            extract=False,
            use_uncertainty=use_uncertainty
        )
        collate_fn = padding_collate_fun(supply=['flux', 'wavelength', 'time'], mask_by="flux", multimodal=False)
    elif spectra_or_lightcurves == "lightcurves":
        test_data = TESSDatasetProcessedSubset(
            num_lightcurves=config["testing"]["num_lightcurves"], 
            data_dir=data_path / 'lightcurves' / test_name, 
            train=False, 
            extract=False,
            use_uncertainty=use_uncertainty
        )
        collate_fn = padding_collate_fun(supply=['flux', 'time'], mask_by="flux", multimodal=False)
    elif spectra_or_lightcurves == "both":
        from daep.datasets.TESSlightcurve_dataset import TESSDatasetProcessed
        from daep.datasets.GALAHspectra_dataset import GALAHDatasetProcessed
        spectra_data = GALAHDatasetProcessed(
            data_dir=data_path / 'spectra' / config["data"]["spectra_test_name"], 
            train=False, 
            extract=False,
            use_uncertainty=use_uncertainty
        )
        lightcurve_data = TESSDatasetProcessed(
            data_dir=data_path / 'lightcurves' / config["data"]["lightcurve_test_name"], 
            train=False, 
            extract=False,
            use_uncertainty=config["model"]["use_uncertainty"]
        )
        test_data = TESSGALAHDatasetProcessedSubset(
            num_samples=config["testing"]["num_test_instances"],
            lightcurve_dataset=lightcurve_data,
            spectra_dataset=spectra_data,
            use_uncertainty=use_uncertainty
        )
        collate_fn = padding_collate_fun(supply=['flux', 'wavelength', 'time'], mask_by="flux", multimodal=True)
    
    print("Creating test data loader...")
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        num_workers=config["data_processing"]["num_workers"],
        pin_memory=config["data_processing"]["pin_memory"]
    )
    
    return test_data, test_loader

def evaluate_model(model: Union[unimodaldaep, multimodaldaep], test_loader, test_dataset,
                   num_samples: int = 100, spectra_or_lightcurves: str = "spectra",
                   input_modalities: Optional[list] = None, use_uncertainty: bool = False) -> Dict[str, np.ndarray]:
    """
    Evaluate the model on test data and generate predictions with uncertainty estimates.
    
    Parameters
    ----------
    model : unimodaldaep
        Trained model
    test_loader : DataLoader
        DataLoader for test data
    test_dataset : Dataset
        Test dataset for getting actual spectra
    num_samples : int, default=100
        Number of Monte Carlo samples for uncertainty estimation
    spectra_or_lightcurves : str, optional
        "spectra" or "lightcurves" or "both" to specify the type of data to train on. Defaults to "spectra".
    Returns
    -------
    dict
        Dictionary containing predictions, uncertainties, and ground truth
    """
    model.eval()
    all_predictions = []
    all_ground_truth = []
    all_ground_truth_uncertainties = []
    all_uncertainties = []
    all_wavelengths_or_times = []
    all_test_instance_idxs = []
    all_star_ids = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions", unit="batch"):
            batch = to_device(batch)
            
            # Generate multiple samples for uncertainty estimation using diffusion sampling
            predictions = []
            if use_uncertainty:
                predictions_uncertainties = []
            for _ in range(num_samples):
                # Use the model's reconstruct method to get actual reconstructions
                # This samples from the diffusion process to generate reconstructions
                if isinstance(model, unimodaldaep):
                    reconstructed = model.reconstruct(batch)
                elif isinstance(model, multimodaldaep):
                    alt_modalities_dict = {"spectra": ["spectra"], "lightcurves": ["photometry"], "both": ["spectra", "photometry"]}
                    input_modalities = alt_modalities_dict[input_modalities]
                    out_keys = alt_modalities_dict[spectra_or_lightcurves]
                    
                    reconstructed = model.reconstruct(batch, condition_keys=input_modalities, out_keys=out_keys)
                
                # Handle different return types from the model
                if use_uncertainty:
                    if not isinstance(reconstructed, tuple):
                        raise ValueError(f"Unexpected model output type: {type(reconstructed)}")
                    pred_flux, pred_flux_uncertainty = reconstructed
                    pred_flux = pred_flux['flux']
                elif isinstance(reconstructed, dict) and 'flux' in reconstructed:
                    # Model returns a dictionary with 'flux' key
                    pred_flux = reconstructed['flux']
                elif isinstance(reconstructed, dict) and ('photometry' in reconstructed):
                    pred_flux = reconstructed['photometry']['flux']
                elif isinstance(reconstructed, dict) and ('spectra' in reconstructed):
                    pred_flux = reconstructed['spectra']['flux']
                elif isinstance(reconstructed, torch.Tensor):
                    # Model returns a tensor directly
                    pred_flux = reconstructed
                else:
                    raise ValueError(f"Unexpected model output type: {type(reconstructed)}")
                
                predictions.append(pred_flux.cpu().numpy())
                if use_uncertainty:
                    predictions_uncertainties.append(pred_flux_uncertainty.cpu().numpy())
            
            predictions = np.array(predictions)  # Shape: (num_samples, batch_size, seq_len)
            if use_uncertainty:
                predictions_uncertainties = np.array(predictions_uncertainties)  # Shape: (num_samples, batch_size, seq_len)
            
            # Calculate mean and std across samples
            pred_mean = np.mean(predictions, axis=0)
            if use_uncertainty:
                pred_uncertainty_mean = np.mean(predictions_uncertainties, axis=0)
            pred_std = np.std(predictions, axis=0)
            
            # Get ground truth - handle batch indexing properly
            try:
                test_instance_idx = batch['idx'].cpu().numpy()
            except KeyError:
                if 'photometry' in batch:
                    test_instance_idx = batch['photometry']['lightcurve_idx'].cpu().numpy()
                elif 'spectra' in batch:
                    test_instance_idx = batch['spectra']['spectra_idx'].cpu().numpy()
                else:
                    raise ValueError(f"No 'idx' or 'lightcurve_idx' key in batch")
            
            # Get actual spectra or lightcurves for each item in the batch
            ground_truth_batch = []
            ground_truth_uncertainties_batch = []
            wavelengths_or_times_batch = []
            star_ids_batch = []
            
            for i, idx in enumerate(test_instance_idx):
                if isinstance(test_dataset, GALAHDatasetProcessedSubset):
                    actual_test_instance = test_dataset.get_actual_spectrum(idx)
                    wavelengths_or_times_batch.append(actual_test_instance['wavelength'])
                elif isinstance(test_dataset, TESSDatasetProcessedSubset):
                    actual_test_instance = test_dataset.get_actual_lightcurve(idx)
                    wavelengths_or_times_batch.append(actual_test_instance['time'])
                elif isinstance(test_dataset, TESSGALAHDatasetProcessedSubset):
                    if spectra_or_lightcurves == "spectra":
                        actual_test_instance = test_dataset.spectra_dataset.get_actual_spectrum(idx)#.get_actual_data(test_dataset.pre_subset_idx_to_idx(idx))['spectrum']
                        wavelengths_or_times_batch.append(actual_test_instance['wavelength'])
                    elif spectra_or_lightcurves == "lightcurves":
                        actual_test_instance = test_dataset.lightcurve_dataset.get_actual_lightcurve(idx)#.get_actual_data(test_dataset.pre_subset_idx_to_idx(idx))['photometry']
                        wavelengths_or_times_batch.append(actual_test_instance['time'])
                ground_truth_batch.append(actual_test_instance['flux'])
                ground_truth_uncertainties_batch.append(actual_test_instance['flux_errs'])
                star_ids_batch.append(actual_test_instance['ids'][2])  # sobject_id/TICID is in column 2
            
            ground_truth = np.array(ground_truth_batch)
            ground_truth_uncertainties = np.array(ground_truth_uncertainties_batch)
            wavelengths_or_times = np.array(wavelengths_or_times_batch)
            
            # print(f"Prediction before conversion: {pred_mean}")
            # Convert prediction to actual flux (undo log10 and normalization)
            if spectra_or_lightcurves == "spectra":
                if isinstance(test_dataset, GALAHDatasetProcessedSubset):
                    pred_mean = test_dataset.unprocess_spectra(flux=pred_mean, idx=test_instance_idx)
                    if use_uncertainty:
                        pred_uncertainty_mean = pred_uncertainty_mean * np.repeat(test_dataset._fluxes_stds[test_instance_idx][:, None], pred_std.shape[1], axis=1)
                elif isinstance(test_dataset, TESSGALAHDatasetProcessedSubset):
                    pred_mean = test_dataset.spectra_dataset.unprocess_spectra(flux=pred_mean, idx=test_instance_idx)
                    if use_uncertainty:
                        pred_uncertainty_mean = pred_uncertainty_mean * np.repeat(test_dataset._fluxes_stds[test_instance_idx][:, None], pred_std.shape[1], axis=1)
            elif spectra_or_lightcurves == "lightcurves":
                if isinstance(test_dataset, TESSDatasetProcessedSubset):
                    unprocessed_predictions = test_dataset.unprocess_lightcurves(idx=test_instance_idx, time=wavelengths_or_times, flux=pred_mean)
                    if use_uncertainty:
                        pred_uncertainty_mean = pred_uncertainty_mean * np.repeat(test_dataset._fluxes_stds[test_instance_idx][:, None], pred_std.shape[1], axis=1)
                elif isinstance(test_dataset, TESSGALAHDatasetProcessedSubset):
                    unprocessed_predictions = test_dataset.lightcurve_dataset.unprocess_lightcurves(idx=test_instance_idx, time=wavelengths_or_times, flux=pred_mean)
                    if use_uncertainty:
                        pred_uncertainty_mean = pred_uncertainty_mean * np.repeat(test_dataset.lightcurve_dataset._fluxes_errs[test_instance_idx][:, None], pred_std.shape[1], axis=1)
                pred_mean = unprocessed_predictions['flux']
                wavelengths_or_times = unprocessed_predictions['time']
            
            if np.all(pred_std <= 1e-9):
                pred_std = np.zeros_like(pred_mean)
            else:
                # print(f"pred_std: {pred_std}")
                # print(f"test_dataset._fluxes_stds[test_instance_idx]: {test_dataset._fluxes_stds[test_instance_idx]}")
                # Duplicate and reshape _fluxes_stds to match pred_std's shape for correct broadcasting
                pred_std = pred_std * np.repeat(test_dataset._fluxes_stds[test_instance_idx][:, None], pred_std.shape[1], axis=1)
                # print(f"pred_std after: {pred_std}")
            # print(f"Prediction after conversion: {pred_mean}")
            
            all_test_instance_idxs.append(test_instance_idx)
            all_predictions.append(pred_mean)
            all_ground_truth.append(ground_truth)
            all_ground_truth_uncertainties.append(ground_truth_uncertainties)
            if use_uncertainty:
                all_uncertainties.append(pred_uncertainty_mean)
            else:
                all_uncertainties.append(pred_std)
            all_wavelengths_or_times.append(wavelengths_or_times)
            all_star_ids.extend(star_ids_batch)
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)
    ground_truth_uncertainties = np.concatenate(all_ground_truth_uncertainties, axis=0)
    uncertainties = np.concatenate(all_uncertainties, axis=0)
    wavelengths_or_times = np.concatenate(all_wavelengths_or_times, axis=0)
    test_instance_idxs = np.concatenate(all_test_instance_idxs, axis=0)
    star_ids = all_star_ids
    
    results = {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'ground_truth_uncertainties': ground_truth_uncertainties,
        'uncertainties': uncertainties,
        'test_instance_idxs': test_instance_idxs,
        }
    if spectra_or_lightcurves == "spectra":
        results['wavelengths'] = wavelengths_or_times
        results['sobject_ids'] = star_ids
    elif spectra_or_lightcurves == "lightcurves":
        results['times'] = wavelengths_or_times
        results['ticids'] = star_ids
    return results


def evaluate_model_multimodal(model: multimodaldaep, test_loader, test_dataset, num_samples: int = 100,
                              input_modalities: list = ["spectra", "lightcurves"],
                              output_modalities: list = ["spectra", "lightcurves"],
                              use_uncertainty: bool = False) -> Dict[str, np.ndarray]:
    """
    Evaluate the model on test data and generate predictions with uncertainty estimates.
    
    Parameters
    ----------
    model : multimodaldaep
        Trained model
    test_loader : DataLoader
        DataLoader for test data
    test_dataset : Dataset
        Test dataset for getting actual spectra
    num_samples : int, default=100
        Number of Monte Carlo samples for uncertainty estimation
        spectra_or_lightcurves : str, optional
            "spectra" or "lightcurves" or "both" to specify the type of data to train on. Defaults to "spectra".
    Returns
    -------
    dict
        Dictionary containing predictions, uncertainties, and ground truth
    """
    all_results = {input_modality: {output_modality: None for output_modality in output_modalities} for input_modality in input_modalities}
    
    for input_modality in input_modalities:
        for output_modality in output_modalities:
            print(f"Evaluating input modality: {input_modality} to output modality: {output_modality}")
            all_results[input_modality][output_modality] = evaluate_model(model,
                                                                          test_loader,
                                                                          test_dataset,
                                                                          num_samples,
                                                                          spectra_or_lightcurves=output_modality,
                                                                          input_modalities=input_modality,
                                                                          use_uncertainty=use_uncertainty)
    
    return all_results

def plot_examples_multimodal(results, test_dataset, num_examples, analysis_dir, input_modalities, output_modalities):
    """
    Plot example spectra and lightcurves for multimodal model.
    """
    for input_modality in input_modalities:
        for output_modality in output_modalities:
            if output_modality == "spectra":
                plot_example_spectra(results[input_modality][output_modality], test_dataset.spectra_dataset, num_examples, analysis_dir / f"input_{input_modality}_output_{output_modality}")
            elif output_modality == "lightcurves":
                plot_example_lightcurves(results[input_modality][output_modality], test_dataset.lightcurve_dataset, num_examples, analysis_dir / f"input_{input_modality}_output_{output_modality}")


def run_tests(config_path: str = "config.json", spectra_or_lightcurves: str = "spectra",
              checkpoint_path: Optional[str] = None, num_samples: Optional[int] = None,
              num_examples: Optional[int] = None, save_results_only: bool = False,
              results_dir: Optional[str] = None, use_saved_results: bool = False, **kwargs):
    """
    Main function to test the GALAHspectra model using configuration from file.
    
    Parameters
    ----------
    config_path : str, default="config_test.json"
        Path to the configuration JSON file
    spectra_or_lightcurves : str, optional
        "spectra" or "lightcurves" or "both" to specify the type of data to train on. Defaults to "spectra".
    checkpoint_path : str, optional
        Path to the trained model. If None, will auto-detect.
    num_samples : int, optional
        Number of Monte Carlo samples for uncertainty estimation
    num_examples : int, optional
        Number of example spectra to plot
    save_results_only : bool, default=False
        If True, only save results without plotting
    results_dir : str, optional
        Directory to load results from (for plotting only)
    use_saved_results : bool, default=False
        If True, load and plot existing results instead of running evaluation
    **kwargs : dict
        Additional parameters to override config values
    """
    # Validate input parameter
    if spectra_or_lightcurves not in ["spectra", "lightcurves", "both"]:
        raise ValueError(f"spectra_or_lightcurves must be 'spectra' or 'lightcurve', got '{spectra_or_lightcurves}'")
    
    if spectra_or_lightcurves == "both":
        input_modalities = ["spectra", "lightcurves", "both"]
        output_modalities = ["spectra", "lightcurves"]
    else:
        input_modalities = [spectra_or_lightcurves]
        output_modalities = [spectra_or_lightcurves]
    # Load and update configuration
    config = load_and_update_config(config_path, **kwargs)
    
    # Extract paths from config
    data_path = Path(config["data"]["data_path"])
    models_path = Path(config["data"]["models_path"])
    test_name = config["data"]["test_name"]
    
    # Auto-detect model path if not provided
    if config["testing"]["use_checkpoint_path"] and config["testing"]["checkpoint_path"] is not None:
        checkpoint_path = Path(config["testing"]["checkpoint_path"])
    else:
        if spectra_or_lightcurves == "both":
            models_subdir = "speclc"
        else:
            models_subdir = spectra_or_lightcurves
        checkpoint_path = auto_detect_model_path(config, models_path / models_subdir, test_name)
    print(f"Loading model from: {checkpoint_path}")
    
    # Extract epoch number from model path
    epoch_number = extract_epoch_from_model_path(checkpoint_path)
    
    # Create analysis directory
    if len(checkpoint_path.name) > 40:
        if spectra_or_lightcurves == "spectra":
            data_name = "GALAHspectra"
        elif spectra_or_lightcurves == "lightcurves":
            data_name = "TESSlightcurve"
        elif spectra_or_lightcurves == "both":
            data_name = "TESSGALAHspeclc"
        analysis_dir = create_analysis_directory(config, models_path / spectra_or_lightcurves, test_name, epoch_number, data_name=data_name)
    else:
        analysis_dir = checkpoint_path.parent.parent / "analysis_results" / f"epoch_{epoch_number}"
    print(f"Saving results to analysis directory: {analysis_dir}")
    
    # Use parameters from config with command line overrides
    batch_size = config["testing"]["batch_size"]
    num_samples = num_samples or config["testing"]["num_samples"]
    num_examples = num_examples or config["testing"]["num_examples"]
    
    # If use_saved_results is True, load and plot existing results
    if use_saved_results:
        if results_dir is None:
            results_dir = analysis_dir / 'saved_results'
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        print("Loading test dataset for plotting...")
        if spectra_or_lightcurves == "spectra":
            test_data = GALAHDatasetProcessedSubset(
                num_spectra=config["testing"]["num_spectra"], 
                data_dir=data_path / 'spectra' / test_name, 
                train=False, 
                extract=False
            )
        elif spectra_or_lightcurves == "lightcurves":
            test_data = TESSDatasetProcessedSubset(
                num_lightcurves=config["testing"]["num_lightcurves"], 
                data_dir=data_path / 'lightcurves' / test_name, 
                train=False, 
                extract=False
            )
        plot_results_from_saved(results_dir, test_data, num_examples=num_examples, save_dir=analysis_dir,
                                spectra_or_lightcurves=spectra_or_lightcurves, input_modalities=input_modalities,
                                output_modalities=output_modalities)
        return
    
    print(f"Configuration loaded from: {config_path}")
    print(f"Test dataset: {test_name}")
    print(f"Batch size: {batch_size}")
    print(f"Number of samples: {num_samples}")
    print(f"Number of examples: {num_examples}")
    
    # Set up test data and loader
    # if spectra_or_lightcurves == "both":
    #     test_data, test_loader = setup_test_data_and_loader_multimodal(config, data_path, test_name, batch_size, modalities=modalities)
    # else:
    test_data, test_loader = setup_test_data_and_loader(config, data_path, test_name, batch_size, spectra_or_lightcurves)
    
    print(f"Loading trained model from: {checkpoint_path}")
    model = load_trained_model(Path(checkpoint_path), device, config, spectra_or_lightcurves)
    
    print("Evaluating model...")
    if spectra_or_lightcurves == "both":
        results = evaluate_model_multimodal(model, test_loader, test_data, num_samples, input_modalities=input_modalities, output_modalities=output_modalities, use_uncertainty=config["model"]["use_uncertainty"])
    else:
        results = evaluate_model(model, test_loader, test_data, num_samples, spectra_or_lightcurves, use_uncertainty=config["model"]["use_uncertainty"])
    
    print("Calculating metrics...")
    metrics = calculate_metrics(results, input_modalities=input_modalities, output_modalities=output_modalities)
    
    # Print metrics
    print_evaluation_metrics(metrics, input_modalities=input_modalities, output_modalities=output_modalities)
    
    print(f"\nSaving results to: {analysis_dir / 'saved_results'}")
    save_results(results, metrics, analysis_dir / 'saved_results', spectra_or_lightcurves, input_modalities=input_modalities, output_modalities=output_modalities)
    
    # Create plots if not save_results_only
    if not save_results_only:
        print("Creating example plots...")
        if spectra_or_lightcurves == "spectra":
            plot_example_spectra(results, test_data, num_examples, analysis_dir)
        elif spectra_or_lightcurves == "lightcurves":
            plot_example_lightcurves(results, test_data, num_examples, analysis_dir)
        elif spectra_or_lightcurves == "both":
            plot_examples_multimodal(results, test_data, num_examples, analysis_dir, input_modalities, output_modalities)
        
        print("Creating metrics summary...")
        plot_metrics_summary(metrics, analysis_dir, input_modalities=input_modalities, output_modalities=output_modalities)
        
        print(f"\nTesting complete! Results and plots saved in '{analysis_dir}' directory.")
    else:
        print(f"\nResults saved to '{analysis_dir}'. Run with --results_dir to create plots.")


if __name__ == "__main__":
    import fire
    
    # Use fire to handle command line arguments
    fire.Fire(run_tests)
