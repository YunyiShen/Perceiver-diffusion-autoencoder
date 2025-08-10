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

from daep.utils.general_utils import load_and_update_config
from daep.utils.test_utils import (load_trained_model, auto_detect_model_path, extract_epoch_from_model_path,
                        create_analysis_directory, plot_results_from_saved, print_evaluation_metrics,
                        calculate_metrics, plot_example_spectra, plot_metrics_summary, save_results,
                        plot_example_lightcurves)
from daep.LitWrapperAll import daepClassifierUnimodal
import pytorch_lightning as L
from daep.datasets.dataloaders import create_dataloader

# Global device variable
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def make_confusion_matrix(model, class_names, device, outputdir):
    metric = model.conf_matrix.to(device)
    fig_, ax_ = metric.plot(labels=class_names)
    plt.savefig(outputdir / 'confusion_matrix.png')
    plt.close()

def test_save_best_model(top_k_checkpoints, test_loader):
    best_acc = 0.0
    best_loss = 0.0
    best_trainer = None
    best_model = None
    
    # Test each of the top 5 models
    # print('here',top_k_checkpoints)
    for checkpoint_path in top_k_checkpoints:
        # print('here')
        model = daepClassifierUnimodal.load_from_checkpoint(checkpoint_path)
        trainer = L.Trainer()
        test = trainer.test(model, test_loader)
        test_acc = test[0]['test_acc_epoch']
        test_loss = test[0]['test_loss_epoch']
        if (test_acc > best_acc) or (test_acc == best_acc and test_loss < best_loss):
            best_acc = test_acc
            best_loss = test_loss
            best_trainer = trainer
            best_model = model
    return best_trainer, best_model, best_acc, best_loss


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
    
    test_loader = create_dataloader(config, spectra_or_lightcurves, train=False)
    checkpoint_callback = L.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=4, mode='max', filename='{epoch:02d}-{val_acc:.2f}') # type: ignore

    # test saved model
    best_trainer, best_model, best_test_acc, best_test_loss = test_save_best_model(checkpoint_callback.best_k_models, test_loader)
    print('Test Accuracy:', best_test_acc)
    print('Test Loss:', best_test_loss)

    # save best model
    best_trainer.save_checkpoint(config['Training']['output_dir'] / 'LCC_test_acc_' + str(rounded_test_acc) + '.ckpt') # type: ignore

    # make confusion matrix
    print('Making confusion matrix')
    make_confusion_matrix(best_model, config['Model']['class_names'], config['Training']['accelerator'], config['Training']['output_dir'])
    
    # if config['Training']['plot_misclassified']:
    #     # plot misclassified light curves
    #     print('plotting misclassified curves') 
    #     plot_misclassified(best_model, config['Training']['output_dir'], config['Model']['class_names'])
    # print('finished')


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
