import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, Union
import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns

# Import the necessary modules
from daep.data_util import to_device
from daep.daep import unimodaldaepclassifier, multimodaldaepclassifier
from torch.utils.data import DataLoader, Dataset

from daep.datasets.GALAHspectra_dataset import GALAHDatasetProcessed
from daep.datasets.TESSlightcurve_dataset import TESSDatasetProcessed
from daep.datasets.TESSGALAHspeclc_dataset import TESSGALAHDatasetProcessed

from daep.utils.general_utils import load_config, update_config
from daep.datasets.dataloaders import create_dataloader
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as L
from daep.LitWrapperAll import (
    daepClassifierUnimodal,
    daepClassifierMultimodal,
)
from pathlib import Path

from daep.utils.test_utils import (
    get_best_model,
    all_subsets,
    calculate_classification_metrics,
    print_classification_metrics,
    plot_confusion_matrix,
    plot_classification_metrics_summary,
    save_classification_results,
)

# Global device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Default configs directory (mirrors testing.py)
DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


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


def evaluate_classifier_model(model: L.LightningModule,
                              test_loader, test_dataset,
                              input_modalities: list,
                              output_modalities: list) -> Dict[str, np.ndarray]:
    """
    Evaluate the classifier model on test data and generate predictions.
    
    Parameters
    ----------
    model : L.LightningModule
        Trained classifier model
    test_loader : DataLoader
        DataLoader for test data
    test_dataset : Dataset
        Test dataset for getting actual labels
    input_modalities : list
        List of input modalities
    output_modalities : list
        List of output modalities (same as input for classification)
    
    Returns
    -------
    dict
        Dictionary containing predictions, ground truth, and metadata
    """
    def eval_classifier_for_modality(model, test_loader, test_dataset, input_modalities, output_modality):
        model.eval()
        model = model.to(device)
        base_model = getattr(model, "model", model).to(device)
        all_predictions = []
        all_ground_truth = []
        all_prediction_probs = []
        all_test_instance_idxs = []
        all_star_ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions", unit="batch"):
                model_input, targets = extract_targets_from_batch(batch, device)
                
                if model_input is None:
                    continue
                
                # Generate predictions
                if isinstance(base_model, unimodaldaepclassifier):
                    prediction_probs = base_model(model_input)
                elif isinstance(base_model, multimodaldaepclassifier):
                    alt_modalities_dict = {"spectra": ["spectra"], "lightcurves": ["photometry"], "both": ["spectra", "photometry"]}
                    condition_keys = alt_modalities_dict[input_modalities]
                    prediction_probs = base_model.predict(model_input, condition_keys=condition_keys)
                
                # Convert to numpy
                prediction_probs = prediction_probs.cpu().numpy()
                targets = targets.cpu().numpy()
                
                # Get predicted class (argmax of probabilities)
                predicted_classes = np.argmax(prediction_probs, axis=1)
                
                # Get ground truth class indices
                if len(targets.shape) > 1 and targets.shape[1] > 1:
                    # One-hot encoded targets
                    ground_truth_classes = np.argmax(targets, axis=1)
                else:
                    # Single class labels
                    ground_truth_classes = targets.flatten()
                
                # Ensure both are 1D arrays
                predicted_classes = predicted_classes.flatten()
                ground_truth_classes = ground_truth_classes.flatten()
                
                # Get test instance indices
                try:
                    test_instance_idx = batch['idx'].cpu().numpy()
                except KeyError:
                    if 'photometry' in batch:
                        test_instance_idx = batch['photometry']['lightcurve_idx'].cpu().numpy()
                    elif 'spectra' in batch:
                        test_instance_idx = batch['spectra']['spectra_idx'].cpu().numpy()
                    else:
                        raise ValueError(f"No 'idx' or 'lightcurve_idx' key in batch")
                
                # Get star IDs for each item in the batch
                star_ids_batch = []
                for i, idx in enumerate(test_instance_idx):
                    if hasattr(test_dataset, 'get_actual_spectrum'):
                        actual_test_instance = test_dataset.get_actual_spectrum(idx)
                    elif hasattr(test_dataset, 'get_actual_lightcurve'):
                        actual_test_instance = test_dataset.get_actual_lightcurve(idx)
                    elif hasattr(test_dataset, 'spectra_dataset') and hasattr(test_dataset, 'lightcurve_dataset'):
                        if output_modality == "spectra":
                            actual_test_instance = test_dataset.spectra_dataset.get_actual_spectrum(idx)
                        elif output_modality == "lightcurves":
                            actual_test_instance = test_dataset.lightcurve_dataset.get_actual_lightcurve(idx)
                    else:
                        raise ValueError("Test dataset not found")
                    star_ids_batch.append(actual_test_instance['ids'][2])  # sobject_id/TICID is in column 2
                
                all_test_instance_idxs.append(test_instance_idx)
                all_predictions.append(predicted_classes)
                all_ground_truth.append(ground_truth_classes)
                all_prediction_probs.append(prediction_probs)
                all_star_ids.extend(star_ids_batch)
        
        predictions = np.concatenate(all_predictions, axis=0)
        ground_truth = np.concatenate(all_ground_truth, axis=0)
        prediction_probs = np.concatenate(all_prediction_probs, axis=0)
        test_instance_idxs = np.concatenate(all_test_instance_idxs, axis=0)
        star_ids = all_star_ids
        
        results = {
            'predictions': predictions,
            'ground_truth': ground_truth,
            'prediction_probs': prediction_probs,
            'test_instance_idxs': test_instance_idxs,
        }
        
        if output_modality == "spectra":
            results['sobject_ids'] = star_ids
        elif output_modality == "lightcurves":
            results['ticids'] = star_ids
        
        return results

    all_results = {input_modality: {output_modality: None for output_modality in output_modalities} for input_modality in input_modalities}
    
    for input_modality in input_modalities:
        for output_modality in output_modalities:
            all_results[input_modality][output_modality] = eval_classifier_for_modality(
                model, test_loader, test_dataset, 
                input_modalities=input_modality,
                output_modality=output_modality
            )
    
    return all_results

def run_classification_tests(config_path: str,
                            plot_from_saved: bool = False,
                            save_results_only: bool = False,
                            **kwargs):
    """
    Main function to test the classifier model using configuration from file.
    
    Parameters
    ----------
    config_path : str
        Path to the (non-default) configuration YAML file. Must include testing.model_dir
    plot_from_saved : bool, optional
        Unused for classification; kept for API parity
    save_results_only : bool, default=False
        If True, only save metrics without generating plots
    **kwargs : dict
        Additional parameters to override config values (merged into config)
    """
    # Load additional config and hparams from model_dir to mirror testing.py
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    additional_config = load_config(str(config_path))

    model_dir = additional_config["testing"]["model_dir"]
    model_dir = Path(model_dir)
    if model_dir is None:
        raise ValueError("testing.model_dir must be provided in the config")

    # Load hparams from training run
    hparams_path = model_dir / "hparams.yaml"
    hparams = load_config(hparams_path)["config"]
    model_type = hparams["model_type"]
    if model_type != "classifier":
        raise ValueError("This entry point is for classifier models. For reconstruction, use testing.py")
    data_types = hparams["data_types"]
    data_names = hparams["data_names"]
    test_name = hparams["test_name"]

    # Merge with default classifier config, then hparams, then additional
    default_config_path = DEFAULT_CONFIGS_DIR / "config_classifier_default.yaml"
    default_config = load_config(default_config_path)
    config = update_config(default_config, hparams)
    config = update_config(config, additional_config)

    # Build test dataloader
    num_test_instances = config["testing"]["num_test_instances"]
    testing_loader = create_dataloader(config, data_types, data_names, train=False, subset_size=num_test_instances)
    # Unwrap subset dataset if it exists
    test_dataset = getattr(testing_loader.dataset, 'dataset', testing_loader.dataset)

    # Select correct Lightning model class
    if len(data_types) == 1:
        model_class = daepClassifierUnimodal
    else:
        model_class = daepClassifierMultimodal

    # Evaluate all checkpoints and pick best using Lightning test loop
    print(f"Testing all checkpoints in {model_dir} to determine best model")
    model = get_best_model(model_dir, testing_loader, model_class, use_val_loss=True)

    # Determine modality combinations
    input_modality_combos = all_subsets(data_types)
    output_modalities = data_types

    # Run evaluation
    print("Evaluating classifier model...")
    results = evaluate_classifier_model(model, testing_loader, test_dataset, input_modality_combos, output_modalities)
    
    # Metrics
    print("Calculating classification metrics...")
    metrics = calculate_classification_metrics(results, test_dataset,
                                               input_modalities=input_modality_combos,
                                               output_modalities=output_modalities)
    
    # Print metrics
    print_classification_metrics(metrics,
                                 input_modalities=input_modality_combos,
                                 output_modalities=output_modalities)
    
    # Prepare analysis directory and save results
    analysis_dir = model_dir / "analysis_results"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {analysis_dir}")
    save_classification_results(results, metrics, analysis_dir,
                                input_modalities=input_modality_combos,
                                output_modalities=output_modalities,
                                dataset=test_dataset)
    
    # Plots
    if not save_results_only:
        print("Creating classification plots...")
        plot_confusion_matrix(results, analysis_dir,
                              input_modalities=input_modality_combos,
                              output_modalities=output_modalities,
                              dataset=test_dataset)
        
        print("Creating metrics summary...")
        plot_classification_metrics_summary(metrics, analysis_dir,
                                            input_modalities=input_modality_combos,
                                            output_modalities=output_modalities)
        
        print(f"\nClassification testing complete! Results and plots saved in '{analysis_dir}' directory.")
    else:
        print(f"\nResults saved to '{analysis_dir}'. Run again without save_results_only to create plots.")


if __name__ == "__main__":
    import fire
    
    # Use fire to handle command line arguments
    fire.Fire(run_classification_tests) 