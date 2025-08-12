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

# Import shared functions from testing.py
from daep.pipelines.testing import (
    setup_test_data_and_loader, 
    run_tests as run_tests_base,
    device
)

# Import the necessary modules
from daep.data_util import to_device
from daep.daep import unimodaldaepclassifier, multimodaldaepclassifier
from torch.utils.data import DataLoader, Dataset

from daep.datasets.GALAHspectra_dataset import GALAHDatasetProcessedSubset
from daep.datasets.TESSlightcurve_dataset import TESSDatasetProcessedSubset
from daep.datasets.TESSGALAHspeclc_dataset import TESSGALAHDatasetProcessedSubset

from daep.utils.train_utils import load_and_update_config
from daep.utils.test_utils import (load_trained_model, auto_detect_model_path, extract_epoch_from_model_path,
                        create_analysis_directory)


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


def evaluate_classifier_model(model: Union[unimodaldaepclassifier, multimodaldaepclassifier], test_loader, test_dataset,
                             spectra_or_lightcurves: str = "spectra",
                             input_modalities: Optional[list] = None) -> Dict[str, np.ndarray]:
    """
    Evaluate the classifier model on test data and generate predictions.
    
    Parameters
    ----------
    model : unimodaldaepclassifier or multimodaldaepclassifier
        Trained classifier model
    test_loader : DataLoader
        DataLoader for test data
    test_dataset : Dataset
        Test dataset for getting actual labels
    spectra_or_lightcurves : str, optional
        "spectra" or "lightcurves" or "both" to specify the type of data. Defaults to "spectra".
    input_modalities : list, optional
        List of input modalities for multimodal models
        
    Returns
    -------
    dict
        Dictionary containing predictions, ground truth, and metadata
    """
    model.eval()
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
            if isinstance(model, unimodaldaepclassifier):
                prediction_probs = model(model_input)
            elif isinstance(model, multimodaldaepclassifier):
                alt_modalities_dict = {"spectra": ["spectra"], "lightcurves": ["photometry"], "both": ["spectra", "photometry"]}
                input_modalities = alt_modalities_dict[input_modalities]
                prediction_probs = model(model_input, condition_keys=input_modalities)
            
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
                if isinstance(test_dataset, GALAHDatasetProcessedSubset):
                    actual_test_instance = test_dataset.get_actual_spectrum(idx)
                elif isinstance(test_dataset, TESSDatasetProcessedSubset):
                    actual_test_instance = test_dataset.get_actual_lightcurve(idx)
                elif isinstance(test_dataset, TESSGALAHDatasetProcessedSubset):
                    if spectra_or_lightcurves == "spectra":
                        actual_test_instance = test_dataset.spectra_dataset.get_actual_spectrum(idx)
                    elif spectra_or_lightcurves == "lightcurves":
                        actual_test_instance = test_dataset.lightcurve_dataset.get_actual_lightcurve(idx)
                
                star_ids_batch.append(actual_test_instance['ids'][2])  # sobject_id/TICID is in column 2
            
            all_test_instance_idxs.append(test_instance_idx)
            all_predictions.append(predicted_classes)
            all_ground_truth.append(ground_truth_classes)
            all_prediction_probs.append(prediction_probs)
            all_star_ids.extend(star_ids_batch)
    
    # Concatenate all batches
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
    
    if spectra_or_lightcurves == "spectra":
        results['sobject_ids'] = star_ids
    elif spectra_or_lightcurves == "lightcurves":
        results['ticids'] = star_ids
    
    return results


def evaluate_classifier_model_multimodal(model: multimodaldaepclassifier, test_loader, test_dataset,
                                        input_modalities: list = ["spectra", "lightcurves"],
                                        output_modalities: list = ["spectra", "lightcurves"]) -> Dict[str, np.ndarray]:
    """
    Evaluate the multimodal classifier model on test data and generate predictions.
    
    Parameters
    ----------
    model : multimodaldaepclassifier
        Trained multimodal classifier model
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
        Dictionary containing predictions, ground truth, and metadata for each modality combination
    """
    all_results = {input_modality: {output_modality: None for output_modality in output_modalities} for input_modality in input_modalities}
    
    for input_modality in input_modalities:
        for output_modality in output_modalities:
            all_results[input_modality][output_modality] = evaluate_classifier_model(
                model, test_loader, test_dataset, 
                spectra_or_lightcurves=output_modality,
                input_modalities=input_modality
            )
    
    return all_results


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
            'confusion_matrix': cm,
            'class_metrics': class_metrics,
            'total_metrics': total_metrics,
            'num_classes': num_classes,
            'class_names': class_names
        }
    
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
    
    for input_modality in input_modalities:
        for output_modality in output_modalities:
            print(f"=== Input: {input_modality} to Output: {output_modality} ===")
            primary(metrics[input_modality][output_modality])


def plot_confusion_matrix(results: Dict[str, np.ndarray], save_dir: Path, 
                         input_modalities: Optional[list] = None, 
                         output_modalities: Optional[list] = None,
                         dataset: Optional[Dataset] = None):
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
    def primary(results, save_dir, modality_name="", dataset=None):
        predictions = results['predictions']
        ground_truth = results['ground_truth']
        
        # Get class names if available
        if dataset is not None and hasattr(dataset, 'starclass_name_to_int'):
            class_names = list(dataset.starclass_name_to_int.keys())
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
        plt.title(f'Confusion Matrix{modality_name}')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        save_path = save_dir / f'confusion_matrix{modality_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {save_path}")
    
    for input_modality in input_modalities:
        for output_modality in output_modalities:
            modality_name = f"_input_{input_modality}_output_{output_modality}"
            primary(results[input_modality][output_modality], save_dir, modality_name, dataset=dataset)


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
    
    for input_modality in input_modalities:
        for output_modality in output_modalities:
            modality_name = f"_input_{input_modality}_output_{output_modality}"
            primary(metrics[input_modality][output_modality], save_dir, modality_name, dataset=dataset)


def save_classification_results(results: Dict[str, np.ndarray], metrics: Dict[str, float], 
                               save_dir: Path, spectra_or_lightcurves: str,
                               input_modalities: Optional[list] = None, 
                               output_modalities: Optional[list] = None,
                               dataset: Optional[Dataset] = None):
    """
    Save classification results and metrics to files.
    
    Parameters
    ----------
    results : dict
        Dictionary containing predictions and ground truth
    metrics : dict
        Dictionary containing classification metrics
    save_dir : Path
        Directory to save results
    spectra_or_lightcurves : str
        Type of data used for classification
    input_modalities : list, optional
        List of input modalities for multimodal models
    output_modalities : list, optional
        List of output modalities for multimodal models
    dataset : Dataset, optional
        Dataset object to get class names from
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get labels for classification report
    if dataset is not None and hasattr(dataset, 'starclass_name_to_int'):
        labels = list(range(len(dataset.starclass_name_to_int)))
    else:
        # Fallback: get unique classes from predictions and ground truth
        predictions = results['predictions'] if len(input_modalities) <= 1 else results[input_modalities[0]][output_modalities[0]]['predictions']
        ground_truth = results['ground_truth'] if len(input_modalities) <= 1 else results[input_modalities[0]][output_modalities[0]]['ground_truth']
        labels = np.unique(np.concatenate([predictions, ground_truth]))
    
    if len(input_modalities) <= 1 and len(output_modalities) <= 1:
        # Save all results and confusion matrix in a single .npz file as a dictionary
        results_to_save = {
            'predictions': results['predictions'],
            'ground_truth': results['ground_truth'],
            'prediction_probs': results['prediction_probs'],
            'test_instance_idxs': results['test_instance_idxs'],
            'confusion_matrix': metrics['confusion_matrix']
        }
        np.savez(save_dir / 'classification_results.npz', **results_to_save)
        # Added: All arrays are now saved in a single .npz file for easier loading and management.
        
        # Save star IDs as .txt files for easier inspection
        if spectra_or_lightcurves == "spectra":
            np.savetxt(save_dir / 'sobject_ids.txt', results['sobject_ids'], fmt='%s')
        elif spectra_or_lightcurves == "lightcurves":
            np.savetxt(save_dir / 'ticids.txt', results['ticids'], fmt='%s')
        
        # Save metrics
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save detailed classification report with class names if available
        if dataset is not None and hasattr(dataset, 'starclass_name_to_int'):
            target_names = list(dataset.starclass_name_to_int.keys())
            report = classification_report(results['ground_truth'], results['predictions'], 
                                         target_names=target_names, output_dict=True, zero_division=0, labels=labels)
        else:
            report = classification_report(results['ground_truth'], results['predictions'], 
                                         output_dict=True, zero_division=0, labels=labels)
        with open(save_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        # Save metrics in a more readable format
        with open(save_dir / 'performance_table.txt', 'w') as f:
            f.write("Classification Performance Table\n")
            f.write("=" * 50 + "\n\n")
            
            # Write total metrics
            total = metrics['total_metrics']
            f.write(f"Total Metrics:\n")
            f.write(f"Accuracy: {total['accuracy']:.3f}\n")
            f.write(f"Recall: {total['recall']:.3f}\n")
            f.write(f"Precision: {total['precision']:.3f}\n")
            f.write(f"F1: {total['f1']:.3f}\n\n")
            
            # Write per-class metrics in table format
            f.write("Per-Class Performance:\n")
            f.write("Class\t\tAccuracy\tRecall\t\tPrecision\tF1\t\tSupport\n")
            f.write("-" * 80 + "\n")
            
            class_metrics = metrics['class_metrics']
            for class_name, class_data in class_metrics.items():
                f.write(f"{class_name}\t\t{class_data['accuracy']:.3f}\t\t{class_data['recall']:.3f}\t\t"
                       f"{class_data['precision']:.3f}\t\t{class_data['f1']:.3f}\t\t{class_data['support']}\n")
    
        print(f"Classification results saved to: {save_dir}")


def run_classification_tests(config_path: str = "config.json", spectra_or_lightcurves: str = "spectra",
                           checkpoint_path: Optional[str] = None, save_results_only: bool = False,
                           results_dir: Optional[str] = None, use_saved_results: bool = False, **kwargs):
    """
    Main function to test the classifier model using configuration from file.
    
    Parameters
    ----------
    config_path : str, default="config.json"
        Path to the configuration JSON file
    spectra_or_lightcurves : str, optional
        "spectra" or "lightcurves" or "both" to specify the type of data to test on. Defaults to "spectra".
    checkpoint_path : str, optional
        Path to the trained model. If None, will auto-detect.
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
        raise ValueError(f"spectra_or_lightcurves must be 'spectra', 'lightcurves', or 'both', got '{spectra_or_lightcurves}'")
    
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
            models_subdir = "speclc_classifier"
        else:
            models_subdir = f"{spectra_or_lightcurves}_classifier"
        checkpoint_path = auto_detect_model_path(config, models_path / models_subdir, test_name)
    print(f"Loading model from: {checkpoint_path}")
    
    # Extract epoch number from model path
    epoch_number = extract_epoch_from_model_path(checkpoint_path)
    
    # Create analysis directory
    if len(checkpoint_path.name) > 40:
        if spectra_or_lightcurves == "spectra":
            data_name = "GALAHspectra_classifier"
        elif spectra_or_lightcurves == "lightcurves":
            data_name = "TESSlightcurve_classifier"
        elif spectra_or_lightcurves == "both":
            data_name = "TESSGALAHspeclc_classifier"
        analysis_dir = create_analysis_directory(config, models_path / f"{spectra_or_lightcurves}_classifier", test_name, epoch_number, data_name=data_name)
    else:
        analysis_dir = checkpoint_path.parent.parent / "analysis_results" / f"epoch_{epoch_number}"
    print(f"Saving results to analysis directory: {analysis_dir}")
    
    # Use parameters from config with command line overrides
    batch_size = config["testing"]["batch_size"]
    
    # If use_saved_results is True, load and plot existing results
    if use_saved_results:
        if results_dir is None:
            results_dir = analysis_dir / 'saved_results'
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        print("Loading saved classification results for plotting...")
        # This would need to be implemented based on the saved format
        return
    
    print(f"Configuration loaded from: {config_path}")
    print(f"Test dataset: {test_name}")
    print(f"Batch size: {batch_size}")
    
    # Set up test data and loader
    test_data, test_loader = setup_test_data_and_loader(config, data_path, test_name, batch_size, spectra_or_lightcurves)
    
    print(f"Loading trained classifier model from: {checkpoint_path}")
    model = load_trained_model(Path(checkpoint_path), device, config, spectra_or_lightcurves)
    
    print("Evaluating classifier model...")
    if spectra_or_lightcurves == "both":
        results = evaluate_classifier_model_multimodal(model, test_loader, test_data, 
                                                     input_modalities=input_modalities, 
                                                     output_modalities=output_modalities)
    else:
        results = evaluate_classifier_model(model, test_loader, test_data, spectra_or_lightcurves)
    
    print("Calculating classification metrics...")
    metrics = calculate_classification_metrics(results, test_data, input_modalities=input_modalities, output_modalities=output_modalities)
    
    # Print metrics
    print_classification_metrics(metrics, input_modalities=input_modalities, output_modalities=output_modalities)
    
    print(f"\nSaving results to: {analysis_dir / 'saved_results'}")
    save_classification_results(results, metrics, analysis_dir / 'saved_results', 
                               spectra_or_lightcurves, input_modalities=input_modalities, output_modalities=output_modalities, dataset=test_data)
    
    # Create plots if not save_results_only
    if not save_results_only:
        print("Creating classification plots...")
        plot_confusion_matrix(results, analysis_dir, input_modalities=input_modalities, output_modalities=output_modalities, dataset=test_data)
        
        print("Creating metrics summary...")
        plot_classification_metrics_summary(metrics, analysis_dir, input_modalities=input_modalities, output_modalities=output_modalities, dataset=test_data)
        
        print(f"\nClassification testing complete! Results and plots saved in '{analysis_dir}' directory.")
    else:
        print(f"\nResults saved to '{analysis_dir}'. Run with --results_dir to create plots.")


if __name__ == "__main__":
    import fire
    
    # Use fire to handle command line arguments
    fire.Fire(run_classification_tests) 