import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

# Import the necessary modules
from daep.data_util import to_device
from daep.utils.general_utils import load_config, update_config
from daep.datasets.dataloaders import create_dataloader
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
    plot_class_averages,
    plot_bottleneck_umap,
)
from daep.utils.test_callbacks import ClassifierPredictionWriter, BottleneckRepresentationWriter

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
                              test_loader,
                              input_modalities: list,
                              output_modalities: list) -> Dict[str, np.ndarray]:
    """
    Evaluate the classifier model on test data for each input/output modality pair.

    Parameters
    ----------
    model : L.LightningModule
        Trained classifier Lightning module (unimodal or multimodal).
    test_loader : DataLoader
        Prediction dataloader.
    input_modalities : list of str | list of list[str] | list of tuple[str, ...]
        Conditioning modalities per run (e.g., ["spectra", "lightcurves", "both"]) or
        combinations returned by `all_subsets`.
    output_modalities : list[str]
        Output modalities to evaluate. For classification this is used for naming
        star-id fields in the results.

    Returns
    -------
    dict
        Nested results keyed by input and output modality, containing:
        - predictions: np.ndarray of predicted class indices
        - ground_truth: np.ndarray of ground truth class indices
        - prediction_probs: np.ndarray of class probabilities
        - test_instance_idxs: np.ndarray of dataset indices
        - sobject_ids or ticids: list of star IDs depending on output modality
    """

    results: Dict[str, Dict[str, Dict[str, Any]]] = {
        in_mod: {out_mod: None for out_mod in output_modalities} for in_mod in input_modalities
    }

    for in_mod in input_modalities:
        for out_mod in output_modalities:
            writer = ClassifierPredictionWriter(input_modalities=in_mod, output_modality=out_mod, write_interval="batch")
            trainer = L.Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False, callbacks=[writer])
            trainer.predict(model=model, dataloaders=test_loader)
            results[in_mod][out_mod] = writer.to_results(output_modality=out_mod)

    return results

# TODO: Fix this to not repeat predictions
def evaluate_bottleneck_representations(model: L.LightningModule,
                              test_loader,
                              input_modalities: list,
                              output_modalities: list) -> Dict[str, np.ndarray]:
    """
    Evaluate the classifier model on test data for each input/output modality pair.

    Parameters
    ----------
    model : L.LightningModule
        Trained classifier Lightning module (unimodal or multimodal).
    test_loader : DataLoader
        Prediction dataloader.
    input_modalities : list of str | list of list[str] | list of tuple[str, ...]
        Conditioning modalities per run (e.g., ["spectra", "lightcurves", "both"]) or
        combinations returned by `all_subsets`.
    output_modalities : list[str]
        Output modalities to evaluate. For classification this is used for naming
        star-id fields in the results.

    Returns
    -------
    dict
        Nested results keyed by input and output modality, containing:
        - predictions: np.ndarray of predicted class indices
        - ground_truth: np.ndarray of ground truth class indices
        - prediction_probs: np.ndarray of class probabilities
        - test_instance_idxs: np.ndarray of dataset indices
        - sobject_ids or ticids: list of star IDs depending on output modality
    """

    bottleneck_results: Dict[str, Dict[str, Dict[str, Any]]] = {
        in_mod: {out_mod: None for out_mod in output_modalities} for in_mod in input_modalities
    }
    bottleneck_avg_results: Dict[str, Dict[str, Dict[str, Any]]] = {
        in_mod: {out_mod: None for out_mod in output_modalities} for in_mod in input_modalities
    }

    for in_mod in input_modalities:
        for out_mod in output_modalities:
            bottleneck_writer = BottleneckRepresentationWriter(input_modalities=in_mod, output_modality=out_mod, write_interval="batch")
            trainer = L.Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False, callbacks=[bottleneck_writer])
            trainer.predict(model=model, dataloaders=test_loader)
            bottleneck_avg_results[in_mod][out_mod] = bottleneck_writer.get_class_averages()
            bottleneck_results[in_mod][out_mod] = bottleneck_writer.get_bottlenecks()

    return bottleneck_avg_results, bottleneck_results

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
    results = evaluate_classifier_model(model, testing_loader, input_modality_combos, output_modalities)
    
    bottleneck_avg_results, bottleneck_results = evaluate_bottleneck_representations(model, testing_loader, input_modality_combos, output_modalities)
    
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
    # TODO: Clean up by moving input/output modality handling from utils to here
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
        
        print("Creating bottleneck representation plots...")
        plot_class_averages(bottleneck_avg_results, analysis_dir,
                            input_modalities=input_modality_combos,
                            output_modalities=output_modalities)
        
        print("Creating bottleneck UMAP plots...")
        plot_bottleneck_umap(bottleneck_results, analysis_dir,
                            input_modalities=input_modality_combos,
                            output_modalities=output_modalities)
        
        print(f"\nClassification testing complete! Results and plots saved in '{analysis_dir}' directory.")
    else:
        print(f"\nResults saved to '{analysis_dir}'. Run again without save_results_only to create plots.")


if __name__ == "__main__":
    import fire
    
    # Use fire to handle command line arguments
    fire.Fire(run_classification_tests) 