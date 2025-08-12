from pathlib import Path
from typing import Dict, Optional
from itertools import chain, combinations

import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.loggers import CSVLogger

from daep.LitWrapperAll import (
    daepReconstructorUnimodal,
    daepReconstructorMultimodal,
    daepClassifierUnimodal,
    daepClassifierMultimodal,
)
from daep.datasets.dataloaders import create_dataloader
from daep.utils.general_utils import load_config, update_config
from daep.utils.test_callbacks import UnprocessPredictionWriter
from daep.utils.test_utils import (
    calculate_metrics,
    plot_results_from_saved,
    plot_results_from_scratch,
    plot_metrics_summary,
    save_results,
    get_best_model,
    all_subsets,
)

# Global device
device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

def get_predictions(model, model_dir, testing_loader, input_modalities, output_modalities) -> Dict[str, np.ndarray]:
    """
    Use UnprocessPredictionWriter to run Lightning predict and assemble results.
    """
    
    def get_predictions_for_modality(input_modality_combo, output_modality):
        model.set_prediction_modalities(input_modality_combo, output_modality)
        writer = UnprocessPredictionWriter(write_interval="batch")
        logger = CSVLogger(model_dir.parent.parent, name=model_dir.parent.name, version=(model_dir.name + '/analysis_results'))
        # trainer = L.Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False, callbacks=[writer])
        trainer = L.Trainer(logger=logger, callbacks=[writer])
        trainer.predict(model=model, dataloaders=testing_loader)

        predictions = np.concatenate(writer._predictions, axis=0)
        ground_truth = np.concatenate(writer._ground_truth, axis=0)
        ground_truth_uncertainties = np.concatenate(writer._ground_truth_uncertainties, axis=0)
        uncertainties = np.concatenate(writer._uncertainties, axis=0)
        test_instance_idxs = np.concatenate(writer._indices, axis=0)

        out: Dict[str, np.ndarray] = {
            "predictions": predictions,
            "ground_truth": ground_truth,
            "ground_truth_uncertainties": ground_truth_uncertainties,
            "uncertainties": uncertainties,
            "test_instance_idxs": test_instance_idxs,
        }
        if writer._task == "lightcurves":
            out["times"] = np.concatenate(writer._wavelengths_or_times, axis=0)
            out["ticids"] = writer._star_ids
            out["starclass"] = np.asarray(writer._starclass)
        elif writer._task == "spectra":
            out["wavelengths"] = np.concatenate(writer._wavelengths_or_times, axis=0)
            out["sobject_ids"] = writer._star_ids
        else:
            raise ValueError("Unknown task type detected during prediction")

        return out

    results = {input_modality: {output_modality: None for output_modality in output_modalities} for input_modality in input_modalities}
    for input_modality in input_modalities:
        for output_modality in output_modalities:
            print(f"=== Getting predictions for input modalities: {input_modality} to output modality: {output_modality} ===")
            results[input_modality][output_modality] = get_predictions_for_modality(input_modality, output_modality)
    return results

def run_tests(
    config_path: str,
    plot_from_saved: bool = False,
):
    """
    Test a trained DAEP model using the same conventions as training.py (config, dataloader, and model layout).
    
    Parameters
    ----------
    config_path : str
        Path to the configuration YAML file (same schema as training).

    Notes
    -----
    - Uses the same load/merge logic and directory conventions as training.py.
    - Auto-discovers the latest checkpoints under models_path/test_name/model_name/<version>/checkpoints.
    """    
    # Load configuration
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    additional_config = load_config(str(config_path))
    
    # Get hyperparameters from model directory
    model_dir = additional_config["testing"]["model_dir"]
    model_dir = Path(model_dir)
    if model_dir is None:
        raise ValueError("model_dir must be provided in the non-default config")
    hparams_path = model_dir / "hparams.yaml"
    hparams = load_config(hparams_path)['config']
    model_type = hparams["model_type"]
    data_types = hparams["data_types"]
    data_names = hparams["data_names"]
    test_name = hparams["test_name"]
    
    # Get model class based on hyperparameters
    if model_type == "reconstructor":
        if len(data_types) == 1:
            model_class = daepReconstructorUnimodal
        else:
            model_class = daepReconstructorMultimodal
    elif model_type == "classifier":
        if len(data_types) == 1:
            model_class = daepClassifierUnimodal
        else:
            model_class = daepClassifierMultimodal
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # Update default config with hyperparameters and the non-default config
    if model_type == "reconstructor":
        default_config_path = DEFAULT_CONFIGS_DIR / "config_reconstruction_default.yaml"
    elif model_type == "classifier":
        raise NotImplementedError("Use testing_classifier.py for classifier models")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    default_config = load_config(default_config_path)
    config = update_config(default_config, hparams)
    config = update_config(config, additional_config)
    
    # Create testing dataloader
    num_test_instances = config['testing']['num_test_instances']
    testing_loader = create_dataloader(config, data_types, data_names, train=False, subset_size=num_test_instances)
    
    # Load best model
    print(f"Testing all checkpoints in {model_dir} to determine best model")
    model = get_best_model(model_dir, testing_loader, model_class, use_val_loss=True)
    
    all_input_modality_combos = all_subsets(data_types)
    output_modalities = data_types
    
    results = get_predictions(model, model_dir, testing_loader, all_input_modality_combos, output_modalities)
    
    # Output directory next to checkpoint dir
    analysis_dir = model_dir / "analysis_results"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    num_examples = config['testing']['num_examples']
    if model_type == "reconstructor":
        print("Calculating metrics...")
        metrics = calculate_metrics(results, input_modalities=all_input_modality_combos, output_modalities=output_modalities)
        
        print(f"Saving results to: {analysis_dir}")
        save_results(results, metrics, analysis_dir, input_modalities=all_input_modality_combos, output_modalities=output_modalities)
        
        if plot_from_saved:
            print("Plotting from saved results...")
            plot_results_from_saved(analysis_dir, testing_loader.dataset, num_examples, analysis_dir, input_modalities=all_input_modality_combos, output_modalities=output_modalities)
        else:
            print("Plotting from scratch...")
            plot_results_from_scratch(results, testing_loader, num_examples, analysis_dir, input_modalities=all_input_modality_combos, output_modalities=output_modalities)
    
        plot_metrics_summary(metrics, analysis_dir, input_modalities=all_input_modality_combos, output_modalities=output_modalities)
        print(f"Examples & metrics plotted & saved under: {analysis_dir}")
    
    elif model_type == "classifier":
        raise NotImplementedError("Use testing_classifier.py for classifier models")

if __name__ == "__main__":
    import fire
    fire.Fire(run_tests)
