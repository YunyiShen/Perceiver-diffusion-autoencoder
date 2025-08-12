import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Any, Dict, List, Sequence, Tuple, Union
import pytorch_lightning as L
from pytorch_lightning.callbacks import BasePredictionWriter
from daep.daep import unimodaldaepclassifier, multimodaldaepclassifier


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
        if pl_module.predict_output_modality == "spectra":
            self._task = "spectra"
            # If the dataset is a multimodal dataset, set the dataset to the spectra dataset
            if hasattr(ds, "spectra_dataset"):
                self._dataset = ds.spectra_dataset
        elif pl_module.predict_output_modality == "lightcurves" or pl_module.predict_output_modality == "photometry":
            self._task = "lightcurves"
            # If the dataset is a multimodal dataset, set the dataset to the lightcurve dataset
            if hasattr(ds, "lightcurve_dataset"):
                self._dataset = ds.lightcurve_dataset
        else:
            raise ValueError(f"Unsupported output modality: {pl_module.predict_output_modality}")
        
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
        # If the model is multimodal, extract indices from the correct modality in the batch.
        output_modality = pl_module.predict_output_modality
        if isinstance(batch.get(output_modality, None), dict):
            if output_modality == "spectra":
                idxs = batch[output_modality]['spectra_idx']
            elif output_modality == "photometry":
                idxs = batch[output_modality]['lightcurve_idx']
        else:
            idxs = batch["idx"]
        idxs = self._to_numpy(idxs)  # matches testing.py using batch['idx']

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
        output_modality = pl_module.predict_output_modality
        if self.save_dir is not None:
            np.savez_compressed(self.save_dir / f"prediction_results_input_{'-'.join(input_modalities)}_output_{output_modality}.npz", **out)

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


class ClassifierPredictionWriter(BasePredictionWriter):
    """
    Collects classifier predictions, ground-truth labels, and metadata during prediction.

    This callback mirrors the interface of `UnprocessPredictionWriter` but for
    classification. It can optionally constrain the modalities used for prediction
    when the underlying model is multimodal by setting `input_modalities`.

    Parameters
    ----------
    input_modalities : {"spectra", "lightcurves", "both"} or list[str]
        Conditioning modalities for a multimodal classifier. Ignored for unimodal.
        If a list is provided, it should use underlying model keys {"spectra", "photometry"}.
    write_interval : {"batch", "epoch"}
        Callback write interval. Defaults to "batch".

    Notes
    -----
    - Expects batches to include `idx` and `starclass` (labels). If `idx` is not present,
      attempts to use dataset-specific keys as a fallback.
    - Works with underlying base models of type `unimodaldaepclassifier` or
      `multimodaldaepclassifier`.
    """

    def __init__(self, input_modalities: Union[str, List[str]], output_modality: str, write_interval: str = "batch") -> None:
        super().__init__(write_interval=write_interval)
        # Normalize conditioning keys for multimodal models
        if isinstance(input_modalities, str):
            mapping: Dict[str, List[str]] = {
                "spectra": ["spectra"],
                "lightcurves": ["photometry"],
                "both": ["spectra", "photometry"],
                "photometry": ["photometry"],
            }
            self._condition_keys: List[str] = mapping[input_modalities]
        else:
            self._condition_keys = list(input_modalities)

        # Accumulators
        self._output_modality: str = output_modality
        self._prediction_probs: List[np.ndarray] = []
        self._predicted_classes: List[np.ndarray] = []
        self._ground_truth: List[np.ndarray] = []
        self._indices: List[np.ndarray] = []
        self._star_ids: List[Any] = []
        self._dataset: Any = None

    def on_predict_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Resolve and cache the base dataset reference for fetching metadata.
        """
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
        Compute or accept predictions and accumulate per-batch arrays.

        This recomputes predictions using the underlying base model to honor
        `input_modalities` for multimodal classifiers.
        """
        base_model = getattr(pl_module, "model", pl_module)

        # Generate prediction probabilities/logits
        # Prepare model inputs (exclude targets)
        model_input = {k: v for k, v in batch.items() if k != "starclass"}

        if isinstance(base_model, unimodaldaepclassifier):
            pred_logits_or_probs = base_model(model_input)
        elif isinstance(base_model, multimodaldaepclassifier):
            pred_logits_or_probs = base_model.predict(model_input, condition_keys=self._condition_keys)
        else:
            raise ValueError("Unsupported base model type for ClassifierPredictionWriter")

        # Underlying models already return probabilities
        pred_probs = pred_logits_or_probs

        # Ground truth handling (supports one-hot or class indices)
        targets = batch.get("starclass")
        if targets is None:
            raise ValueError("Batch does not contain 'starclass' labels required for classification evaluation")

        # To numpy
        pred_probs_np = self._to_numpy(pred_probs)
        targets_np = self._to_numpy(targets)

        # Predicted class indices
        predicted_np = np.argmax(pred_probs_np, axis=1).astype(np.int64)

        # Ground truth indices
        if targets_np.ndim > 1 and targets_np.shape[1] > 1:
            ground_truth_np = np.argmax(targets_np, axis=1).astype(np.int64)
        else:
            ground_truth_np = targets_np.reshape(-1).astype(np.int64)

        # Test instance indices
        try:
            idxs = self._to_numpy(batch["idx"]).astype(np.int64)
        except KeyError:
            if "photometry" in batch and isinstance(batch["photometry"], dict) and "lightcurve_idx" in batch["photometry"]:
                idxs = self._to_numpy(batch["photometry"]["lightcurve_idx"]).astype(np.int64)
            elif "spectra" in batch and isinstance(batch["spectra"], dict) and "spectra_idx" in batch["spectra"]:
                idxs = self._to_numpy(batch["spectra"]["spectra_idx"]).astype(np.int64)
            else:
                raise ValueError("No 'idx' or dataset-specific index key present in batch")

        # Star IDs for each sample in batch
        star_ids_batch: List[Any] = []
        for i, idx in enumerate(idxs):
            if hasattr(self._dataset, "get_actual_spectrum"):
                actual = self._dataset.get_actual_spectrum(int(idx))
            elif hasattr(self._dataset, "get_actual_lightcurve"):
                actual = self._dataset.get_actual_lightcurve(int(idx))
            elif hasattr(self._dataset, "spectra_dataset") and hasattr(self._dataset, "lightcurve_dataset"):
                # Combined dataset
                if self._output_modality == "spectra":
                    actual = self._dataset.spectra_dataset.get_actual_spectrum(int(idx))
                elif self._output_modality == "lightcurves":
                    actual = self._dataset.lightcurve_dataset.get_actual_lightcurve(int(idx))
                else:
                    raise ValueError("output_modality must be one of {'spectra','lightcurves'}")
            else:
                raise ValueError("Unsupported dataset type for fetching star IDs")
            star_ids_batch.append(actual["ids"][2])

        # Accumulate
        self._indices.append(idxs)
        self._prediction_probs.append(pred_probs_np)
        self._predicted_classes.append(predicted_np)
        self._ground_truth.append(ground_truth_np)
        self._star_ids.extend(star_ids_batch)

    def to_results(self, output_modality: str) -> Dict[str, Any]:
        """
        Assemble concatenated results for a single modality combination.

        Parameters
        ----------
        output_modality : {"spectra", "lightcurves"}
            Used to decide the star ID field name in the returned dict.

        Returns
        -------
        dict
            Results dictionary with keys matching the previous testing pipeline.
        """
        predictions = np.concatenate(self._predicted_classes, axis=0) if len(self._predicted_classes) else np.empty((0,), dtype=np.int64)
        ground_truth = np.concatenate(self._ground_truth, axis=0) if len(self._ground_truth) else np.empty((0,), dtype=np.int64)
        prediction_probs = np.concatenate(self._prediction_probs, axis=0) if len(self._prediction_probs) else np.empty((0, 0), dtype=np.float32)
        test_instance_idxs = np.concatenate(self._indices, axis=0) if len(self._indices) else np.empty((0,), dtype=np.int64)

        out: Dict[str, Any] = {
            "predictions": predictions,
            "ground_truth": ground_truth,
            "prediction_probs": prediction_probs,
            "test_instance_idxs": test_instance_idxs,
        }
        if output_modality == "spectra":
            out["sobject_ids"] = self._star_ids
        elif output_modality == "lightcurves":
            out["ticids"] = self._star_ids
        else:
            raise ValueError("output_modality must be one of {'spectra','lightcurves'}")
        return out

    # --------- helpers ---------
    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if x is None:
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)