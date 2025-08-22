import pytorch_lightning as L
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
import torch
import numpy as np
from datetime import datetime
import yaml
from pathlib import Path


from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore2stages
from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore2stages
from daep.Perceiver import PerceiverEncoder
from daep.Classifier import LCC, PhotClassifier, TransformerClassifier
from daep.util_layers import MLPFlatten
from daep.daep import unimodaldaep, multimodaldaep, modality_drop, unimodaldaepclassifier, multimodaldaepclassifier
from functools import partial

class daepReconstructor(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        print(self.training_config['weight_decay'])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.training_config['lr']), 
                                      weight_decay=float(self.training_config['weight_decay']))
        return [optimizer]
    
    def training_step(self, batch, batch_idx):
        x = batch
        loss = self.model(x)
        batch_size = self.training_config['batch']
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss = self.model(x)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        loss = self.model(x)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

class daepReconstructorUnimodal(daepReconstructor):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.data_type = config['data_types'][0]
        self.data_name = config['data_names'][0]
        self.architecture_config = config['unimodal']['architecture']
        self.training_config = config['training']
        
        self.predict_input_modalities = _convert_modalities(tuple([self.data_type]))
        self.predict_output_modality = _convert_modalities(self.data_type)
        
        architecture_config = self.architecture_config
        if self.data_type == "spectra":
            self.encoder = spectraTransceiverEncoder(
                bottleneck_length=architecture_config['shape']['bottlenecklen'],
                bottleneck_dim=architecture_config['shape']['bottleneckdim'],
                model_dim=architecture_config['shape']['model_dim'],
                num_heads=architecture_config['shape']['encoder_heads'],
                num_layers=architecture_config['shape']['encoder_layers'],
                ff_dim=architecture_config['shape']['model_dim'],
                concat=architecture_config['components']['concat'],
            )
            self.score = spectraTransceiverScore2stages(
                bottleneck_dim=architecture_config['shape']['bottleneckdim'],
                model_dim=architecture_config['shape']['model_dim'],
                num_heads=architecture_config['shape']['decoder_heads'],
                num_layers=architecture_config['shape']['decoder_layers'],
                ff_dim=architecture_config['shape']['model_dim'],
                concat=architecture_config['components']['concat'],
                cross_attn_only=architecture_config['components']['cross_attn_only'],
                output_uncertainty=architecture_config['components']['use_uncertainty']
            )
            self.model = unimodaldaep(self.encoder, self.score, regularize=architecture_config['components']['regularize'])
        elif self.data_type == "lightcurves":
            self.encoder = photometricTransceiverEncoder(
                num_bands=1,
                bottleneck_length=architecture_config['shape']['bottlenecklen'],
                bottleneck_dim=architecture_config['shape']['bottleneckdim'],
                model_dim=architecture_config['shape']['model_dim'],
                num_heads=architecture_config['shape']['encoder_heads'],
                ff_dim=architecture_config['shape']['model_dim'],
                num_layers=architecture_config['shape']['encoder_layers'],
                concat=architecture_config['components']['concat'],
                sinpos=architecture_config['components']['sinpos_embed'],
                fourier=architecture_config['components']['fourier_embed'],
            )
            self.score = photometricTransceiverScore2stages(
                num_bands=1,
                bottleneck_dim=architecture_config['shape']['bottleneckdim'],
                model_dim=architecture_config['shape']['model_dim'],
                num_heads=architecture_config['shape']['decoder_heads'],
                ff_dim=architecture_config['shape']['model_dim'],
                num_layers=architecture_config['shape']['decoder_layers'],
                concat=architecture_config['components']['concat'],
                cross_attn_only=architecture_config['components']['cross_attn_only'],
                sinpos=architecture_config['components']['sinpos_embed'],
                fourier=architecture_config['components']['fourier_embed'],
                output_uncertainty=architecture_config['components']['use_uncertainty'],
            )
            self.model = unimodaldaep(self.encoder, self.score, regularize=architecture_config['components']['regularize'],
                                 output_uncertainty=architecture_config['components']['use_uncertainty'],
                                 sinpos=architecture_config['components']['sinpos_embed'],
                                 fourier=architecture_config['components']['fourier_embed'])
        else:
            raise ValueError(f"Invalid data type: {self.data_type}, must be 'spectra' or 'lightcurves'")

    def reconstruct(self, x):
        reconstructed = self.model.reconstruct(x)
        if self.architecture_config['components']['use_uncertainty']:
            pred_flux, pred_flux_uncertainty = reconstructed
            pred_flux = pred_flux['flux']
        else:
            pred_flux = reconstructed['flux']
            pred_flux_uncertainty = np.zeros(pred_flux.shape)
        return {'flux': pred_flux, 'flux_uncertainty': pred_flux_uncertainty}
    
    def set_prediction_modalities(self, input_modalities: list[str], output_modality: str):
        """
        Configure predict-time modalities.

        Parameters
        ----------
        input_modalities : {'spectra','lightcurres','both'} or list[str]
            Conditioning modalities.
        output_modality : {'spectra','lightcurves'}
            Target modality to reconstruct.
        """
        input_modalities = _convert_modalities(input_modalities)
        output_modality = _convert_modalities(output_modality)
        if input_modalities != self.predict_input_modalities or output_modality != self.predict_output_modality:
            raise ValueError("Input or output modalities cannot be changed for unimodal models")

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """
        Predict-step for Lightning predict loop. Reads self.predict_input_modalities
        and self.predict_output_modality to produce a single-modality output that
        UnprocessPredictionWriter can parse.

        Returns
        -------
        dict
            {'spectra': {'flux': ...}} or {'photometry': {'flux': ...}}
            (If you later enable uncertainties, return a tuple of
            (flux_dict, flux_err_dict) for the writer.)
        """
        # Defaults if not set
        try:
            input_modalities = self.predict_input_modalities
            output_modality = self.predict_output_modality
        except AttributeError:
            raise ValueError("Predict-time modalities not set. Call set_prediction_modalities() before predict_step()")

        # Use wrapper's reconstruct to honor your existing API
        predictions = self.reconstruct(batch)

        # Wrap for UnprocessPredictionWriter
        return predictions
    
    def model_name(self):
        return f"daep_reconstructor_unimodal_{self.data_type}"
    
    def model_instance_str(self):
        model_str = f"dim_"
        model_str += f"{self.architecture_config['shape']['bottlenecklen']}-"
        model_str += f"{self.architecture_config['shape']['bottleneckdim']}-"
        model_str += f"{self.architecture_config['shape']['encoder_heads']}-"
        model_str += f"{self.architecture_config['shape']['decoder_heads']}-"
        model_str += f"{self.architecture_config['shape']['encoder_layers']}-"
        model_str += f"{self.architecture_config['shape']['decoder_layers']}-"
        model_str += f"{self.architecture_config['shape']['model_dim']}_"
        
        model_str += f"uncert{self.architecture_config['components']['use_uncertainty']}_"
        model_str += f"concat{self.architecture_config['components']['concat']}_"
        if self.data_type == "lightcurves":
            model_str += f"sinpos{self.architecture_config['components']['sinpos_embed']}_"
            model_str += f"fourier{self.architecture_config['components']['fourier_embed']}_"
        
        model_str += f"lr{self.training_config['lr']}_"
        model_str += f"weight_decay{self.training_config['weight_decay']}_"
        
        model_str += f"date{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        
        return model_str


class daepReconstructorMultimodal(daepReconstructor):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.data_types = config['data_types']
        self.data_name_spectra = config['data_names'][0]
        self.data_name_lightcurves = config['data_names'][1]
        self.architecture_config = config['multimodal']['architecture']
        self.training_config = config['training']
        
        architecture_config = self.architecture_config
        self.spectra_tokenizer = spectraTransceiverEncoder(
                bottleneck_length = architecture_config["shape"]["bottlenecklen"],
                bottleneck_dim = architecture_config["shape"]["spectra_tokens"],
                model_dim = architecture_config["shape"]["model_dim"],
                ff_dim = architecture_config["shape"]["model_dim"],
                num_layers = architecture_config["shape"]["encoder_layers"],
                num_heads = architecture_config["shape"]["encoder_heads"],
            ) 
        self.photometry_tokenizer = photometricTransceiverEncoder(
                num_bands = 1, 
                bottleneck_length = architecture_config["shape"]["bottlenecklen"],
                bottleneck_dim = architecture_config["shape"]["photometry_tokens"],
                model_dim = architecture_config["shape"]["model_dim"], 
                ff_dim = architecture_config["shape"]["model_dim"],
                num_layers = architecture_config["shape"]["encoder_layers"],
                num_heads = architecture_config["shape"]["encoder_heads"],
                sinpos = architecture_config["components"]["sinpos_embed"],
                fourier=architecture_config["components"]["fourier_embed"],
            )
        self.encoder = PerceiverEncoder(
                        bottleneck_length = architecture_config["shape"]["bottlenecklen"],
                        bottleneck_dim = architecture_config["shape"]["bottleneckdim"],
                        model_dim = architecture_config["shape"]["model_dim"],
                        ff_dim = architecture_config["shape"]["model_dim"],
                        num_layers = architecture_config["shape"]["encoder_layers"],
                        num_heads = architecture_config["shape"]["encoder_heads"],
                        selfattn = architecture_config["components"]["mixer_selfattn"]
        )
        self.spectra_score = spectraTransceiverScore2stages(
                        bottleneck_dim = architecture_config["shape"]["bottleneckdim"],
                        model_dim = architecture_config["shape"]["model_dim"],
                        ff_dim = architecture_config["shape"]["model_dim"],
                        num_heads = architecture_config["shape"]["decoder_heads"],
                        num_layers = architecture_config["shape"]["decoder_layers"],
                        concat = architecture_config["components"]["concat"],
                        output_uncertainty=architecture_config["components"]["use_uncertainty"]
                        )
        self.photometry_score = photometricTransceiverScore2stages(
                bottleneck_dim = architecture_config["shape"]["bottleneckdim"],
                    num_bands = 1,
                    model_dim = architecture_config["shape"]["model_dim"],
                    ff_dim = architecture_config["shape"]["model_dim"],
                    num_heads = architecture_config["shape"]["decoder_heads"],
                    num_layers = architecture_config["shape"]["decoder_layers"],
                    concat = architecture_config["components"]["concat"],
                    sinpos = architecture_config["components"]["sinpos_embed"],
                    fourier=architecture_config["components"]["fourier_embed"],
                    output_uncertainty=architecture_config["components"]["use_uncertainty"]
            )
        tokenizers = {
            "spectra": self.spectra_tokenizer,
            "photometry": self.photometry_tokenizer
        }
        scores = {
            "spectra": self.spectra_score,
            "photometry": self.photometry_score
        }
        self.model = multimodaldaep(tokenizers, self.encoder, scores,
            measurement_names={"spectra": "flux", "photometry": "flux"},
            modality_dropping_during_training=partial(modality_drop, p_drop=architecture_config["components"]["dropping_prob"]),
            output_uncertainty=architecture_config["components"]["use_uncertainty"],
            sinpos = architecture_config["components"]["sinpos_embed"],
            fourier = architecture_config["components"]["fourier_embed"]
        )
        
        if 'use_uncertainty' in self.architecture_config['components'] and self.architecture_config['components']['use_uncertainty']:
            raise ValueError("Use uncertainty is not supported for multimodal models")

    def reconstruct(self, x, input_modalities: list[str], output_modality: str):
        reconstructed = self.model.reconstruct(x, condition_keys=input_modalities, out_keys=[output_modality])
        # If the model is multimodal, extract the output_modality from the reconstructed result.
        if isinstance(reconstructed, dict) and output_modality in reconstructed:
            reconstructed = reconstructed[output_modality]
        if self.architecture_config['components']['use_uncertainty']:
            pred_flux, pred_flux_uncertainty = reconstructed
            pred_flux = pred_flux['flux']
        else:
            pred_flux = reconstructed['flux']
            pred_flux_uncertainty = np.zeros(pred_flux.shape)
        return {'flux': pred_flux, 'flux_uncertainty': pred_flux_uncertainty}
    
    def set_prediction_modalities(self, input_modalities: list[str], output_modality: str):
        """
        Configure predict-time modalities.

        Parameters
        ----------
        input_modalities : {'spectra','lightcurres','both'} or list[str]
            Conditioning modalities.
        output_modality : {'spectra','lightcurves'}
            Target modality to reconstruct.
        """
        self.predict_input_modalities = _convert_modalities(input_modalities)
        self.predict_output_modality = _convert_modalities(output_modality)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """
        Predict-step for Lightning predict loop. Reads self.predict_input_modalities
        and self.predict_output_modality to produce a single-modality output that
        UnprocessPredictionWriter can parse.

        Returns
        -------
        dict
            {'spectra': {'flux': ...}} or {'photometry': {'flux': ...}}
            (If you later enable uncertainties, return a tuple of
            (flux_dict, flux_err_dict) for the writer.)
        """
        # Defaults if not set
        try:
            input_modalities = self.predict_input_modalities
            output_modality = self.predict_output_modality
        except AttributeError:
            raise ValueError("Predict-time modalities not set. Call set_prediction_modalities() before predict_step()")

        # Use wrapper's reconstruct to honor your existing API
        predictions = self.reconstruct(batch,
                                       input_modalities=input_modalities,
                                       output_modality=output_modality)

        return predictions
    
    def model_name(self):
        return f"daep_reconstructor_multimodal_spectra_lightcurves"
    
    def model_instance_str(self):
        model_str = f"dim_"
        model_str += f"{self.architecture_config['shape']['bottlenecklen']}-"
        model_str += f"{self.architecture_config['shape']['bottleneckdim']}-"
        model_str += f"{self.architecture_config['shape']['encoder_heads']}-"
        model_str += f"{self.architecture_config['shape']['decoder_heads']}-"
        model_str += f"{self.architecture_config['shape']['encoder_layers']}-"
        model_str += f"{self.architecture_config['shape']['decoder_layers']}-"
        model_str += f"{self.architecture_config['shape']['model_dim']}-"
        model_str += f"{self.architecture_config['shape']['spectra_tokens']}-"
        model_str += f"{self.architecture_config['shape']['photometry_tokens']}_"
        
        model_str += f"modaldropP{self.architecture_config['components']['dropping_prob']}_"
        model_str += f"concat{self.architecture_config['components']['concat']}_"
        model_str += f"sinpos{self.architecture_config['components']['sinpos_embed']}_"
        model_str += f"fourier{self.architecture_config['components']['fourier_embed']}_"
        model_str += f"mixerselfattn{self.architecture_config['components']['mixer_selfattn']}_"
        
        model_str += f"lr{self.training_config['lr']}_"
        model_str += f"weight_decay{self.training_config['weight_decay']}_"
        
        model_str += f"date{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        
        return model_str



class daepClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.training_config['lr']), 
                                      weight_decay=float(self.training_config['weight_decay']))
        return [optimizer]
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        y_actual = batch['starclass']
        x = {k: v for k, v in batch.items() if k != 'starclass'}
        y_pred = self.forward(x)
        y_pred_indices = torch.argmax(y_pred, dim=1)
        # Ensure targets are class indices for loss/metrics
        y_actual_indices = y_actual.argmax(dim=1) if y_actual.ndim > 1 else y_actual
        
        loss = self.loss_fn(y_pred, y_actual)
        
        # torchmetrics.Accuracy supports (N,C) logits with (N,) labels
        acc = self.accuracy(y_pred, y_actual_indices)
        
        batch_size = self.training_config['batch']

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        y_actual = batch['starclass']
        x = {k: v for k, v in batch.items() if k != 'starclass'}
        y_pred = self.forward(x)
        y_pred_indices = torch.argmax(y_pred, dim=1)
        y_actual_indices = y_actual.argmax(dim=1) if y_actual.ndim > 1 else y_actual

        loss = self.loss_fn(y_pred, y_actual)
        acc = self.accuracy(y_pred, y_actual_indices)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Update confusion matrix incrementally per batch
        self.conf_matrix.update(y_pred_indices.to(self.conf_matrix.device), y_actual_indices.to(self.conf_matrix.device))

        return loss

    def test_step(self, batch, batch_idx):
        y_actual = batch['starclass']
        y_actual_indices = y_actual.argmax(dim=1) if y_actual.ndim > 1 else y_actual
        x = {k: v for k, v in batch.items() if k != 'starclass'}
        y_pred = self.forward(x)
        y_pred_indices = torch.argmax(y_pred, dim=1)

        loss = self.loss_fn(y_pred, y_actual)
        acc = self.accuracy(y_pred_indices, y_actual_indices)

        self.conf_matrix.update(y_pred_indices.to(self.conf_matrix.device), y_actual_indices.to(self.conf_matrix.device))
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        for i in range(len(y_pred_indices)):
            if y_pred_indices[i] != y_actual_indices[i]:
                self.misclassified_tests.append((x['idx'][i], y_pred[i], y_actual[i]))

        return loss
    
class daepClassifierUnimodal(daepClassifier):
    def __init__(self, config, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_type = config['data_types'][0]
        self.training_config = config['training']
        self.architecture_config = config['unimodal']['architecture']
        self.num_classes = self.architecture_config['classifier']['shape']['num_classes']
        
        if class_weights is not None:
            class_weights = torch.tensor(class_weights).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
        
        self.misclassified_tests = []
        self.predicted = []
        
        architecture_config = self.architecture_config
        
        if architecture_config['encoder']['use_pretrained_encoder']:
            pretrained_encoder_dir = Path(architecture_config['encoder']['pretrained']['pretrained_encoder_path'])
            
            # Search for checkpoint files matching the epoch number
            # If epoch_num == -1, load the latest checkpoint
            checkpoints_dir = pretrained_encoder_dir / "checkpoints"
            epoch_num = architecture_config['encoder']['pretrained'].get('pretrained_epoch', None)
            if epoch_num == -1:
                checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
            else:
                checkpoint_files = list(checkpoints_dir.glob(f"*epoch={epoch_num}*.ckpt"))
                if not checkpoint_files:
                    raise FileNotFoundError(f"No checkpoint files found in {pretrained_encoder_dir} with epoch={epoch_num}")
            pretrained_encoder_ckpt_path = checkpoint_files[-1]
            
            # Load the pretrained encoder
            try:
                pretrained = daepReconstructorUnimodal.load_from_checkpoint(pretrained_encoder_ckpt_path)
            except Exception as e:
                print(f"Pretrained encoder must be part of a daepReconstructorUnimodal model")
                raise ValueError(f"Failed to load pretrained encoder from {pretrained_encoder_ckpt_path}: {e}")
            pretrained_model = pretrained.model
            self.encoder = pretrained_model.encoder
            print(f"Loaded pretrained encoder from {pretrained_encoder_ckpt_path}")
            
            # Freeze the pretrained encoder parameters if specified
            if architecture_config['encoder']['pretrained']['freeze_pretrained_encoder']:
                for param in self.encoder.parameters():
                    param.requires_grad = False
            
            # Load the pretrained encoder config and update the classifier architecture config to match
            with open(pretrained_encoder_dir / "hparams.yaml", "r") as f:
                pretrained_encoder_config = yaml.safe_load(f)
                pretrained_encoder_config = pretrained_encoder_config['config']
                pretrained_architecture_config = pretrained_encoder_config['unimodal']['architecture']
            
            architecture_config['classifier']['shape']['bottleneckdim'] = pretrained_architecture_config['shape']['bottleneckdim']
            architecture_config['classifier']['shape']['bottlenecklen'] = pretrained_architecture_config['shape']['bottlenecklen']
            
            # Update the encoder architecture config to match the pretrained encoder for model_str construction
            architecture_config['encoder']['new']['shape']['model_dim'] = pretrained_architecture_config['shape']['model_dim']
            architecture_config['encoder']['new']['shape']['encoder_layers'] = pretrained_architecture_config['shape']['encoder_layers']
            architecture_config['encoder']['new']['shape']['encoder_heads'] = pretrained_architecture_config['shape']['encoder_heads']
            architecture_config['encoder']['new']['components']['concat'] = pretrained_architecture_config['components']['concat']
            architecture_config['encoder']['new']['components']['sinpos_embed'] = pretrained_architecture_config['components']['sinpos_embed']
            architecture_config['encoder']['new']['components']['fourier_embed'] = pretrained_architecture_config['components']['fourier_embed']
            architecture_config['encoder']['new']['components']['use_uncertainty'] = pretrained_architecture_config['components']['use_uncertainty']
            
        else:
            # Initialize a new encoder
            if self.data_type == "spectra":
                self.encoder = spectraTransceiverEncoder(
                    bottleneck_length=architecture_config["encoder"]["new"]["shape"]["bottlenecklen"],
                    bottleneck_dim=architecture_config["encoder"]["new"]["shape"]["bottleneckdim"],
                    model_dim=architecture_config["encoder"]["new"]["shape"]["model_dim"],
                    num_heads=architecture_config["encoder"]["new"]["shape"]["encoder_heads"],
                    ff_dim=architecture_config["encoder"]["new"]["shape"]["model_dim"],
                    num_layers=architecture_config["encoder"]["new"]["shape"]["encoder_layers"],
                    concat=architecture_config["encoder"]["new"]["components"]["concat"],
                )
            elif self.data_type == "lightcurves":
                self.encoder = photometricTransceiverEncoder(
                    num_bands=1,
                    bottleneck_length=architecture_config["encoder"]["new"]["shape"]["bottlenecklen"],
                    bottleneck_dim=architecture_config["encoder"]["new"]["shape"]["bottleneckdim"],
                    model_dim=architecture_config["encoder"]["new"]["shape"]["model_dim"],
                    num_heads=architecture_config["encoder"]["new"]["shape"]["encoder_heads"],
                    ff_dim=architecture_config["encoder"]["new"]["shape"]["model_dim"],
                    num_layers=architecture_config["encoder"]["new"]["shape"]["encoder_layers"],
                    concat=architecture_config["encoder"]["new"]["components"]["concat"],
                    sinpos=architecture_config["encoder"]["new"]["components"]["sinpos_embed"],
                    fourier=architecture_config["encoder"]["new"]["components"]["fourier_embed"],
                )
                # Classifier does not use uncertainty
                architecture_config['encoder']['new']['components']['use_uncertainty'] = False
            
            # Update the classifier architecture config to match the encoder
            architecture_config['classifier']['shape']['bottleneckdim'] = architecture_config["encoder"]["new"]["shape"]["bottleneckdim"]
            architecture_config['classifier']['shape']['bottlenecklen'] = architecture_config["encoder"]["new"]["shape"]["bottlenecklen"]
            
        # Initialize a new classifier
        if config['unimodal']['architecture']['classifier']['classifier_type'] == 'mlp':
            # MLP needs flattened input: bottleneck_len * bottleneck_dim
            mlp_input_dim = architecture_config['classifier']['shape']['bottlenecklen'] * architecture_config['classifier']['shape']['bottleneckdim']
            self.classifier = MLPFlatten(mlp_input_dim,
                                  architecture_config['classifier']['shape']['num_classes'],
                                  [64])
        elif config['unimodal']['architecture']['classifier']['classifier_type'] == 'transformer':
            self.classifier = TransformerClassifier(
                bottleneck_dim=architecture_config['classifier']['shape']['bottleneckdim'],
                bottleneck_len=architecture_config['classifier']['shape']['bottlenecklen'],
                dropout_p=architecture_config['classifier']['components']['classifier_dropout'],
                num_classes=architecture_config['classifier']['shape']['num_classes']
            )
        elif config['unimodal']['architecture']['classifier']['classifier_type'] == 'cnn':
            self.classifier = LCC(
                bottleneck_dim=architecture_config['classifier']['shape']['bottleneckdim'],
                bottleneck_len=architecture_config['classifier']['shape']['bottlenecklen'],
                dropout_p=architecture_config['classifier']['components']['classifier_dropout'],
                num_classes=architecture_config['classifier']['shape']['num_classes']
            )
        else:
            raise ValueError(f"Invalid classifier type: {getattr(config['unimodal']['architecture']['classifier'], 'classifier_type')} must be either 'simple' or 'transformer' or 'cnn'")
        
        # Initialize the full model
        model = unimodaldaepclassifier(
                encoder=self.encoder,
                classifier=self.classifier,
                MMD=None,
                regularize=architecture_config["classifier"]["components"]["regularize"])
        self.model = model
        
        print(f"Model {self.model_name()} has {sum(p.numel() for p in model.parameters())} total parameters")
        print(f"Model {self.model_name()} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    def predict_step(self, batch):
        x = batch
        y_pred = self.forward(x)
        return y_pred

    def model_name(self):
        return f"daep_classifier_unimodal_{self.data_type}"
    
    def model_instance_str(self):
        if self.architecture_config['encoder']['use_pretrained_encoder']:
            model_str = "pretrained_"
            if self.architecture_config['encoder']['pretrained']['freeze_pretrained_encoder']:
                model_str += "frozen_"
            else:
                model_str += "unfrozen_"
        else:
            model_str = "fromscratch_"
        
        if self.architecture_config['classifier']['classifier_type'] == 'mlp':
            model_str += "mlp_"
        elif self.architecture_config['classifier']['classifier_type'] == 'transformer':
            model_str += "transformer_"
        elif self.architecture_config['classifier']['classifier_type'] == 'cnn':
            model_str += "cnn_"
        
        if self.architecture_config['classifier']['shape']['weight_by_class']:
            model_str += "weighted_"
        
        model_str += f"dim_"
                
        # Classifier and encoder shape parameters
        model_str += f"{self.architecture_config['classifier']['shape']['bottlenecklen']}-"
        model_str += f"{self.architecture_config['classifier']['shape']['bottleneckdim']}-"
        model_str += f"{self.architecture_config['encoder']['new']['shape']['encoder_layers']}-"
        model_str += f"{self.architecture_config['encoder']['new']['shape']['encoder_heads']}-"
        model_str += f"{self.architecture_config['encoder']['new']['shape']['model_dim']}_"
        
        # Classifier-specific parameters
        model_str += f"classifier_dropout{self.architecture_config['classifier']['components']['classifier_dropout']}-"
        model_str += f"num_classes{self.architecture_config['classifier']['shape']['num_classes']}_"
        
        # Encoder components
        model_str += f"concat{self.architecture_config['encoder']['new']['components']['concat']}_"
        model_str += f"sinpos{self.architecture_config['encoder']['new']['components']['sinpos_embed']}_"
        model_str += f"fourier{self.architecture_config['encoder']['new']['components']['fourier_embed']}_"
        model_str += f"use_uncertainty{self.architecture_config['encoder']['new']['components']['use_uncertainty']}_"
        
        # Classifier components
        model_str += f"regularize{self.architecture_config['classifier']['components']['regularize']}_"
        
        # Optimizer parameters
        model_str += f"lr{self.training_config['lr']}_"
        model_str += f"weight_decay{self.training_config['weight_decay']}_"
        
        # Date stamp
        model_str += f"date{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        
        return model_str
        

class daepClassifierMultimodal(daepClassifier):
    def __init__(self, config, class_weights=None):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.training_config = config['training']
        self.class_weights = class_weights
        self.architecture_config = config['multimodal']['architecture']
        self.num_classes = self.architecture_config['classifier']['shape']['num_classes']
        
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
        
        self.misclassified_tests = []
        self.predicted = []
        
        architecture_config = self.architecture_config
        
        if architecture_config['encoder']['use_pretrained_encoder']:
            pretrained_encoder_dir = Path(architecture_config['encoder']['pretrained']['pretrained_encoder_path'])
            
            # Search for checkpoint files matching the epoch number
            # If epoch_num == -1, load the latest checkpoint
            checkpoints_dir = pretrained_encoder_dir / "checkpoints"
            epoch_num = architecture_config['encoder']['pretrained'].get('pretrained_epoch', None)
            if epoch_num == -1:
                checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
            else:
                checkpoint_files = list(checkpoints_dir.glob(f"*epoch={epoch_num}*.ckpt"))
                if not checkpoint_files:
                    raise FileNotFoundError(f"No checkpoint files found in {pretrained_encoder_dir} with epoch={epoch_num}")
            pretrained_encoder_ckpt_path = checkpoint_files[-1]
            
            # Load the pretrained encoder
            try:
                pretrained_model = daepReconstructorMultimodal.load_from_checkpoint(pretrained_encoder_ckpt_path)
            except Exception as e:
                print(f"Pretrained encoder must be part of a daepReconstructorMultimodal model")
                raise ValueError(f"Failed to load pretrained encoder from {pretrained_encoder_ckpt_path}: {e}")
            
            self.spectra_tokenizer = pretrained_model.tokenizers["spectra"]
            self.photometry_tokenizer = pretrained_model.tokenizers["photometry"]
            self.encoder = pretrained_model.encoder
            print(f"Loaded pretrained encoder from {pretrained_encoder_ckpt_path}")
            
            # Freeze the pretrained encoder parameters if specified
            if architecture_config['encoder']['freeze_pretrained_encoder']:
                for param in self.spectra_tokenizer.parameters():
                    param.requires_grad = False
                for param in self.photometry_tokenizer.parameters():
                    param.requires_grad = False
                for param in self.encoder.parameters():
                    param.requires_grad = False
            
            # Load the pretrained encoder config and update the classifier architecture config to match
            with open(pretrained_encoder_dir / "hparams.yaml", "r") as f:
                pretrained_encoder_config = yaml.safe_load(f)
                pretrained_encoder_config = pretrained_encoder_config['config']
                pretrained_architecture_config = pretrained_encoder_config['multimodal']['architecture']
            
            architecture_config['classifier']['shape']['bottleneckdim'] = pretrained_architecture_config['shape']['bottleneckdim']
            architecture_config['classifier']['shape']['bottlenecklen'] = pretrained_architecture_config['shape']['bottlenecklen']
            
            # Update the encoder architecture config to match the pretrained encoder for model_str construction
            architecture_config['encoder']['new']['shape']['model_dim'] = pretrained_architecture_config['shape']['model_dim']
            architecture_config['encoder']['new']['shape']['encoder_layers'] = pretrained_architecture_config['shape']['encoder_layers']
            architecture_config['encoder']['new']['shape']['encoder_heads'] = pretrained_architecture_config['shape']['encoder_heads']
            architecture_config['encoder']['new']['shape']['spectra_tokens'] = pretrained_architecture_config['shape']['spectra_tokens']
            architecture_config['encoder']['new']['shape']['photometry_tokens'] = pretrained_architecture_config['shape']['photometry_tokens']
            architecture_config['encoder']['new']['components']['dropping_prob'] = pretrained_architecture_config['components']['dropping_prob']
            architecture_config['encoder']['new']['components']['concat'] = pretrained_architecture_config['components']['concat']
            architecture_config['encoder']['new']['components']['sinpos_embed'] = pretrained_architecture_config['components']['sinpos_embed']
            architecture_config['encoder']['new']['components']['fourier_embed'] = pretrained_architecture_config['components']['fourier_embed']
            architecture_config['encoder']['new']['components']['use_uncertainty'] = pretrained_architecture_config['components']['use_uncertainty']
            
        else:
            # Initialize a new encoder
            self.spectra_tokenizer = spectraTransceiverEncoder(
                    bottleneck_length=architecture_config["encoder"]["new"]["shape"]["bottlenecklen"],
                    bottleneck_dim=architecture_config["encoder"]["new"]["shape"]["spectra_tokens"],
                    model_dim=architecture_config["encoder"]["new"]["shape"]["model_dim"],
                    num_heads=architecture_config["encoder"]["new"]["shape"]["encoder_heads"],
                    ff_dim=architecture_config["encoder"]["new"]["shape"]["model_dim"],
                    num_layers=architecture_config["encoder"]["new"]["shape"]["encoder_layers"],
                    concat=architecture_config["encoder"]["new"]["components"]["concat"],
                    use_uncertainty=architecture_config["encoder"]["new"]["components"]["use_uncertainty"]
                )
            self.photometry_tokenizer = photometricTransceiverEncoder(
                num_bands=1,
                bottleneck_length=architecture_config["encoder"]["new"]["shape"]["bottlenecklen"],
                bottleneck_dim=architecture_config["encoder"]["new"]["shape"]["photometry_tokens"],
                model_dim=architecture_config["encoder"]["new"]["shape"]["model_dim"],
                num_heads=architecture_config["encoder"]["new"]["shape"]["encoder_heads"],
                ff_dim=architecture_config["encoder"]["new"]["shape"]["model_dim"],
                num_layers=architecture_config["encoder"]["new"]["shape"]["encoder_layers"],
                concat=architecture_config["encoder"]["new"]["components"]["concat"],
                sinpos=architecture_config["encoder"]["new"]["components"]["sinpos_embed"],
                fourier=architecture_config["encoder"]["new"]["components"]["fourier_embed"],
                use_uncertainty=architecture_config["encoder"]["new"]["components"]["use_uncertainty"]
            )
            self.encoder = PerceiverEncoder(
                bottleneck_length = architecture_config["encoder"]["new"]["shape"]["bottlenecklen"],
                bottleneck_dim = architecture_config["encoder"]["new"]["shape"]["bottleneckdim"],
                model_dim = architecture_config["encoder"]["new"]["shape"]["model_dim"],
                ff_dim = architecture_config["encoder"]["new"]["shape"]["model_dim"],
                num_layers = architecture_config["encoder"]["new"]["shape"]["encoder_layers"],
                num_heads = architecture_config["encoder"]["new"]["shape"]["encoder_heads"],
                selfattn = architecture_config["encoder"]["new"]["components"]["mixer_selfattn"]
            )
            
            # Update the classifier architecture config to match the encoder
            architecture_config['classifier']['shape']['bottleneckdim'] = architecture_config["encoder"]["new"]["shape"]["bottleneckdim"]
            architecture_config['classifier']['shape']['bottlenecklen'] = architecture_config["encoder"]["new"]["shape"]["bottlenecklen"]
            
        # Initialize a new classifier
        classifier = LCC(
            bottleneck_dim=architecture_config['classifier']['shape']['bottleneckdim'],
            bottleneck_len=architecture_config['classifier']['shape']['bottlenecklen'],
            dropout_p=architecture_config['classifier']['components']['classifier_dropout'],
            num_classes=architecture_config['classifier']['shape']['num_classes']
        )
        
        # Initialize the full model
        model = multimodaldaepclassifier(
                tokenizers={"spectra": self.spectra_tokenizer, "photometry": self.photometry_tokenizer},
                encoder=self.encoder,
                classifier=classifier,
                measurement_names={"spectra": "flux", "photometry": "flux"},
                modality_dropping_during_training=partial(modality_drop, p_drop=architecture_config["encoder"]["new"]["components"]["dropping_prob"])
            )
        self.model = model
        
        print(f"Model {self.model_name()} has {sum(p.numel() for p in model.parameters())} total parameters")
        print(f"Model {self.model_name()} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    def predict_step(self, batch):
        x = batch
        test_instance_idx = batch['idx']
        y_pred = self.forward(x)
        for i in range(len(y_pred)):
            self.predicted.append((x[i], y_pred[i]))
        return y_pred

    def model_name(self):
        return f"daep_classifier_unimodal_{self.data_type}"
    
    def model_instance_str(self):
        if self.architecture_config['encoder']['use_pretrained_encoder']:
            model_str = "pretrained_"
            if self.architecture_config['encoder']['pretrained']['freeze_pretrained_encoder']:
                model_str += "frozen_"
            else:
                model_str += "unfrozen_"
        else:
            model_str = "fromscratch_"
        
        if self.architecture_config['classifier']['simple']:
            model_str += "simple_"
        
        if self.class_weights is not None:
            model_str += "weighted_"
        
        model_str += f"dim_"
        
        # Classifier and encoder shape parameters
        model_str += f"{self.architecture_config['classifier']['shape']['bottlenecklen']}-"
        model_str += f"{self.architecture_config['classifier']['shape']['bottleneckdim']}-"
        model_str += f"{self.architecture_config['encoder']['new']['shape']['encoder_layers']}-"
        model_str += f"{self.architecture_config['encoder']['new']['shape']['encoder_heads']}-"
        model_str += f"{self.architecture_config['encoder']['new']['shape']['model_dim']}_"
        model_str += f"{self.architecture_config['encoder']['new']['shape']['spectra_tokens']}-"
        model_str += f"{self.architecture_config['encoder']['new']['shape']['photometry_tokens']}-"
        
        # Classifier-specific parameters
        model_str += f"classifier_dropout{self.architecture_config['classifier']['components']['classifier_dropout']}-"
        model_str += f"num_classes{self.architecture_config['classifier']['components']['num_classes']}_"
        
        # Encoder components
        model_str += f"dropping_prob{self.architecture_config['encoder']['new']['components']['dropping_prob']}_"
        model_str += f"concat{self.architecture_config['encoder']['new']['components']['concat']}_"
        model_str += f"mixerselfattn{self.architecture_config['encoder']['new']['components']['mixer_selfattn']}_"
        model_str += f"sinpos{self.architecture_config['encoder']['new']['components']['sinpos_embed']}_"
        model_str += f"fourier{self.architecture_config['encoder']['new']['components']['fourier_embed']}_"
        model_str += f"use_uncertainty{self.architecture_config['encoder']['new']['components']['use_uncertainty']}_"
        
        # Classifier components
        model_str += f"regularize{self.architecture_config['classifier']['regularize']}_"
        
        # Optimizer parameters
        model_str += f"lr{self.training_config['lr']}_"
        model_str += f"weight_decay{self.training_config['weight_decay']}_"
        
        # Date stamp
        model_str += f"date{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        
        return model_str
        

def _convert_modalities(modalities: list[str]) -> list[str]:
    mapping = {"spectra": "spectra", "lightcurves": "photometry"}
    if isinstance(modalities, str):
        return mapping[modalities]
    elif isinstance(modalities, tuple):
        return tuple([mapping[modality] for modality in modalities])
    else:
        raise ValueError(f"Invalid modalities type: {type(modalities)}")


class PhotClassifierUnimodal(daepClassifier):
    """
    Lightning wrapper for the photometric classifier using a two-stage photometric encoder.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Expected keys mirror the unimodal architecture in `LitWrapperAll`:
        - ``"training"``: optimizer/lr/weight_decay/batch
        - ``"unimodal" -> "architecture" -> "shape"``:
            ``bottlenecklen``, ``bottleneckdim``, ``model_dim``, ``encoder_heads``, ``encoder_layers``
        - ``"unimodal" -> "architecture" -> "components"``:
            ``concat``, ``fourier_embed`` (and optionally ``sinpos_embed``)
        - ``"unimodal" -> "architecture" -> "classifier" -> "shape"``:
            ``num_classes``
    class_weights : torch.Tensor | None
        Optional class weights for `CrossEntropyLoss`.
    """
    def __init__(self, config, class_weights=None):
        super().__init__()
        self.save_hyperparameters()

        self.data_type = "lightcurves"
        self.training_config = config["training"]
        self.architecture_config = config["unimodal"]["architecture"]
        self.class_weights = class_weights

        # Required classifier meta
        self.num_classes = self.architecture_config["classifier"]["shape"]["num_classes"]

        # Metrics and loss
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

        self.misclassified_tests = []
        self.predicted = []
        
        # Override the architecture config with the phot classifier config
        self.architecture_config['encoder']['new']['shape']['bottlenecklen'] = 1
        self.architecture_config['encoder']['new']['shape']['bottleneckdim'] = 128
        self.architecture_config['encoder']['new']['shape']['hidden_len'] = 32
        self.architecture_config['encoder']['new']['shape']['encoder_layers'] = 4
        self.architecture_config['encoder']['new']['shape']['encoder_heads'] = 8
        self.architecture_config['encoder']['new']['shape']['model_dim'] = 128
        self.architecture_config['encoder']['new']['shape']['ff_dim'] = 256
        self.architecture_config['encoder']['new']['components']['concat'] = True
        self.architecture_config['encoder']['new']['components']['selfattn'] = False
        self.architecture_config['encoder']['new']['components']['fourier_embed'] = True
        self.architecture_config['encoder']['new']['components']['sinpos_embed'] = False
        self.architecture_config['classifier']['components']['classifier_dropout'] = 0.1
        self.architecture_config['classifier']['components']['regularize'] = 0.0
        self.architecture_config['classifier']['components']['out_middle'] = [64]

        # Build underlying nn.Module model
        shape = self.architecture_config["encoder"]["new"]["shape"]
        comps = self.architecture_config["encoder"]["new"]["components"]
        comps_classifier = self.architecture_config["classifier"]["components"]

        self.model = PhotClassifier(
            num_classes=self.num_classes,
            num_bands=1,
            bottleneck_length=shape["bottlenecklen"],
            bottleneck_dim=shape["bottleneckdim"],
            hidden_len=shape["hidden_len"],
            model_dim=shape["model_dim"],
            num_heads=shape["encoder_heads"],
            ff_dim=shape["ff_dim"],
            num_layers=shape["encoder_layers"],
            dropout=comps_classifier["classifier_dropout"],
            out_middle=comps_classifier["out_middle"],
            selfattn=comps["selfattn"],
            concat=comps["concat"],
            fourier=comps["fourier_embed"],
        )

        print(f"Model {self.model_name()} has {sum(p.numel() for p in self.model.parameters())} total parameters")
        print(f"Model {self.model_name()} has {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters")

    def model_name(self):
        return "phot_classifier_unimodal_lightcurves"

    def model_instance_str(self):
        """
        Returns
        -------
        str
            A concise, human-readable identifier of the model instance for logging/output paths.
        """
        shape = self.architecture_config["encoder"]["new"]["shape"]
        comps = self.architecture_config["encoder"]["new"]["components"]
        s = "dim_"
        s += f"{shape['bottlenecklen']}-"
        s += f"{shape['bottleneckdim']}-"
        s += f"{shape['encoder_heads']}-"
        s += f"{shape['encoder_layers']}-"
        s += f"{shape['model_dim']}_"

        s += f"classifier_dropout{self.training_config.get('dropout', 0.1)}-"
        s += f"num_classes{self.num_classes}_"

        s += f"concat{comps['concat']}_"
        if 'sinpos_embed' in comps:
            s += f"sinpos{comps['sinpos_embed']}_"
        s += f"fourier{comps['fourier_embed']}_"

        s += f"lr{self.training_config['lr']}_"
        s += f"weight_decay{self.training_config['weight_decay']}_"
        s += f"date{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        return s

    def predict_step(self, batch):
        """
        Parameters
        ----------
        batch : dict
            Batch with keys such as ``'flux'``, ``'time'``, ``'mask'`` (and optionally ``'band'``).

        Returns
        -------
        torch.Tensor
            Class probabilities or logits from the classifier.
        """
        x = batch
        y_pred = self.forward(x)
        for i in range(len(y_pred)):
            self.predicted.append((x[i], y_pred[i]))
        return y_pred