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
from daep.Classifier import LCC
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_config['lr'], 
                                      weight_decay=float(self.training_config['weight_decay']))
        return [optimizer]
    
    def training_step(self, batch, batch_idx):
        x = batch
        loss = self.model(x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
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
        
        architecture_config = self.architecture_config
        if self.data_type == "spectra":
            encoder = spectraTransceiverEncoder(
                bottleneck_length=architecture_config['shape']['bottlenecklen'],
                bottleneck_dim=architecture_config['shape']['bottleneckdim'],
                model_dim=architecture_config['shape']['model_dim'],
                num_heads=architecture_config['shape']['encoder_heads'],
                num_layers=architecture_config['shape']['encoder_layers'],
                ff_dim=architecture_config['shape']['model_dim'],
                concat=architecture_config['components']['concat'],
            )
            score = spectraTransceiverScore2stages(
                bottleneck_dim=architecture_config['shape']['bottleneckdim'],
                model_dim=architecture_config['shape']['model_dim'],
                num_heads=architecture_config['shape']['decoder_heads'],
                num_layers=architecture_config['shape']['decoder_layers'],
                ff_dim=architecture_config['shape']['model_dim'],
                concat=architecture_config['components']['concat'],
                cross_attn_only=architecture_config['components']['cross_attn_only'],
                output_uncertainty=architecture_config['components']['use_uncertainty']
            )
            model = unimodaldaep(encoder, score, regularize=architecture_config['components']['regularize'])
        elif self.data_type == "lightcurves":
            encoder = photometricTransceiverEncoder(
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
            score = photometricTransceiverScore2stages(
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
            model = unimodaldaep(encoder, score, regularize=architecture_config['components']['regularize'],
                                 output_uncertainty=architecture_config['components']['use_uncertainty'],
                                 sinpos=architecture_config['components']['sinpos_embed'],
                                 fourier=architecture_config['components']['fourier_embed'])
        else:
            raise ValueError(f"Invalid data type: {self.data_type}, must be 'spectra' or 'lightcurves'")
        self.model = model

    def reconstruct(self, x):
        reconstructed = self.model.reconstruct(x)
        if self.architecture_config['components']['use_uncertainty']:
            pred_flux, pred_flux_uncertainty = reconstructed
            pred_flux = pred_flux['flux']
        else:
            pred_flux = reconstructed['flux']
            pred_flux_uncertainty = np.zeros(pred_flux.shape)
        return {'flux': pred_flux, 'flux_uncertainty': pred_flux_uncertainty}
    
    def model_name(self):
        return f"daep_reconstructor_unimodal_{self.data_type}"
    
    def model_instance_str(self):
        model_str = f"dim_"
        model_str += f"bottlenecklen{self.architecture_config['shape']['bottlenecklen']}-"
        model_str += f"bottleneckdim{self.architecture_config['shape']['bottleneckdim']}-"
        model_str += f"encoder_layers{self.architecture_config['shape']['encoder_heads']}-"
        model_str += f"encoder_heads{self.architecture_config['shape']['decoder_heads']}-"
        model_str += f"decoder_layers{self.architecture_config['shape']['encoder_layers']}-"
        model_str += f"decoder_heads{self.architecture_config['shape']['decoder_layers']}-"
        model_str += f"model_dim{self.architecture_config['shape']['model_dim']}_"
        
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
        tokenizers = {
            "spectra": spectraTransceiverEncoder(
                bottleneck_length = architecture_config["shape"]["bottlenecklen"],
                bottleneck_dim = architecture_config["shape"]["spectra_tokens"],
                model_dim = architecture_config["shape"]["model_dim"],
                ff_dim = architecture_config["shape"]["model_dim"],
                num_layers = architecture_config["shape"]["encoder_layers"],
                num_heads = architecture_config["shape"]["encoder_heads"],
            ), 
            "photometry": photometricTransceiverEncoder(
                num_bands = 1, 
                bottleneck_length = architecture_config["shape"]["bottlenecklen"],
                bottleneck_dim = architecture_config["shape"]["photometry_tokens"],
                model_dim = architecture_config["shape"]["model_dim"], 
                ff_dim = architecture_config["shape"]["model_dim"],
                num_layers = architecture_config["shape"]["encoder_layers"],
                num_heads = architecture_config["shape"]["encoder_heads"],
                sinpos = architecture_config["components"]["sinpos_embed"],
                fourier=architecture_config["components"]["fourier_embed"],
                output_uncertainty=architecture_config["components"]["use_uncertainty"]
            )
        }
        encoder = PerceiverEncoder(
                        bottleneck_length = architecture_config["shape"]["bottlenecklen"],
                        bottleneck_dim = architecture_config["shape"]["bottleneckdim"],
                        model_dim = architecture_config["shape"]["model_dim"],
                        ff_dim = architecture_config["shape"]["model_dim"],
                        num_layers = architecture_config["shape"]["encoder_layers"],
                        num_heads = architecture_config["shape"]["encoder_heads"],
                        selfattn = architecture_config["components"]["mixer_selfattn"]
        )
        scores = {
            "spectra":spectraTransceiverScore2stages(
                        bottleneck_dim = architecture_config["shape"]["bottleneckdim"],
                        model_dim = architecture_config["shape"]["model_dim"],
                        ff_dim = architecture_config["shape"]["model_dim"],
                        num_heads = architecture_config["shape"]["decoder_heads"],
                        num_layers = architecture_config["shape"]["decoder_layers"],
                        concat = architecture_config["components"]["concat"],
                        output_uncertainty=architecture_config["components"]["use_uncertainty"]
                        ), 
            "photometry": photometricTransceiverScore2stages(
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
        }
        model = multimodaldaep(
            tokenizers, encoder, scores,
            measurement_names={"spectra": "flux", "photometry": "flux"},
            modality_dropping_during_training=partial(modality_drop, p_drop=architecture_config["components"]["dropping_prob"]),
            output_uncertainty=architecture_config["components"]["use_uncertainty"],
            sinpos = architecture_config["components"]["sinpos_embed"],
            fourier = architecture_config["components"]["fourier_embed"]
        )
        self.model = model
        
        if 'use_uncertainty' in self.architecture_config['components'] and self.architecture_config['components']['use_uncertainty']:
            raise ValueError("Use uncertainty is not supported for multimodal models")

    def reconstruct(self, x, input_modalities, output_modalities):
        alt_modalities_dict = {"spectra": ["spectra"], "lightcurves": ["photometry"],
                               "both": ["spectra", "photometry"], ["spectra", "photometry"]: ["spectra", "photometry"]}
        input_modalities = alt_modalities_dict[input_modalities]
        output_modalities = alt_modalities_dict[output_modalities]
        reconstructed = self.model.reconstruct(x, condition_keys=input_modalities, out_keys=output_modalities)
        
        if 'photometry' in reconstructed:
            pred_flux = reconstructed['photometry']['flux']
        elif 'spectra' in reconstructed:
            pred_flux = reconstructed['spectra']['flux']
        
        return pred_flux
    
    def model_name(self):
        return f"daep_reconstructor_multimodal_spectra_lightcurves"
    
    def model_instance_str(self):
        model_str = f"dim_"
        model_str += f"bottlenecklen{self.architecture_config['shape']['bottlenecklen']}-"
        model_str += f"bottleneckdim{self.architecture_config['shape']['bottleneckdim']}-"
        model_str += f"encoder_layers{self.architecture_config['shape']['encoder_heads']}-"
        model_str += f"encoder_heads{self.architecture_config['shape']['decoder_heads']}-"
        model_str += f"decoder_layers{self.architecture_config['shape']['encoder_layers']}-"
        model_str += f"decoder_heads{self.architecture_config['shape']['decoder_layers']}-"
        model_str += f"model_dim{self.architecture_config['shape']['model_dim']}-"
        model_str += f"spectra_tokens{self.architecture_config['shape']['spectra_tokens']}-"
        model_str += f"photometry_tokens{self.architecture_config['shape']['photometry_tokens']}_"
        
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_config['lr'], 
                                      weight_decay=float(self.training_config['weight_decay']))
        return [optimizer]
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        y_actual = batch['starclass']
        x = batch
        y_pred = self.forward(x)
        y_pred_indices = torch.argmax(y_pred, dim=1)
        
        loss = self.loss_fn(y_pred, y_actual)
        acc = self.accuracy(y_pred_indices, y_actual)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_actual = batch['starclass']
        x = {k: v for k, v in batch.items() if k != 'starclass'}
        y_pred = self.forward(x)
        y_pred_indices = torch.argmax(y_pred, dim=1)
        y_actual_indices = torch.argmax(y_actual, dim=1)
        
        loss = self.loss_fn(y_pred, y_actual)
        acc = self.accuracy(y_pred_indices, y_actual)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Update confusion matrix incrementally per batch
        self.conf_matrix.update(y_pred_indices.to(self.conf_matrix.device), y_actual_indices.to(self.conf_matrix.device))
        
        return loss
    
    def test_step(self, batch, batch_idx):
        y_actual = batch['starclass']
        y_actual_indices = torch.argmax(y_actual, dim=1)
        x = batch
        y_pred = self.forward(x)
        y_pred_indices = torch.argmax(y_pred, dim=1)
        
        loss = self.loss_fn(y_pred, y_actual)
        acc = self.accuracy(y_pred_indices, y_actual_indices)
        
        self.conf_matrix.update(y_pred_indices.to(self.conf_matrix.device), y_actual_indices.to(self.conf_matrix.device))
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for i in range(len(y_pred_indices)):
            if y_pred_indices[i] != y_actual_indices[i]:
                self.misclassified_tests.append((x[i], y_pred[i], y_actual[i]))
        
        return loss


class daepClassifierUnimodal(daepClassifier):
    def __init__(self, config, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_type = config['data_types'][0]
        self.training_config = config['training']
        self.class_weights = class_weights
        self.num_classes = self.architecture_config['classifier']['shape']['num_classes']
        
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
        
        self.misclassified_tests = []
        self.predicted = []
        
        architecture_config = self.config['unimodal']['architecture']
        
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
            pretrained_model = torch.load(pretrained_encoder_ckpt_path)
            encoder = pretrained_model.encoder
            print(f"Loaded pretrained encoder from {pretrained_encoder_ckpt_path}")
            
            # Freeze the pretrained encoder parameters if specified
            if architecture_config['encoder']['freeze_pretrained_encoder']:
                for param in encoder.parameters():
                    param.requires_grad = False
            
            # Load the pretrained encoder config and update the classifier architecture config to match
            with open(pretrained_encoder_dir / "hparams.yaml", "r") as f:
                pretrained_encoder_config = yaml.safe_load(f)
            
            architecture_config['classifier']['shape']['bottleneckdim'] = pretrained_encoder_config['model']['bottleneckdim']
            architecture_config['classifier']['shape']['bottlenecklen'] = pretrained_encoder_config['model']['bottlenecklen']
            
            # Update the encoder architecture config to match the pretrained encoder for model_str construction
            architecture_config['encoder']['new']['shape']['model_dim'] = pretrained_encoder_config['model']['model_dim']
            architecture_config['encoder']['new']['shape']['encoder_layers'] = pretrained_encoder_config['model']['encoder_layers']
            architecture_config['encoder']['new']['shape']['encoder_heads'] = pretrained_encoder_config['model']['encoder_heads']
            architecture_config['encoder']['new']['components']['concat'] = pretrained_encoder_config['model']['concat']
            architecture_config['encoder']['new']['components']['sinpos_embed'] = pretrained_encoder_config['model']['sinpos_embed']
            architecture_config['encoder']['new']['components']['fourier_embed'] = pretrained_encoder_config['model']['fourier_embed']
            architecture_config['encoder']['new']['components']['use_uncertainty'] = pretrained_encoder_config['model']['use_uncertainty']
            
        else:
            # Initialize a new encoder
            if self.data_type == "spectra":
                encoder = spectraTransceiverEncoder(
                    bottleneck_length=architecture_config["encoder"]["new"]["shape"]["bottlenecklen"],
                    bottleneck_dim=architecture_config["encoder"]["new"]["shape"]["bottleneckdim"],
                    model_dim=architecture_config["encoder"]["new"]["shape"]["model_dim"],
                    num_heads=architecture_config["encoder"]["new"]["shape"]["encoder_heads"],
                    ff_dim=architecture_config["encoder"]["new"]["shape"]["model_dim"],
                    num_layers=architecture_config["encoder"]["new"]["shape"]["encoder_layers"],
                    concat=architecture_config["encoder"]["new"]["components"]["concat"],
                )
            elif self.data_type == "lightcurves":
                encoder = photometricTransceiverEncoder(
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
        model = unimodaldaepclassifier(
                encoder=encoder,
                classifier=classifier,
                MMD=None,
                regularize=architecture_config["classifier"]["components"]["regularize"])
        self.model = model
        
        print(f"Model {self.model_name()} has {sum(p.numel() for p in model.parameters())} total parameters")
        print(f"Model {self.model_name()} has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    def predict_step(self, batch):
        x = batch
        y_pred = self.forward(x)
        for i in range(len(y_pred)):
            self.predicted.append((x[i], y_pred[i]))
        return y_pred

    def model_name(self):
        return f"daep_classifier_unimodal_{self.data_type}"
    
    def model_instance_str(self):
        model_str = f"dim_"
        
        # Classifier and encoder shape parameters
        model_str += f"bottlenecklen{self.architecture_config['classifier']['shape']['bottlenecklen']}-"
        model_str += f"bottleneckdim{self.architecture_config['classifier']['shape']['bottleneckdim']}-"
        model_str += f"encoder_layers{self.architecture_config['encoder']['new']['shape']['encoder_layers']}-"
        model_str += f"encoder_heads{self.architecture_config['encoder']['new']['shape']['encoder_heads']}-"
        model_str += f"model_dim{self.architecture_config['encoder']['new']['shape']['model_dim']}_"
        
        # Classifier-specific parameters
        model_str += f"classifier_dropout{self.architecture_config['classifier']['components']['classifier_dropout']}-"
        model_str += f"num_classes{self.architecture_config['classifier']['components']['num_classes']}_"
        
        # Encoder components
        model_str += f"concat{self.architecture_config['encoder']['new']['components']['concat']}_"
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
        

class daepClassifierMultimodal(daepClassifier):
    def __init__(self, config, class_weights=None):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.training_config = config['training']
        self.class_weights = class_weights
        self.num_classes = self.architecture_config['classifier']['shape']['num_classes']
        
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
        
        self.misclassified_tests = []
        self.predicted = []
        
        architecture_config = self.config['multimodal']['architecture']
        
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
            pretrained_model = torch.load(pretrained_encoder_ckpt_path)
            spectra_tokenizer = pretrained_model.tokenizers["spectra"]
            photometry_tokenizer = pretrained_model.tokenizers["photometry"]
            encoder = pretrained_model.encoder
            print(f"Loaded pretrained encoder from {pretrained_encoder_ckpt_path}")
            
            # Freeze the pretrained encoder parameters if specified
            if architecture_config['encoder']['freeze_pretrained_encoder']:
                for param in spectra_tokenizer.parameters():
                    param.requires_grad = False
                for param in photometry_tokenizer.parameters():
                    param.requires_grad = False
                for param in encoder.parameters():
                    param.requires_grad = False
            
            # Load the pretrained encoder config and update the classifier architecture config to match
            with open(pretrained_encoder_dir / "hparams.yaml", "r") as f:
                pretrained_encoder_config = yaml.safe_load(f)
            
            architecture_config['classifier']['shape']['bottleneckdim'] = pretrained_encoder_config['model']['bottleneckdim']
            architecture_config['classifier']['shape']['bottlenecklen'] = pretrained_encoder_config['model']['bottlenecklen']
            
            # Update the encoder architecture config to match the pretrained encoder for model_str construction
            architecture_config['encoder']['new']['shape']['model_dim'] = pretrained_encoder_config['model']['model_dim']
            architecture_config['encoder']['new']['shape']['encoder_layers'] = pretrained_encoder_config['model']['encoder_layers']
            architecture_config['encoder']['new']['shape']['encoder_heads'] = pretrained_encoder_config['model']['encoder_heads']
            architecture_config['encoder']['new']['shape']['spectra_tokens'] = pretrained_encoder_config['model']['spectra_tokens']
            architecture_config['encoder']['new']['shape']['photometry_tokens'] = pretrained_encoder_config['model']['photometry_tokens']
            architecture_config['encoder']['new']['components']['dropping_prob'] = pretrained_encoder_config['model']['dropping_prob']
            architecture_config['encoder']['new']['components']['concat'] = pretrained_encoder_config['model']['concat']
            architecture_config['encoder']['new']['components']['sinpos_embed'] = pretrained_encoder_config['model']['sinpos_embed']
            architecture_config['encoder']['new']['components']['fourier_embed'] = pretrained_encoder_config['model']['fourier_embed']
            architecture_config['encoder']['new']['components']['use_uncertainty'] = pretrained_encoder_config['model']['use_uncertainty']
            
        else:
            # Initialize a new encoder
            spectra_tokenizer = spectraTransceiverEncoder(
                    bottleneck_length=architecture_config["encoder"]["new"]["shape"]["bottlenecklen"],
                    bottleneck_dim=architecture_config["encoder"]["new"]["shape"]["spectra_tokens"],
                    model_dim=architecture_config["encoder"]["new"]["shape"]["model_dim"],
                    num_heads=architecture_config["encoder"]["new"]["shape"]["encoder_heads"],
                    ff_dim=architecture_config["encoder"]["new"]["shape"]["model_dim"],
                    num_layers=architecture_config["encoder"]["new"]["shape"]["encoder_layers"],
                    concat=architecture_config["encoder"]["new"]["components"]["concat"],
                    use_uncertainty=architecture_config["encoder"]["new"]["components"]["use_uncertainty"]
                )
            photometry_tokenizer = photometricTransceiverEncoder(
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
            encoder = PerceiverEncoder(
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
                tokenizers={"spectra": spectra_tokenizer, "photometry": photometry_tokenizer},
                encoder=encoder,
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
        model_str = f"dim_"
        
        # Classifier and encoder shape parameters
        model_str += f"bottlenecklen{self.architecture_config['classifier']['shape']['bottlenecklen']}-"
        model_str += f"bottleneckdim{self.architecture_config['classifier']['shape']['bottleneckdim']}-"
        model_str += f"encoder_layers{self.architecture_config['encoder']['new']['shape']['encoder_layers']}-"
        model_str += f"encoder_heads{self.architecture_config['encoder']['new']['shape']['encoder_heads']}-"
        model_str += f"model_dim{self.architecture_config['encoder']['new']['shape']['model_dim']}_"
        model_str += f"spectra_tokens{self.architecture_config['encoder']['new']['shape']['spectra_tokens']}-"
        model_str += f"photometry_tokens{self.architecture_config['encoder']['new']['shape']['photometry_tokens']}-"
        
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
        
        