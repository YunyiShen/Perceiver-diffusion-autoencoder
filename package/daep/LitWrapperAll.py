import lightning as L
from daep.daep import unimodaldaep
import torch
import numpy as np

from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore2stages
from daep.PhotometricLayers import photometricTransceiverEncoder, photometricTransceiverScore2stages
from daep.Perceiver import PerceiverEncoder
from daep.daep import multimodaldaep, modality_drop
from functools import partial


class daepReconstructorUnimodal(L.LightningModule):
    def __init__(self, data_type, config):
        super().__init__()
        self.save_hyperparameters()
        self.test_name = config['test_name']
        self.data_type = data_type
        # self.model_type = config['model_type']
        self.architecture_config = config['architecture']
        self.optimizer_config = config['optimizer']
        
        architecture_config = self.architecture_config
        if data_type == "spectra":
            encoder = spectraTransceiverEncoder(
                bottleneck_length = architecture_config['shape']['bottlenecklen'],
                bottleneck_dim = architecture_config['shape']['bottleneckdim'],
                model_dim = architecture_config['shape']['model_dim'],
                num_heads = architecture_config['shape']['encoder_heads'],
                num_layers = architecture_config['shape']['encoder_layers'],
                ff_dim = architecture_config['shape']['model_dim'],
                concat = architecture_config['components']['concat'],
                use_uncertainty = architecture_config['components']['use_uncertainty']
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
        elif data_type == "lightcurves":
            encoder = photometricTransceiverEncoder(
                num_bands=1,
                bottleneck_length=architecture_config['shape']['bottlenecklen'],
                bottleneck_dim=architecture_config['shape']['bottleneckdim'],
                model_dim=architecture_config['shape']['model_dim'],
                num_heads=architecture_config['shape']['encoder_heads'],
                ff_dim=architecture_config['shape']['model_dim'],
                num_layers=architecture_config['shape']['encoder_layers'],
                concat=architecture_config['components']['concat'],
                sinpos = architecture_config['components']['sinpos_embed'],
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
                sinpos = architecture_config['components']['sinpos_embed'],
                fourier=architecture_config['components']['fourier_embed'],
                output_uncertainty=architecture_config['components']['use_uncertainty'],
            )
            model = unimodaldaep(encoder, score, regularize=architecture_config['components']['regularize'],
                                sinpos=architecture_config['components']['sinpos_embed'],
                                fourier=architecture_config['components']['fourier_embed'])
        else:
            raise ValueError(f"Invalid data type: {data_type}, must be 'spectra' or 'lightcurves'")
        self.model = model

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.optimizer_config['lr'], 
                                      weight_decay=self.optimizer_config['weight_decay'])
        return [optimizer]
    
    def training_step(self, batch, batch_idx):
        x = batch
        loss = self.model(x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss = self.model(x)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        loss = self.model(x)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

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
        model_str += f"mixerselfattn{self.architecture_config['components']['mixer_selfattn']}_"
        
        model_str += f"lr{self.optimizer_config['lr']}_"
        model_str += f"weight_decay{self.optimizer_config['weight_decay']}_"
        
        return model_str


class daepReconstructorMultimodal(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.test_name = config['test_name']
        self.architecture_config = config['architecture']
        self.optimizer_config = config['optimizer']
        
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

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.optimizer_config['lr'], 
                                      weight_decay=self.optimizer_config['weight_decay'])
        return [optimizer]
    
    def training_step(self, batch, batch_idx):
        x = batch
        loss = self.model(x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss = self.model(x)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        loss = self.model(x)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

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
        
        model_str += f"lr{self.optimizer_config['lr']}_"
        model_str += f"weight_decay{self.optimizer_config['weight_decay']}_"
        
        return model_str


class daepClassifierUnimodal(L.LightningModule):
    def __init__(self, data_type, config, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.test_name = config['test_name']
        self.data_type = data_type
        self.architecture_config = config['architecture']
        self.optimizer_config = config['optimizer']
        self.class_weights = class_weights
        
        
        
        




        
        
        