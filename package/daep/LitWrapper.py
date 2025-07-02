# lit_model.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
import torch
from torch import optim
runDir  = '/n/holylabs/LABS/iaifi_lab/Users/agagliano/galaxyAutoencoder/'
sys.path.insert(0, runDir + '/ssl-legacysurvey/')
from ssl_legacysurvey.data_loaders import datamodules
from ssl_legacysurvey.utils import format_logger
from torch import nn
from focal_frequency_loss import FocalFrequencyLoss as FFL

import torch
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl

from daep.daep import unimodaldaep

import torch
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim
import torchvision
import pytorch_lightning as pl
from daep.ImgLayers import HostImgTransceiverEncoder, HostImgTransceiverScore

class LitMaskeddaep(pl.LightningModule):
    def __init__(self, params):
        """
        params: dict containing at least
            - img_size
            - latent_len
            - latent_dim
            - patch_size, in_channels, etc.
            - lr_vae (learning rate for the VAE)
            - beta (if you want to weight KL)
            - sincosin 
            - lambda_roi (how much to upweight the galaxy region)
            - grid_every (create image grid every N epochs)
        """
        super().__init__()

        self.save_hyperparameters(params)

        img_encoder = HostImgTransceiverEncoder(img_size=params['img_size'],
                    bottleneck_length = params['bottleneck_len'],
                    bottleneck_dim = params['bottleneck_dim'],
                    model_dim = params['model_dim'],
                    num_layers = params['num_encoder_layers'],
                    sincosin = params['fixed_positional_encoding'],
                    patch_size=params['patch_size']).to(self.device)

        img_score = HostImgTransceiverScore(
            img_size = params['img_size'],
            bottleneck_dim = params['bottleneck_dim'],
            model_dim = params['model_dim'],
            num_layers = params['num_decoder_layers'],
            sincosin = params['fixed_positional_encoding'],
            patch_size=params['patch_size']
        ).to(self.device)

        self.model = unimodaldaep(img_encoder, img_score, regularize = params['beta'], T = params["diffusion_steps"]).to(self.device)

        # Learning rate
        self.lr = params.get("learning_rate", 5.e-4)
        self.mean_dict= params['mean_dict']
        self.full_means = torch.Tensor(self.mean_dict['full_means'], device=self.device)
        self.full_stds = torch.Tensor(self.mean_dict['full_stds'], device=self.device)
        self.segment = params.get("segment", False)
        self.lambda_roi = params.get("lambda_roi", 0.5)
        self.grid_every = params.get("grid_every", 1)

    def forward(self, batch):
        """
        If batch is a dict with key "image", or a tuple/list, handle both.
        """
        if self.segment:
            x, seg_mask, y, z = batch
        else:
            x, y, z = batch
            seg_mask = None

        return x, seg_mask, self.model(x, K=1)

    def training_step(self, batch, batch_idx):
        if self.segment:
            images, seg_mask, y, phot = batch
        else:
            images, y, phot   = batch
            seg_mask       = None

        x = { self.model.name: images }

        z = self.model.encode(x)

        # MMD
        qz_x = self.model.prior(*self.model.prior_params) \
                           .rsample([z.shape[0]]).to(z.device)
        flat_z  = z.view(z.size(0), -1)
        flat_qz = qz_x.view(z.size(0), -1)

        train_mmd_loss = self.model.regularize * self.model.MMD(flat_z, flat_qz)
        
        per_pixel_loss = self.model.diffusion_trainer(self.model.score, x, z, self.model.name)

        if self.segment:
            seg_mask = seg_mask.unsqueeze(1).to(per_pixel_loss.device)  # [B, 1, H, W]
        
            # Create a weighted mask for central galaxy vs background
            weighted_mask = self.lambda_roi * seg_mask + (1 - self.lambda_roi) * (~seg_mask)

            # Apply mask and renormalize so overall loss magnitude stays stable
            train_score_loss = (per_pixel_loss * weighted_mask).sum() / weighted_mask.sum()
        else:
            train_score_loss = per_pixel_loss.mean()
            
        train_loss = train_mmd_loss + train_score_loss

        # log losses
        self.log("train_mmd_loss",   train_mmd_loss,    on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_score_loss", train_score_loss,  on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss", train_loss,  on_step=True, on_epoch=True, prog_bar=True)

        if (self.current_epoch % self.grid_every == 0) and (self.global_rank == 0) and (batch_idx == 0):
            # get reconstructions
            rec = self.model.reconstruct(x)
            recon_img = rec[self.model.name]
            real_img  = images.clone()

            recon_grid = torchvision.utils.make_grid(recon_img[:4], nrow=4)
            real_grid  = torchvision.utils.make_grid(real_img[:4],  nrow=4)

            if self.segment:
                mask_grid = torchvision.utils.make_grid(seg_mask[:4].float(), nrow=4)
                self.logger.experiment.add_image(
                    "train_seg_mask", mask_grid, self.current_epoch
                )   

            self.logger.experiment.add_image(
                "train_reconstructions", recon_grid, self.current_epoch
            )
            self.logger.experiment.add_image(
                "train_ground_truth",    real_grid,  self.current_epoch
            )

        return train_loss

    def test_step(self, batch, batch_idx):
        if self.segment:
            images, seg_mask, y, phot = batch
        else:
            images, y, phot = batch
            seg_mask = torch.zeros_like(images[:, :1])  # dummy mask

        x   = {self.model.name: images}
        z   = self.model.encode(x)
        rec = self.model.reconstruct(x)[self.model.name]

        qz_x = self.model.prior(*self.model.prior_params).rsample([z.size(0)]).to(z.device)
        mmd  = self.model.regularize * self.model.MMD(
            z.view(z.size(0), -1),
            qz_x.view(z.size(0), -1),
        )
        score = self.model.diffusion_trainer(
            self.model.score, x, z, self.model.name
        ).mean()

        # the callback wants exactly these keys:
        return {
            "img":   images,
            "mask":  seg_mask,
            "rec":   rec,
            "lat":   z,
            "y":     y,
            "rmag":  phot[:, 1],
            "mmd":   mmd,
            "score": score,
        }

    def validation_step(self, batch, batch_idx):
        if self.segment:
            images, seg_mask, y, phot = batch
        else:
            images, y, phot   = batch
            seg_mask       = None

        x = { self.model.name: images }

        z = self.model.encode(x)

        qz_x = self.model.prior(*self.model.prior_params).rsample([z.shape[0]]).to(z.device)
        flat_z    = z.view(z.size(0), -1)
        flat_qz_x = qz_x.view(z.size(0), -1)

        val_mmd_loss  = self.model.regularize * self.model.MMD(flat_z, flat_qz_x)

        per_pixel_loss = self.model.diffusion_trainer(self.model.score, x, z, self.model.name)

        if self.segment:
            seg_mask = seg_mask.unsqueeze(1).to(per_pixel_loss.device)  # [B, 1, H, W]
        
            # Create a weighted mask: galaxy pixels get `lambda_roi`, others get `1 - lambda_roi`
            weighted_mask = self.lambda_roi * seg_mask + (1 - self.lambda_roi) * (~seg_mask)

            # Apply mask and renormalize so overall loss magnitude stays stable
            val_score_loss = (per_pixel_loss * weighted_mask).sum() / weighted_mask.sum()
        else:
            val_score_loss = per_pixel_loss.mean()

        val_loss = val_mmd_loss + val_score_loss

        # 3) log them
        self.log("val_mmd_loss",   val_mmd_loss,   on_epoch=True, prog_bar=False)
        self.log("val_score_loss", val_score_loss, on_epoch=True, prog_bar=False)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        # 4) occasionally log reconstructions
        if (self.current_epoch % self.grid_every == 0) and (batch_idx == 0):
            rec = self.model.reconstruct(x)
            rec_img  = rec[self.model.name]
            real_img = images.clone()

            recon_grid = torchvision.utils.make_grid(rec_img[:4], nrow=4)
            real_grid  = torchvision.utils.make_grid(real_img[:4], nrow=4)

            if self.segment:
                mask_grid = torchvision.utils.make_grid(seg_mask[:4].float(), nrow=4)

                self.logger.experiment.add_image(
                    "val_seg_mask", mask_grid, self.current_epoch
                )

            self.logger.experiment.add_image(
                "val_reconstructions", recon_grid, self.current_epoch
            )

            self.logger.experiment.add_image(
                "val_ground_truth",    real_grid,  self.current_epoch
            )

        return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return [optimizer]

class Litdaep(pl.LightningModule):
    def __init__(self, params):
        """
        params: dict containing at least
            - img_size
            - latent_len
            - latent_dim
            - patch_size, in_channels, etc.
            - lr_vae (learning rate for the VAE)
            - beta (if you want to weight KL)
            - prior, likelihood, posterior (torch.distributions classes)
        """
        super().__init__()

        self.save_hyperparameters(params)

        img_encoder = HostImgTransceiverEncoder(img_size=params['img_size'],
                    bottleneck_length = params['bottleneck_len'],
                    bottleneck_dim = params['bottleneck_dim'],
                    model_dim = params['model_dim'],
                    num_layers = params['num_encoder_layers'],
                    sincosin = params['fixed_positional_encoding'],
                    patch_size=params['patch_size']).to(self.device)
        
        img_score = HostImgTransceiverScore(
            img_size = params['img_size'],
            bottleneck_dim = params['bottleneck_dim'],
            model_dim = params['model_dim'],
            num_layers = params['num_decoder_layers'],
            sincosin = params['fixed_positional_encoding'],
            patch_size=params['patch_size']
        ).to(self.device)

        self.model = unimodaldaep(img_encoder, img_score, regularize = params['beta'], T = params["diffusion_steps"]).to(self.device)

        # Learning rate
        self.lr = params.get("learning_rate", 5.e-4)
        self.mean_dict= params['mean_dict']
        self.full_means = torch.Tensor(self.mean_dict['full_means'], device=self.device)
        self.full_stds = torch.Tensor(self.mean_dict['full_stds'], device=self.device)
        self.segment = params.get("segment", False)
        self.lambda_roi = params.get("lambda_roi", 0.5)
        self.grid_every = params.get("grid_every", 1)

    def forward(self, batch):
        """
        If batch is a dict with key "image", or a tuple/list, handle both.
        """
        if self.segment:
            x, seg_mask, y, z = batch
        else: 
            x, y, z = batch
            seg_mask = None

        return x, seg_mask, self.model(x, K=1)

    def training_step(self, batch, batch_idx):
        if self.segment:
            images, seg_mask, y, phot = batch
        else:
            images, y, phot   = batch
            seg_mask       = None

        x = { self.model.name: images }

        z = self.model.encode(x)

        # MMD
        qz_x = self.model.prior(*self.model.prior_params) \
                           .rsample([z.shape[0]]).to(z.device)
        flat_z  = z.view(z.size(0), -1)
        flat_qz = qz_x.view(z.size(0), -1)

        train_mmd_loss = self.model.regularize * self.model.MMD(flat_z, flat_qz)
        train_score_loss = self.model.diffusion_trainer(self.model.score, x, z, self.model.name).mean()
        train_loss = train_mmd_loss + train_score_loss

        # log losses
        self.log("train_mmd_loss",   train_mmd_loss,    on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_score_loss", train_score_loss,  on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss", train_loss,  on_step=True, on_epoch=True, prog_bar=True)

        if (self.current_epoch % 5 == 0) and (self.global_rank == 0) and (batch_idx == 0):
            # get reconstructions
            rec = self.model.reconstruct(x)
            recon_img = rec[self.model.name]
            real_img  = images.clone()

            recon_grid = torchvision.utils.make_grid(recon_img[:4], nrow=4)
            real_grid  = torchvision.utils.make_grid(real_img[:4],  nrow=4)

            self.logger.experiment.add_image(
                "train_reconstructions", recon_grid, self.current_epoch
            )
            self.logger.experiment.add_image(
                "train_ground_truth",    real_grid,  self.current_epoch
            )

        return train_loss


    def validation_step(self, batch, batch_idx):
        if self.segment:
            images, seg_mask, y, phot = batch
        else:
            images, y, phot   = batch
            seg_mask       = None

        x = { self.model.name: images }

        z = self.model.encode(x)

        qz_x = self.model.prior(*self.model.prior_params).rsample([z.shape[0]]).to(z.device)
        flat_z    = z.view(z.size(0), -1)
        flat_qz_x = qz_x.view(z.size(0), -1)

        val_mmd_loss  = self.model.regularize * self.model.MMD(flat_z, flat_qz_x)
        val_score_loss = self.model.diffusion_trainer(self.model.score, x, z, self.model.name).mean()
        val_loss = val_mmd_loss + val_score_loss

        # 3) log them
        self.log("val_mmd_loss",   val_mmd_loss,   on_epoch=True, prog_bar=False)
        self.log("val_score_loss", val_score_loss, on_epoch=True, prog_bar=False)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        # 4) occasionally log reconstructions
        if (self.current_epoch % 5 == 0) and (batch_idx == 0):
            rec = self.model.reconstruct(x)
            rec_img  = rec[self.model.name]
            real_img = images.clone()

            recon_grid = torchvision.utils.make_grid(rec_img[:4], nrow=4)
            real_grid  = torchvision.utils.make_grid(real_img[:4], nrow=4)

            self.logger.experiment.add_image(
                "val_reconstructions", recon_grid, self.current_epoch
            )
            self.logger.experiment.add_image(
                "val_ground_truth",    real_grid,  self.current_epoch
            )

        return val_loss

    def test_step(self, batch, batch_idx):
        if self.segment:
            images, seg_mask, y, phot = batch
        else:
            images, y, phot = batch
            seg_mask = torch.zeros_like(images[:, :1])  # dummy segmentation mask

        x = { self.model.name: images }
        z = self.model.encode(x)
        rec = self.model.reconstruct(x)[self.model.name]

        qz_x = self.model.prior(*self.model.prior_params).rsample([z.shape[0]]).to(z.device)
        flat_z = z.view(z.size(0), -1)
        flat_qz_x = qz_x.view(z.size(0), -1)

        mmd = self.model.regularize * self.model.MMD(flat_z, flat_qz_x)
        score = self.model.diffusion_trainer(self.model.score, x, z, self.model.name).mean()

        return {
            "img": images,
            "mask": seg_mask,
            "rec": rec,
            "lat": z,
            "y": y,
            "rmag": phot[:, 1],
            "mmd": mmd,
            "score": score,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return [optimizer]

