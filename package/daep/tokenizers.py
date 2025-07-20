import torch
from torch import nn
from torch.nn import functional as F
from .util_layers import *

class photometryTokenizer(nn.Module):
    def __init__(self, num_bands = 6, model_dim = 32):
        super(photometryTokenizer, self).__init__()
        self.time_embd = SinusoidalMLPPositionalEmbedding(model_dim)
        self.bandembd = nn.Embedding(num_bands, model_dim) if num_bands > 1 else None
        self.fluxfc = nn.Linear(1, model_dim)
        self.lcfc = MLP(model_dim * 2, model_dim, [model_dim])
        self.model_dim = model_dim

    def forward(self, x):
        '''
        Args:
            flux: flux (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            time: time (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            band: band of the photometry being taken [batch_size, photometry_length]
        Return:
            encoding of size [batch_size, bottleneck_length, bottleneck_dim]

        '''
        flux, time = x["flux"], x['time']
        band = x.get('band') # does not have to have band
        
        return self.lcfc(torch.cat((self.fluxfc(flux[:, :, None]), self.time_embd(time)), axis = -1)) + (self.bandembd(band) if self.bandembd is not None else 0.0)


class spectraTokenizer(nn.Module):
    def __init__(self, model_dim = 32, concat = False):
        '''
        spectra embedding, sinusoidal-MLP embedding for phase and added to 
            linear embedding of flux, the append in seq space of phase
        Arg: model_dim: model dimension
            concat: if we use concatenate then projection to combine flux and wavelength
        '''
        super(spectraTokenizer, self).__init__()
        self.phase_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand phase to bottleneck
        self.concat = concat
        self.wavelength_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand wavelength to bottleneck
        self.flux_embd = nn.Linear(1, model_dim)
        self.model_dim = model_dim
        if concat:
            self.spfc = MLP(2*model_dim, model_dim, [model_dim])
    
    def forward(self, x):
        '''
        Args:
            wavelength, flux: wavelength and flux, of size [batch, len]
            phase: phase, size [batch,]

        '''
        wavelength, flux, phase = x['wavelength'], x['flux'], x['phase']
        
        if self.concat:
            flux_embd = self.spfc(torch.cat([self.flux_embd(flux[:, :, None]), self.wavelength_embd_layer(wavelength)], -1))
        else:
            flux_embd = self.flux_embd(flux[:, :, None]) + self.wavelength_embd_layer(wavelength)
        phase_embd = self.phase_embd_layer(phase[:, None])
        return torch.cat([flux_embd, phase_embd], dim=1)


class imgTokenizer(nn.Module):
    def __init__(self, 
                    img_size,
                    patch_size=4, 
                    in_channels=3,
                    model_dim = 32, 
                    sincosin = False):
        super().__init__()
        assert img_size % patch_size == 0, "image size has to be divisible to patch size"
        self.model_dim = model_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, model_dim)
        if sincosin:
            pos_embed = SinusoidalPositionalEmbedding2D(model_dim, img_size//patch_size,img_size//patch_size)._build_embedding()
            self.register_buffer('pos_embed', pos_embed, persistent=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, model_dim))
    
    def forward(self, img):
        image_embd = self.patch_embed(img)  # [B, N, D]
        image_embd = image_embd + self.pos_embed  # [B, N, D]
        return image_embd
