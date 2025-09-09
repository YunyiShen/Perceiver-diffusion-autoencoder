import torch
from torch import nn
from torch.nn import functional as F
from .util_layers import * # useful base layers
from .Perceiver import PerceiverEncoder, PerceiverDecoder, PerceiverDecoder2stages, PerceiverEncoder2stages

###############################
# Transceivers for spectra data
###############################

class spectraEmbedding(nn.Module):
    def __init__(self, model_dim = 32, concat = False):
        '''
        spectra embedding, sinusoidal-MLP embedding for phase and added to 
            linear embedding of flux, the append in seq space of phase
        Arg: model_dim: model dimension
            concat: if we use concatenate then projection to combine flux and wavelength
        '''
        super(spectraEmbedding, self).__init__()
        self.phase_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand phase to bottleneck
        self.concat = concat
        self.wavelength_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand wavelength to bottleneck
        self.flux_embd = nn.Linear(1, model_dim)
        if concat:
            self.spfc = MLP(2*model_dim, model_dim, [model_dim])
    
    def forward(self, wavelength, flux, phase):
        '''
        Args:
            wavelength, flux: wavelength and flux, of size [batch, len]
            phase: phase, size [batch,]

        '''
        if self.concat:
            flux_embd = self.spfc(torch.cat([self.flux_embd(flux[:, :, None]), self.wavelength_embd_layer(wavelength)], -1))
        else:
            flux_embd = self.flux_embd(flux[:, :, None]) + self.wavelength_embd_layer(wavelength)
        phase_embd = self.phase_embd_layer(phase[:, None])
        return torch.cat([flux_embd, phase_embd], dim=1)



class wavelengthphaseEmbedding(nn.Module):
    def __init__(self, model_dim = 32):
        '''
        spectra embedding, sinusoidal-MLP embedding for phase and added to 
            linear embedding of flux, the append in seq space of phase
        Arg: model_dim: model dimension
            concat: if we use concatenate then projection to combine flux and wavelength
        '''
        super(wavelengthphaseEmbedding, self).__init__()
        self.phase_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand phase to bottleneck
        self.wavelength_embd_layer = SinusoidalMLPPositionalEmbedding(model_dim)# expand wavelength to bottleneck
    
    def forward(self, wavelength, phase):
        '''
        Args:
            wavelength, flux: wavelength and flux, of size [batch, len]
            phase: phase, size [batch,]

        '''
        
        phase_embd = self.phase_embd_layer(phase[:, None])
        return torch.cat([self.wavelength_embd_layer(wavelength), phase_embd], dim=1)


class spectraTransceiverScore(nn.Module):
    def __init__(self,
                 bottleneck_dim,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 selfattn=False,
                 concat = True,
                 cross_attn_only = False
                 ):
        '''
        A transformer to decode something (latent) into spectra given time and band
        Args:
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            num_bands: number of bands, currently embedded as class
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the latent
            cross_attn_only: if we want the score function to only have cross attention to the latent, better speed
        '''
        super(spectraTransceiverScore, self).__init__()
        self.decoder = PerceiverDecoder(
            bottleneck_dim,
                 1,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout, 
                 selfattn,
                 cross_attn_only
        )
        self.spectraEmbd = spectraEmbedding(model_dim, concat)
        self.model_dim = model_dim
        
    
    def forward(self, x, bottleneck, aux):
        '''
        Args:
            x: a dictionary of: 
                wavelength: wavelength of the spectra being taken [batch_size, spectra_length]
                phase: phase of the spectra being taken [batch_size, 1]
                mask: mask of spectra [batch_size, spectra_length]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            Decoded spectra of shape [batch_size, spectra_length]
        '''
        flux, wavelength, phase, mask = x['flux'] ,x['wavelength'], x['phase'], x['mask']
        x = self.spectraEmbd(wavelength, flux, phase)
        if aux is not None:
            aux = torch.cat((x[:, -1,:][:, None, :], aux), axis = 1) # aux has original aux (diffusion time usually) and phase embedding
        else:
            aux = x[:, -1,:][:, None, :]
        x = x[:, :-1, :]
        return self.decoder(bottleneck, x, aux, mask).squeeze(-1) 
    
    
class spectraTransceiverScore2stages(nn.Module):
    def __init__(self,
                 bottleneck_dim,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 selfattn=False,
                 concat = True,
                 cross_attn_only = False,
                 hidden_len = 256
                 ):
        '''
        A transformer to decode something (latent) into spectra given time and frequency, with a two stage procedure for efficiency
        Args:
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            num_bands: number of bands, currently embedded as class
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the latent
            cross_attn_only: if we want the score function to only have cross attention to the latent, better speed
        '''
        super(spectraTransceiverScore2stages, self).__init__()
        self.decoder = PerceiverDecoder2stages(
            bottleneck_dim,
            hidden_len,
                 1,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout, 
                 selfattn,
                 cross_attn_only
        )
        self.spectraEmbd = spectraEmbedding(model_dim, concat)
        self.model_dim = model_dim
        
    
    def forward(self, x, bottleneck, aux):
        '''
        Args:
            x: a dictionary of: 
                wavelength: wavelength of the spectra being taken [batch_size, spectra_length]
                phase: phase of the spectra being taken [batch_size, 1]
                mask: mask of spectra [batch_size, spectra_length]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            Decoded spectra of shape [batch_size, spectra_length]
        '''
        flux, wavelength, phase, mask = x['flux'] ,x['wavelength'], x['phase'], x['mask']
        x = self.spectraEmbd(wavelength, flux, phase)
        if aux is not None:
            aux = torch.cat((x[:, -1,:][:, None, :], aux), axis = 1) # aux has original aux (diffusion time usually) and phase embedding
        else:
            aux = x[:, -1,:][:, None, :]
        x = x[:, :-1, :]
        return self.decoder(bottleneck, x, aux, mask).squeeze(-1) 

# this will generate bottleneck, in encoder
class spectraTransceiverEncoder(nn.Module):
    def __init__(self, bottleneck_length,
                 bottleneck_dim,
                 model_dim = 32, 
                 num_heads = 4, 
                 num_layers = 4,
                 ff_dim = 32, 
                 dropout = 0.1, 
                 selfattn = False, 
                 concat = True):
        '''
        Transceiver encoder for spectra, with cross attention pooling
        Args:
            bottleneck_length: spectra are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            bottleneck_dim: spectra are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the given spectra

        '''
        super(spectraTransceiverEncoder, self).__init__()
        self.encoder = PerceiverEncoder(bottleneck_length,
                 bottleneck_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, 
                 dropout, 
                 selfattn)
        
        self.spectraEmbd = spectraEmbedding(model_dim, concat)
        self.model_dim = model_dim
        self.bottleneck_length = bottleneck_length
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x):
        '''
        Args:
            x: a tuple of 
                flux: flux of the spectra being taken of shape [batch_size, spectra_length]
                wavelength: wavelength of the spectra being taken [batch_size, spectra_length]
                phase: phase of the spectra being taken [batch_size, 1]
                mask: which are not measured [batch_size, spectra_length]
        Return:
            Encoded spectra of shape [batch_size, bottleneck_length, bottleneck_dim]
        '''
        flux, wavelength, phase, mask = x['flux'], x['wavelength'], x['phase'], x['mask']
        x = self.spectraEmbd(wavelength, flux, phase)
        if mask is not None:
           # add a false at end to account for the added phase embd
           mask = torch.cat([mask, torch.zeros(mask.shape[0], 1).bool().to(mask.device) ], dim=1)
        x = self.encoder(x, mask)
        return x
        
# this will generate bottleneck, in encoder
class spectraTransceiverEncoder2stages(nn.Module):
    def __init__(self, bottleneck_length,
                 bottleneck_dim,
                 hidden_len = 256,
                 model_dim = 256, 
                 num_heads = 8, 
                 num_layers = 4,
                 ff_dim = 256, 
                 dropout = 0.1, 
                 selfattn = False, 
                 concat = True):
        '''
        Transceiver encoder for spectra, with cross attention pooling
        Args:
            bottleneck_length: spectra are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            bottleneck_dim: spectra are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the given spectra

        '''
        super(spectraTransceiverEncoder2stages, self).__init__()
        self.encoder = PerceiverEncoder2stages(bottleneck_length,
                 bottleneck_dim,
                 hidden_len,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, 
                 dropout, 
                 selfattn)
        
        self.spectraEmbd = spectraEmbedding(model_dim, concat)
        self.model_dim = model_dim
        self.bottleneck_length = bottleneck_length
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x):
        '''
        Args:
            x: a tuple of 
                flux: flux of the spectra being taken of shape [batch_size, spectra_length]
                wavelength: wavelength of the spectra being taken [batch_size, spectra_length]
                phase: phase of the spectra being taken [batch_size, 1]
                mask: which are not measured [batch_size, spectra_length]
        Return:
            Encoded spectra of shape [batch_size, bottleneck_length, bottleneck_dim]
        '''
        flux, wavelength, phase, mask = x['flux'], x['wavelength'], x['phase'], x['mask']
        x = self.spectraEmbd(wavelength, flux, phase)
        if mask is not None:
           # add a false at end to account for the added phase embd
           mask = torch.cat([mask, torch.zeros(mask.shape[0], 1).bool().to(mask.device) ], dim=1)
        x = self.encoder(x, mask)
        return x




class spectraTransceiverDecoder(nn.Module):
    def __init__(self,
                 bottleneck_dim,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 selfattn=False,
                 cross_attn_only = False
                 ):
        '''
        A transformer to decode something (latent) into spectra given time and band
        Args:
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            num_bands: number of bands, currently embedded as class
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the latent
            cross_attn_only: if we want the score function to only have cross attention to the latent, better speed
        '''
        super(spectraTransceiverDecoder, self).__init__()
        self.decoder = PerceiverDecoder(
            bottleneck_dim,
                 1,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout, 
                 selfattn,
                 cross_attn_only
        )
        self.spectraEmbd = wavelengthphaseEmbedding(model_dim)
        self.model_dim = model_dim
        
    
    def forward(self, x, bottleneck, aux):
        '''
        Args:
            x: a dictionary of: 
                wavelength: wavelength of the spectra being taken [batch_size, spectra_length]
                phase: phase of the spectra being taken [batch_size, 1]
                mask: mask of spectra [batch_size, spectra_length]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            Decoded spectra of shape [batch_size, spectra_length]
        '''
        wavelength, phase, mask = x['wavelength'], x['phase'], x['mask']
        x = self.spectraEmbd(wavelength, phase)
        if aux is not None:
            aux = torch.cat((x[:, -1,:][:, None, :], aux), axis = 1) # aux has original aux (diffusion time usually) and phase embedding
        else:
            aux = x[:, -1,:][:, None, :]
        x = x[:, :-1, :]
        return self.decoder(bottleneck, x, aux, mask).squeeze(-1) 


class spectraTransceiverDecoder2stages(nn.Module):
    def __init__(self,
                 bottleneck_dim,
                 hidden_len = 256,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 selfattn=False,
                 cross_attn_only = False
                 ):
        '''
        A transformer to decode something (latent) into spectra given time and band
        Args:
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            num_bands: number of bands, currently embedded as class
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the latent
            cross_attn_only: if we want the score function to only have cross attention to the latent, better speed
        '''
        super(spectraTransceiverDecoder2stages, self).__init__()
        self.decoder = PerceiverDecoder2stages(
            bottleneck_dim,
            hidden_len,
                 1,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout, 
                 selfattn,
                 cross_attn_only
        )
        self.spectraEmbd = wavelengthphaseEmbedding(model_dim)
        self.model_dim = model_dim
        
    
    def forward(self, x, bottleneck, aux):
        '''
        Args:
            x: a dictionary of: 
                wavelength: wavelength of the spectra being taken [batch_size, spectra_length]
                phase: phase of the spectra being taken [batch_size, 1]
                mask: mask of spectra [batch_size, spectra_length]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            Decoded spectra of shape [batch_size, spectra_length]
        '''
        wavelength, phase, mask = x['wavelength'], x['phase'], x['mask']
        x = self.spectraEmbd(wavelength, phase)
        if aux is not None:
            aux = torch.cat((x[:, -1,:][:, None, :], aux), axis = 1) # aux has original aux (diffusion time usually) and phase embedding
        else:
            aux = x[:, -1,:][:, None, :]
        x = x[:, :-1, :]
        return self.decoder(bottleneck, x, aux, mask).squeeze(-1) 
    
    
    
    
    
class spectraTransceiverMAEDecoder(nn.Module):
    def __init__(self,
                 bottleneck_dim,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 selfattn=False,
                 concat = True,
                 cross_attn_only = False
                 ):
        '''
        A transformer to decode something (latent) into spectra given time and band
        Args:
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            num_bands: number of bands, currently embedded as class
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the latent
            cross_attn_only: if we want the score function to only have cross attention to the latent, better speed
        '''
        super(spectraTransceiverMAEDecoder, self).__init__()
        self.decoder = PerceiverDecoder(
            bottleneck_dim,
                 1,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout, 
                 selfattn,
                 cross_attn_only
        )
        self.spectraEmbd = spectraEmbedding(model_dim, concat)
        self.model_dim = model_dim
        self.maskembd = nn.Parameter(torch.randn(1, 1, model_dim))
        
    
    def forward(self, x, bottleneck, maemask, aux = None):
        '''
        Args:
            x: a dictionary of: 
                wavelength: wavelength of the spectra being taken [batch_size, spectra_length]
                phase: phase of the spectra being taken [batch_size, 1]
                mask: mask of spectra [batch_size, spectra_length]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            Decoded spectra of shape [batch_size, spectra_length]
        '''
        flux, wavelength, phase, mask = x['flux'], x['wavelength'], x['phase'], x['mask']
        x = self.spectraEmbd(wavelength, flux, phase)
        if aux is not None:
            aux = torch.cat((x[:, -1,:][:, None, :], aux), axis = 1) # aux has original aux (diffusion time usually) and phase embedding
        else:
            aux = x[:, -1,:][:, None, :]
        x = x[:, :-1, :]
        #breakpoint()
        maemask = maemask[:, :, None].expand(-1,-1,x.shape[-1])
        maskembd = self.maskembd.expand(x.shape[0], x.shape[1], -1)
        x = torch.where(maemask, maskembd, x)
        return self.decoder(bottleneck, x, aux, mask).squeeze(-1) 


class spectraTransceiverMAEDecoder2stages(nn.Module):
    def __init__(self,
                 bottleneck_dim,
                 hidden_len = 256,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 concat = True, 
                 selfattn=False,
                 cross_attn_only = False
                 ):
        '''
        A transformer to decode something (latent) into spectra given time and band
        Args:
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            num_bands: number of bands, currently embedded as class
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the latent
            cross_attn_only: if we want the score function to only have cross attention to the latent, better speed
        '''
        super().__init__()
        self.decoder = PerceiverDecoder2stages(
            bottleneck_dim,
            hidden_len,
                 1,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout, 
                 selfattn,
                 cross_attn_only
        )
        self.spectraEmbd = spectraEmbedding(model_dim, concat)
        self.model_dim = model_dim
        self.maskembd = nn.Parameter(torch.randn(1, 1, model_dim))
        
    
    def forward(self, x, bottleneck, maemask, aux):
        '''
        Args:
            x: a dictionary of: 
                wavelength: wavelength of the spectra being taken [batch_size, spectra_length]
                phase: phase of the spectra being taken [batch_size, 1]
                mask: mask of spectra [batch_size, spectra_length]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            Decoded spectra of shape [batch_size, spectra_length]
        '''
        flux, wavelength, phase, mask = x['flux'], x['wavelength'], x['phase'], x['mask']
        x = self.spectraEmbd(wavelength, flux, phase)
        if aux is not None:
            aux = torch.cat((x[:, -1,:][:, None, :], aux), axis = 1) # aux has original aux (diffusion time usually) and phase embedding
        else:
            aux = x[:, -1,:][:, None, :]
        x = x[:, :-1, :]
        #breakpoint()
        maemask = maemask[:, :, None].expand(-1,-1,x.shape[-1])
        maskembd = self.maskembd.expand(x.shape[0], x.shape[1], -1)
        x = torch.where(maemask, maskembd, x)
        return self.decoder(bottleneck, x, aux, mask).squeeze(-1) 