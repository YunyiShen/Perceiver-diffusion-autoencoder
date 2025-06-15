import torch
from torch import nn
from torch.nn import functional as F
from .util_layers import *
from .Perceiver import PerceiverEncoder, PerceiverDecoder



###############################
# Transceivers for spectra data
###############################
class timebandEmbedding(nn.Module):
    def __init__(self, num_bands = 6, model_dim = 32):
        super(timebandEmbedding, self).__init__()
        self.time_embd = SinusoidalMLPPositionalEmbedding(model_dim)
        self.bandembd = nn.Embedding(num_bands, model_dim)
    
    def forward(self, time, band):
        return self.time_embd(time) + self.bandembd(band)



class photometryEmbedding(nn.Module):
    def __init__(self, num_bands = 6, model_dim = 32):
        super(photometryEmbedding, self).__init__()
        self.time_band_embd = timebandEmbedding(num_bands, model_dim)
        self.fluxfc = nn.Linear(1, model_dim)

    def forward(self, flux, time, band):
        '''
        Args:
            flux: flux (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            time: time (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            band: band of the photometry being taken [batch_size, photometry_length]
        Return:
            encoding of size [batch_size, bottleneck_length, bottleneck_dim]

        '''
        return (self.fluxfc(flux[:, :, None]) + self.time_band_embd(time, band))


class photometryEmbeddingConcat(nn.Module):
    def __init__(self, num_bands = 6, model_dim = 32):
        super(photometryEmbeddingConcat, self).__init__()
        self.time_embd = SinusoidalMLPPositionalEmbedding(model_dim)
        self.bandembd = nn.Embedding(num_bands, model_dim)
        self.fluxfc = nn.Linear(1, model_dim)
        self.lcfc = MLP(model_dim * 3, model_dim, [model_dim])

    def forward(self, flux, time, band):
        '''
        Args:
            flux: flux (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            time: time (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            band: band of the photometry being taken [batch_size, photometry_length]
        Return:
            encoding of size [batch_size, bottleneck_length, bottleneck_dim]

        '''
        return self.lcfc(torch.cat((self.fluxfc(flux[:, :, None]), self.time_embd(time) , self.bandembd(band)), axis = -1))


class photometricTransceiverDecoder(nn.Module):
    def __init__(self, 
                 bottleneck_dim,
                 num_bands,
                 model_dim = 32,
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 donotmask=False,
                 selfattn=False
                 ):
        '''
        A transformer to decode something (latent) into photometry given time and band
        Args:
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            num_bands: number of bands, currently embedded as class
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            donotmask: should we ignore the mask when decoding?
            selfattn: if we want self attention to the latent
        '''
        super(photometricTransceiverDecoder, self).__init__()
        self.decoder = PerceiverDecoder(
            bottleneck_dim,
                 1,
                 model_dim, 
                 num_heads, 
                 ff_dim, 
                 num_layers,
                 dropout, 
                 selfattn
        )
        self.time_band_embd = timebandEmbedding(num_bands, model_dim)
        self.donotmask = donotmask
    
    def forward(self, x, bottleneck):
        '''
        Args:
            x: a tuple of 
                time: time of the photometry being taken [batch_size, photometry_length]
                band: band of the photometry being taken [batch_size, photometry_length]
                mask: mask [batch_size, photometry_length]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            flux of the decoded photometry, [batch_size, photometry_length]
        '''
        time, band, mask = x
        if self.donotmask:
            mask = None
        x = self.time_band_embd(time, band)
        return self.decoder(bottleneck, x, None, mask).squeeze(-1)
         

# this will generate bottleneck, in encoder
class photometricTransceiverEncoder(nn.Module):
    def __init__(self,
                 num_bands, 
                 bottleneck_length,
                 bottleneck_dim,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32,
                 num_layers = 4,
                 dropout=0.1,
                 selfattn=False, 
                 concat = True):
        '''
        Transceiver encoder for photometry, with cross attention pooling
        Args:
            num_bands: number of bands, currently embedded as class
            bottleneck_length: LCs are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            bottleneck_dim: LCs are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the given LC
            concat: how to construct flux, band and time joint embedding. If True, we separately embedding them, concatenate at the last dimension then project using a small MLP to model dimension, otherwise they are separately embedded and added
        '''
        super(photometricTransceiverEncoder, self).__init__()
        self.encoder = PerceiverEncoder(bottleneck_length,
                 bottleneck_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, 
                 dropout, 
                 selfattn)
        if concat:
            self.photometry_embd = photometryEmbeddingConcat(num_bands, model_dim)
        else:
            self.photometry_embd = photometryEmbedding(num_bands, model_dim)

    def forward(self, x):
        '''
        Args:
            flux: flux (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            time: time (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            band: band of the photometry being taken [batch_size, photometry_length]
        Return:
            encoding of size [batch_size, bottleneck_length, bottleneck_dim]

        '''
        flux, time, band, mask = x
        x = self.photometry_embd(flux, time, band)
        return self.encoder(x, mask) 
        
