import torch
from torch import nn
from torch.nn import functional as F
from .util_layers import *
from .Perceiver import PerceiverEncoder, PerceiverDecoder, PerceiverDecoder2stages, PerceiverEncoder2stages



###############################
# Transceivers for LC data
###############################
class timebandEmbedding(nn.Module):
    def __init__(self, num_bands = 6, model_dim = 32, fourier = False):
        super(timebandEmbedding, self).__init__()
        if fourier:
            self.time_embd = learnable_fourier_encoding(model_dim)
        else:
            self.time_embd = SinusoidalMLPPositionalEmbedding(model_dim)
        self.bandembd = nn.Embedding(num_bands, model_dim) if num_bands > 1 else None
    
    def forward(self, time, band):
        return self.time_embd(time) + (self.bandembd(band) if self.bandembd is not None else 0.0)

class photometryEmbedding(nn.Module):
    def __init__(self, num_bands = 6, model_dim = 32, fourier = False):
        super(photometryEmbedding, self).__init__()
        self.time_band_embd = timebandEmbedding(num_bands, model_dim, fourier)
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
    def __init__(self, num_bands = 6, model_dim = 32, fourier = False):
        super(photometryEmbeddingConcat, self).__init__()
        if fourier:
            self.time_embd = learnable_fourier_encoding(model_dim)
        else:
            self.time_embd = SinusoidalMLPPositionalEmbedding(model_dim)
        self.bandembd = nn.Embedding(num_bands, model_dim) if num_bands > 1 else None
        self.fluxfc = nn.Linear(1, model_dim)
        self.lcfc = MLP(model_dim * 2, model_dim, [model_dim])

    def forward(self, flux, time, band):
        '''
        Args:
            flux: flux (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            time: time (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            band: band of the photometry being taken [batch_size, photometry_length]
        Return:
            encoding of size [batch_size, bottleneck_length, bottleneck_dim]

        '''
        return self.lcfc(torch.cat((self.fluxfc(flux[:, :, None]), self.time_embd(time)), axis = -1)) + (self.bandembd(band) if self.bandembd is not None else 0.0)


class photometricTransceiverScore(nn.Module):
    def __init__(self, 
                 bottleneck_dim,
                 num_bands,
                 model_dim = 32,
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 selfattn=False,
                 concat = True,
                 cross_attn_only = False,
                 fourier = False # use learnable fourier for time embedding?
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
        super(photometricTransceiverScore, self).__init__()
        self.Decoder = PerceiverDecoder(
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
        if concat:
            self.photometry_embd = photometryEmbeddingConcat(num_bands, model_dim, fourier)
        else:
            self.photometry_embd = photometryEmbedding(num_bands, model_dim, fourier)
        self.model_dim = model_dim
        self.bottleneck_dim = bottleneck_dim
    
    def forward(self, x, bottleneck, aux):
        '''
        Args:
            x: a dictionary of 
                flux: noisy photometry
                time: time of the photometry being taken [batch_size, photometry_length]
                band: band of the photometry being taken [batch_size, photometry_length]
                mask: mask [batch_size, photometry_length]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            flux of the decoded photometry, [batch_size, photometry_length]
        '''
        flux, time, mask = x['flux'], x['time'], x['mask']
        band = x.get("band")
        x = self.photometry_embd(flux, time, band)
        #breakpoint()
        return self.Decoder(bottleneck, x, aux, mask).squeeze(-1)



class photometricTransceiverScore2stages(nn.Module):
    def __init__(self, 
                 bottleneck_dim,
                 num_bands,
                 
                 model_dim = 32,
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 selfattn=False,
                 concat = True,
                 cross_attn_only = False,
                 hidden_len = 256,
                 fourier = False
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
        super(photometricTransceiverScore2stages, self).__init__()
        self.Decoder = PerceiverDecoder2stages(
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
        if concat:
            self.photometry_embd = photometryEmbeddingConcat(num_bands, model_dim, fourier)
        else:
            self.photometry_embd = photometryEmbedding(num_bands, model_dim, fourier)
        self.model_dim = model_dim
        self.bottleneck_dim = bottleneck_dim
    
    def forward(self, x, bottleneck, aux):
        '''
        Args:
            x: a dictionary of 
                flux: noisy photometry
                time: time of the photometry being taken [batch_size, photometry_length]
                band: band of the photometry being taken [batch_size, photometry_length]
                mask: mask [batch_size, photometry_length]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            flux of the decoded photometry, [batch_size, photometry_length]
        '''
        flux, time, mask = x['flux'], x['time'], x['mask']
        band = x.get("band")
        x = self.photometry_embd(flux, time, band)
        #breakpoint()
        return self.Decoder(bottleneck, x, aux, mask).squeeze(-1)


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
                 concat = True,
                 fourier = False
                 ):
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
            self.photometry_embd = photometryEmbeddingConcat(num_bands, model_dim, fourier)
        else:
            self.photometry_embd = photometryEmbedding(num_bands, model_dim, fourier)
        self.model_dim = model_dim
        self.bottleneck_length = bottleneck_length
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x):
        '''
        Args:
            flux: flux (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            time: time (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            band: band of the photometry being taken [batch_size, photometry_length]
        Return:
            encoding of size [batch_size, bottleneck_length, bottleneck_dim]

        '''
        flux, time, mask = x['flux'], x['time'],  x['mask']
        band = x.get("band")
        x = self.photometry_embd(flux, time, band)
        return self.encoder(x, mask) 
        



# this will generate bottleneck, in encoder
class photometricTransceiverEncoder2stages(nn.Module):
    def __init__(self,
                 num_bands, 
                 bottleneck_length,
                 bottleneck_dim,
                 hidden_len = 256,
                 model_dim = 256, 
                 num_heads = 8, 
                 ff_dim = 256,
                 num_layers = 4,
                 dropout=0.1,
                 selfattn=False, 
                 concat = True,
                 fourier = False
                 ):
        '''
        Transceiver encoder for photometry with two stage perceiver IO, with cross attention pooling
        Args:
            num_bands: number of bands, currently embedded as class
            bottleneck_length: LCs are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            bottleneck_dim: LCs are encoded as a sequence of size [bottleneck_length, bottleneck_dim]
            hidden_len: length of the hidden sequence in perceiver IO
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the given LC
            concat: how to construct flux, band and time joint embedding. If True, we separately embedding them, concatenate at the last dimension then project using a small MLP to model dimension, otherwise they are separately embedded and added
        '''
        super(photometricTransceiverEncoder2stages, self).__init__()
        self.encoder = PerceiverEncoder2stages(bottleneck_length,
                 bottleneck_dim,
                 hidden_len,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, 
                 dropout, 
                 selfattn)
        if concat:
            self.photometry_embd = photometryEmbeddingConcat(num_bands, model_dim, fourier)
        else:
            self.photometry_embd = photometryEmbedding(num_bands, model_dim, fourier)
        self.model_dim = model_dim
        self.bottleneck_length = bottleneck_length
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x):
        '''
        Args:
            flux: flux (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            time: time (potentially transformed) of the photometry being taken [batch_size, photometry_length]
            band: band of the photometry being taken [batch_size, photometry_length]
        Return:
            encoding of size [batch_size, bottleneck_length, bottleneck_dim]

        '''
        flux, time, mask = x['flux'], x['time'],  x['mask']
        band = x.get("band")
        x = self.photometry_embd(flux, time, band)
        return self.encoder(x, mask) 
        


class photometricTransceiverDecoder(nn.Module):
    def __init__(self, #photometry_length,
                 bottleneck_dim,
                 num_bands,
                 model_dim = 32,
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 fourier = False,
                 dropout=0.1, 
                 selfattn=False,
                 cross_attn_only = False
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
        super().__init__()
        self.Decoder = PerceiverDecoder(
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
        
        self.photometry_embd = timebandEmbedding(num_bands, model_dim, fourier)
        self.model_dim = model_dim
        self.bottleneck_dim = bottleneck_dim
    
    def forward(self, x, bottleneck, aux):
        '''
        Args:
            x: a dictionary of 
                flux: noisy photometry
                time: time of the photometry being taken [batch_size, photometry_length]
                band: band of the photometry being taken [batch_size, photometry_length]
                mask: mask [batch_size, photometry_length]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            flux of the decoded photometry, [batch_size, photometry_length]
        '''
        time, mask = x['time'], x['mask']
        band = x.get("band")
        x = self.photometry_embd(time, band)
        #breakpoint()
        return self.Decoder(bottleneck, x, aux, mask).squeeze(-1)
    
    


class photometricTransceiverMAEDecoder(nn.Module):
    def __init__(self, #photometry_length,
                 bottleneck_dim,
                 num_bands,
                 model_dim = 32,
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 fourier = False,
                 dropout=0.1, 
                 concat = True, 
                 selfattn=False,
                 cross_attn_only = False
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
        super().__init__()
        self.Decoder = PerceiverDecoder(
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
        
        if concat:
            self.photometry_embd = photometryEmbeddingConcat(num_bands, model_dim, fourier)
        else:
            self.photometry_embd = photometryEmbedding(num_bands, model_dim, fourier)
        self.model_dim = model_dim
        self.bottleneck_dim = bottleneck_dim
        self.maskembd = nn.Parameter(torch.randn(1, 1, model_dim))
    
    def forward(self, x, bottleneck, maemask, aux):
        '''
        Args:
            x: a dictionary of 
                flux: noisy photometry
                time: time of the photometry being taken [batch_size, photometry_length]
                band: band of the photometry being taken [batch_size, photometry_length]
                mask: mask [batch_size, photometry_length]
            bottleneck: bottleneck from the encoder [batch_size, bottleneck_length, bottleneck_dim]
        Return:
            flux of the decoded photometry, [batch_size, photometry_length]
        '''
        flux, time, mask = x['flux'], x['time'], x['mask']
        band = x.get("band")
        x = self.photometry_embd(flux, time, band)
        maemask = maemask[:, :, None].expand(-1,-1,x.shape[-1])
        maskembd = self.maskembd.expand(x.shape[0], x.shape[1], -1)
        x = torch.where(maemask, maskembd, x)
        #breakpoint()
        return self.Decoder(bottleneck, x, aux, mask).squeeze(-1)