import torch
from torch import nn
from torch.nn import functional as F
from daep.util_layers import *
from daep.Perceiver import PerceiverEncoder, PerceiverDecoder, PerceiverDecoder2stages, PerceiverEncoder2stages



###############################
# Transceivers for LC data
###############################
class timebandEmbedding(nn.Module):
    def __init__(self, num_bands = 6, model_dim = 32, sinpos = True, fourier = False):
        super(timebandEmbedding, self).__init__()
        self.use_fourier = fourier
        self.use_sinpos = sinpos
        self.time_embd_fourier = learnable_fourier_encoding(model_dim)
        self.time_embd_sinpos = SinusoidalMLPPositionalEmbedding(model_dim)
        self.bandembd = nn.Embedding(num_bands, model_dim) if num_bands > 1 else None
    
    def forward(self, time, band):
        if self.use_fourier and self.use_sinpos:
            time_embd = self.time_embd_fourier(time) + self.time_embd_sinpos(time)
        elif self.use_fourier:
            time_embd = self.time_embd_fourier(time)
        elif self.use_sinpos:
            time_embd = self.time_embd_sinpos(time)
        else:
            time_embd = time
        return time_embd + (self.bandembd(band) if self.bandembd is not None else 0.0)

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
    def __init__(self, num_bands = 6, model_dim = 32, sinpos = True, fourier = False):
        super(photometryEmbeddingConcat, self).__init__()
        self.use_fourier = fourier
        self.use_sinpos = sinpos
        self.time_embd_fourier = learnable_fourier_encoding(model_dim)
        self.time_embd_sinpos = SinusoidalMLPPositionalEmbedding(model_dim)
        
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
        if self.use_fourier and self.use_sinpos:
            time_embd = self.time_embd_fourier(time) + self.time_embd_sinpos(time)
        elif self.use_fourier:
            time_embd = self.time_embd_fourier(time)
        elif self.use_sinpos:
            time_embd = self.time_embd_sinpos(time)
        else:
            time_embd = time
        return self.lcfc(torch.cat((self.fluxfc(flux[:, :, None]), time_embd), axis = -1)) + (self.bandembd(band) if self.bandembd is not None else 0.0)


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
                 sinpos = True,
                 fourier = False, # use learnable fourier for time embedding?
                 output_uncertainty = False
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
            output_uncertainty: if True, output both prediction and log-variance uncertainty
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
                 cross_attn_only,
                 output_uncertainty
        )
        if concat:
            self.photometry_embd = photometryEmbeddingConcat(num_bands, model_dim, sinpos, fourier)
        else:
            self.photometry_embd = photometryEmbedding(num_bands, model_dim, sinpos, fourier)
        self.model_dim = model_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_uncertainty = output_uncertainty
    
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
            if output_uncertainty=False: flux of the decoded photometry, [batch_size, photometry_length]
            if output_uncertainty=True: tuple of (prediction, log_variance) each of shape [batch_size, photometry_length]
        '''
        flux, time, mask = x['flux'], x['time'], x['mask']
        band = x.get("band")
        x = self.photometry_embd(flux, time, band)
        #breakpoint()
        output = self.Decoder(bottleneck, x, aux, mask)
        if self.output_uncertainty:
            pred, logvar = output
            return pred.squeeze(-1), logvar.squeeze(-1)
        else:
            return output.squeeze(-1)



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
                 sinpos = True,
                 fourier = False,
                 output_uncertainty = False
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
            output_uncertainty: if True, output both prediction and log-variance uncertainty
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
                 cross_attn_only,
                 output_uncertainty
        )
        if concat:
            self.photometry_embd = photometryEmbeddingConcat(num_bands, model_dim, sinpos, fourier)
        else:
            self.photometry_embd = photometryEmbedding(num_bands, model_dim, sinpos, fourier)
        self.model_dim = model_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_uncertainty = output_uncertainty
    
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
            if output_uncertainty=False: flux of the decoded photometry, [batch_size, photometry_length]
            if output_uncertainty=True: tuple of (prediction, log_variance) each of shape [batch_size, photometry_length]
        '''
        flux, time, mask = x['flux'], x['time'], x['mask']
        band = x.get("band")
        x = self.photometry_embd(flux, time, band)
        #breakpoint()
        output = self.Decoder(bottleneck, x, aux, mask)
        if self.output_uncertainty:
            pred, logvar = output
            return pred.squeeze(-1), logvar.squeeze(-1)
        else:
            return output.squeeze(-1)


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
                 sinpos = True,
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
            self.photometry_embd = photometryEmbeddingConcat(num_bands, model_dim, sinpos, fourier)
        else:
            self.photometry_embd = photometryEmbedding(num_bands, model_dim, sinpos, fourier)
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

# self.classifier = LCC(emb_d=transformer_kwargs['emb'], num_heads=transformer_kwargs['heads'], layers=transformer_kwargs['layers'], dropout_p=transformer_kwargs['dropout_p'], ffn_d=transformer_kwargs['hidden'], num_classes=transformer_kwargs['num_classes'])

"""
This is an encoder that uses the TESS-Transformer architecture to encode photometry.
"""

class TESSTransformerPhotometryEncoder(nn.Module):
    def __init__(self,
                 num_bands, 
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
        super(TESSTransformerPhotometryEncoder,self).__init__()
        bottleneck_len = 1
        self.bottleneck_len = bottleneck_len
        self.bottleneck_dim = bottleneck_dim
        self.emb_d = model_dim
        
        self.encoder = TESSTransformerPhotometryEncoderBase(self.emb_d, num_heads, num_layers, dropout, ff_dim)
        

    def forward(self, x,t, mask=None):
        x = self.encoder(x, t, mask)
        return x

class TESSTransformerPhotometryEncoderBase(nn.Module):
    def __init__(self, emb_d, num_heads, layers, dropout_p, ffn_d):
        super(TESSTransformerPhotometryEncoderBase,self).__init__()
        self.lsatt_conv_layers  = nn.ModuleList([LSATT_CONV(emb_d=emb_d, num_heads=num_heads, dropout_p=dropout_p, ffn_d=ffn_d) for _ in range(layers)])
        self.embed1 = nn.Linear(1, emb_d)

    def forward(self, x,t, mask=None):
        x = self.embed1(x.unsqueeze(2))
        for lsatt_conv in self.lsatt_conv_layers:
            x = lsatt_conv(x, t)
        # The dimension of x at this point is (batch_size, sequence_length, emb_d)
        # where:
        #   batch_size: number of samples in the batch
        #   sequence_length: number of time steps in the input sequence
        #   emb_d: embedding dimension (as set by emb_d argument)
        
        return x

class LSATT_CONV(nn.Module):
    def __init__(self, emb_d, num_heads, dropout_p, ffn_d):
        super(LSATT_CONV, self).__init__()
        self.MHSA = TransformerLayer(emb_d=emb_d, ffn_d=ffn_d, num_heads=num_heads, dropout_p=dropout_p)
        self.conv = glu_convolution_layer(emb_d, dropout_p=dropout_p)
        self.conv2 = convolution_layer_lstm(emb_d, dropout_p=dropout_p)
        self.norm1 = nn.LayerNorm(normalized_shape=emb_d)
        self.lstm = nn.LSTM(emb_d,emb_d, batch_first=True, dropout=dropout_p, num_layers=2, bidirectional=True)
        self.norm2 = nn.LayerNorm(normalized_shape=emb_d)   

    def forward(self, x, t, mask=None):
        resid_x = x
        x = self.lstm(x)[0]
        x = self.conv2(x)
        x = x + resid_x
        x = self.norm2(x)
        x = self.MHSA(x,t, mask)
        resid_x = x
        x = self.conv(x)
        x = x + resid_x
        x = self.norm1(x)
        return x

def scaled_dot_prod(q,k,v,mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q,k.transpose(-1,-2))*math.sqrt((1/d_k))
    if mask is not None:
        print('error')
        scaled += mask
    soft = F.softmax(scaled, dim=-1)
    attention = torch.matmul(soft, v)
    return attention

class LSATTTimePositionalEncoding(nn.Module):
    """ Time encodings for Transformer. 
    """

    def __init__(self, d_emb):
        """
        Inputs
            d_emb - Dimensionality when projecting to the fourier feature basis.
        """
        super().__init__()
        self.d_emb = d_emb

    def forward(self, t):
        batch_size = t.shape[0]
        max_len = t.shape[1]
        pe = torch.zeros(batch_size, max_len, self.d_emb).to(t.device)  # (B, T, D)
        div_term = torch.exp(torch.arange(0, self.d_emb, 2).float() * (-math.log(10000.0) / self.d_emb))[None, None, :].to(t.device)  # (1, 1, D / 2)
        t = t.unsqueeze(2)  # (B, 1, T)
        pe[:, :, 0::2] = torch.sin((t / div_term)*(self.d_emb/max_len)) # (B, T, D / 2)
        pe[:, :, 1::2] = torch.cos((t / div_term)*(self.d_emb/max_len))  # (B, T, D / 2)
        return pe  # (B, T, D)

class MHSA(nn.Module):
    def __init__(self, emb_d, num_heads):
        super(MHSA,self).__init__()
        self.emb_d = emb_d
        self.num_heads = num_heads
        self.head_d = emb_d // num_heads
        self.QKV = nn.Linear(emb_d, 3 * emb_d)
        self.ffn = nn.Linear(emb_d, emb_d)
    def forward(self, x, t, mask=None):
        batch_size, max_len, dims = x.shape
        qkv = self.QKV(x)
        qkv = qkv.reshape(batch_size, max_len, self.num_heads, 3 * self.head_d)
        qkv = qkv.permute(0,2,1,3)
        q,k,v = qkv.chunk(3, dim=-1)
        attention = scaled_dot_prod(q,k,v,mask)
        attention = attention.reshape(batch_size, max_len, self.num_heads * self.head_d)
        output = self.ffn(attention)
        return output

class ConvFeedForward(nn.Module):
    def __init__(self, emb_d, hidden, dropout_p=0.1):
        super(ConvFeedForward, self).__init__()
        self.convo1 = nn.Conv1d(emb_d, hidden, kernel_size=1, bias=False)
        self.convo2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding='same')
        self.convo3 = nn.Conv1d(hidden, emb_d, kernel_size=1, bias=False)
        self.swish1 = nn.SiLU()
        self.swish2 = nn.SiLU()
        self.norm = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(p=dropout_p)
    def forward(self,x):
        x = self.convo1(x.transpose(1,2))
        x = self.swish1(x)
        x = self.convo2(x)
        x = self.norm(x)
        x = self.swish2(x)
        x = self.dropout(x)
        x = self.convo3(x)
        x = x.transpose(1,2)
        return x
        
class TransformerLayer(nn.Module):
    def __init__(self, emb_d, num_heads, ffn_d, dropout_p):
        super(TransformerLayer,self).__init__()
        self.time_encoding = LSATTTimePositionalEncoding(emb_d)
        self.attention = MHSA(emb_d=emb_d, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(normalized_shape=emb_d)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.cffn = ConvFeedForward(emb_d, ffn_d, dropout_p=dropout_p)
        self.norm2 = nn.LayerNorm(normalized_shape=emb_d)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)

    def forward(self,x,t, mask=None):
        t = t - t[:, 0].unsqueeze(1)
        x = x + self.time_encoding(t)
        x = self.dropout1(x)
        residual_x = x
        x = self.attention(x, t, mask=None)
        x = self.dropout2(x)
        x = self.norm1(x + residual_x)
        residual_x = x
        x = self.cffn(x)
        x = self.dropout3(x)
        x = self.norm2(x + residual_x)
        return x

class glu_convolution_layer(nn.Module):
    def __init__(self, dims, dropout_p=0.3, kernel_size = 3, stride = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pointwise1 = nn.Conv1d(dims,4*dims,kernel_size=1, bias=False)
        self.GLU = nn.GLU(dim=1)
        self.convolution = nn.Conv1d(2*dims,2*dims, kernel_size=3, stride=1, padding='same')
        self.BatchNorm = nn.BatchNorm1d(2*dims)
        self.swish = nn.SiLU()
        self.swish0 = nn.SiLU()
        self.convolution1 = nn.Conv1d(2*dims,2*dims, kernel_size=7, stride=1, padding='same')
        self.BatchNorm1 = nn.BatchNorm1d(2*dims)
        self.swish01 = nn.SiLU()
        self.convolution2 = nn.Conv1d(2*dims,2*dims, kernel_size=15, stride=1, padding='same')
        self.BatchNorm2 = nn.BatchNorm1d(2*dims)
        self.swish02 = nn.SiLU()
        self.pointwise2 = nn.Conv1d(2*dims,dims,kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout1 = nn.Dropout(dropout_p)
        self.longrange = nn.Conv1d(2*dims,2*dims,kernel_size=111, stride=1, padding='same')
        self.batchNorm2 = nn.BatchNorm1d(2*dims)

    def forward(self, x):
        x =  self.dropout1(x)
        x = self.pointwise1(x.transpose(1,2))
        x = self.GLU(x)
        x = self.convolution(x)
        x = self.BatchNorm(x)
        x = self.swish0(x)
        x = self.convolution1(x)
        x = self.BatchNorm1(x)
        x = self.swish01(x)
        x = self.convolution2(x)
        x = self.BatchNorm2(x)
        x = self.swish02(x)
        x = self.longrange(x)
        x = self.batchNorm2(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.pointwise2(x)
        x = x.transpose(1,2)
        return x
        
class convolution_layer_lstm(nn.Module):
    def __init__(self, dims, dropout_p=0.3, kernel_size = 3, stride = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pointwise1 = nn.Conv1d(2*dims,4*dims,kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.convolution = nn.Conv1d(4*dims,4*dims, kernel_size=3, stride=1, padding='same')
        self.BatchNorm = nn.BatchNorm1d(4*dims)
        self.swish0 = nn.SiLU()
        self.pointwise2 = nn.Conv1d(4*dims,dims,kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout1 = nn.Dropout(dropout_p)
        

    def forward(self, x):
        x =  self.dropout1(x)
        x = self.pointwise1(x.transpose(1,2))
        x = self.relu(x)
        x = self.convolution(x)
        x = self.BatchNorm(x)
        x = self.swish0(x)
        x = self.dropout(x)
        x = self.pointwise2(x)
        x = x.transpose(1,2)
        return x
        