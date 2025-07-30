import torch
from torch import nn
from torch.nn import functional as F
from .util_layers import TransformerBlock, singlelayerMLP, MLP# useful base layers


class PerceiverEncoder(nn.Module):
    def __init__(self, bottleneck_length,
                 bottleneck_dim,
                 model_dim, 
                 num_heads, 
                 num_layers,
                 ff_dim, 
                 dropout = 0.1, 
                 selfattn = False):
        '''
        Perceiver encoder, with cross attention pooling
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
        super(PerceiverEncoder, self).__init__()
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, model_dim))
        self.transformerblocks =  nn.ModuleList( [TransformerBlock(model_dim, 
                                                    num_heads, ff_dim, dropout, selfattn) 
                                                 for _ in range(num_layers)] )
        
        self.bottleneckfc = singlelayerMLP(model_dim, bottleneck_dim)
        self.model_dim = model_dim
        self.bottleneck_length = bottleneck_length
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x, mask = None):
        '''
        Arg:
            x: sequence representation to be encoded, assume to be of model dimension
            mask: attention mask
        Return:
            bottleneck representation of size [B, bottleneck_len, bottleneck_dim] 
        '''
        out = self.initbottleneck[None, :, :]
        out = out.repeat(x.shape[0], 1, 1)
        h = out
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, x, context_mask=mask)
        return self.bottleneckfc(out+h) # residual connection

class PerceiverDecoder(nn.Module):
    def __init__(self,
                 bottleneck_dim,
                 out_dim = 1,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 selfattn=False,
                 cross_attn_only = False,
                 ):
        '''
        A transformer to decode something (latent) into dimension out_dim
        Args:
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            out_dim: output dimension
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the latent
            cross_attn_only: if we want the query to only cross attend the latent
        '''

        super(PerceiverDecoder, self).__init__()
        self.transformerblocks = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout, selfattn, cross_attn_only) 
                                                    for _ in range(num_layers)] 
                                                )
        self.contextfc = MLP(bottleneck_dim, model_dim, [model_dim])
        self.outputfc = singlelayerMLP(model_dim, out_dim)
        self.model_dim = model_dim
        
    

    def forward(self, bottleneck, x, aux = None, mask = None):
        '''
        Arg:
            bottleneck: bottleneck representation
            x: initial sequence representation to be decoded, assume to be of model dimension
            aux: auxiliary token to be added to bottleneck, should have dimension [B, len, model_dim]
            mask: attention mask
        Return:
            bottleneck representation of size [B, bottleneck_len, bottleneck_dim] 
        '''
        h = x
        bottleneck = self.contextfc(bottleneck)
        if aux is not None:
            bottleneck = torch.concat([bottleneck, aux], dim=1)
        for transformerblock in self.transformerblocks:
            h = transformerblock(h, bottleneck, mask=mask)
        return self.outputfc(x + h)



class PerceiverEncoder2stages(nn.Module):
    def __init__(self, bottleneck_length,
                 bottleneck_dim,
                 hidden_len = 256,
                 model_dim = 32, 
                 num_heads = 4, 
                 num_layers = 4,
                 ff_dim = 32, 
                 dropout = 0.1, 
                 selfattn = False):
        '''
        Perceiver encoder, with cross attention pooling
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
        super(PerceiverEncoder2stages, self).__init__()
        self.initbottleneck = nn.Parameter(torch.randn(bottleneck_length, model_dim))
        
        self.init_hidden = nn.Parameter(torch.randn(1, hidden_len, model_dim) * 0.02)
        
        self.transformerblocks_input_to_hidden = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout, selfattn, False) 
                                                    for _ in range(num_layers)] 
                                                )
        self.transformerblocks_hidden_to_bottleneck = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout, False, False) 
                                                    for _ in range(num_layers)] 
                                                )
        
        self.bottleneckfc = singlelayerMLP(model_dim, bottleneck_dim)
        self.model_dim = model_dim
        self.bottleneck_length = bottleneck_length
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x, mask = None):
        '''
        Arg:
            x: sequence representation to be encoded, assume to be of model dimension
            mask: attention mask
        Return:
            bottleneck representation of size [B, bottleneck_len, bottleneck_dim] 
        '''
        out = self.initbottleneck[None, :, :]
        out = out.repeat(x.shape[0], 1, 1)
        h = out
        hidden = self.init_hidden.repeat(x.shape[0].shape[0],1,1)
        for transformerblock1, transformerblock2 in zip(self.transformerblocks_input_to_hidden, self.transformerblocks_hidden_to_bottleneck):
            hidden = transformerblock1(hidden, x, context_mask=mask)
            h = transformerblock2(h, hidden)
        return self.outputfc(out + h)


class PerceiverDecoder2stages(nn.Module):
    def __init__(self,
                 bottleneck_dim,
                 hidden_len = 256,
                 out_dim = 1,
                 model_dim = 32, 
                 num_heads = 4, 
                 ff_dim = 32, 
                 num_layers = 4,
                 dropout=0.1, 
                 selfattn=False,
                 cross_attn_only = False,
                 ):
        '''
        A transformer to decode something (latent) into dimension out_dim
        Args:
            bottleneck_dim: dimension of the thing you want to decode, should be a tensor [batch_size, bottleneck_length, bottleneck_dim]
            out_dim: output dimension
            model_dim: dimension the transformer should operate 
            num_heads: number of heads in the multiheaded attention
            ff_dim: dimension of the MLP hidden layer in transformer
            num_layers: number of transformer blocks
            dropout: drop out in transformer
            selfattn: if we want self attention to the latent
            cross_attn_only: if we want the query to only cross attend the latent
        '''

        super(PerceiverDecoder2stages, self).__init__()
        
        self.init_hidden = nn.Parameter(torch.randn(1, hidden_len, model_dim) * 0.02)
        
        self.transformerblocks_bottleneck_to_hidden = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout, selfattn, cross_attn_only) 
                                                    for _ in range(num_layers)] 
                                                )
        self.transformerblocks_hidden_to_out = nn.ModuleList( [TransformerBlock(model_dim, 
                                                 num_heads, ff_dim, dropout, False, True) 
                                                    for _ in range(num_layers)] 
                                                )
        self.contextfc = MLP(bottleneck_dim, model_dim, [model_dim])
        self.outputfc = singlelayerMLP(model_dim, out_dim)
        self.model_dim = model_dim
        
    

    def forward(self, bottleneck, x, aux = None, mask = None):
        '''
        Arg:
            bottleneck: bottleneck representation
            x: initial sequence representation to be decoded, assume to be of model dimension
            aux: auxiliary token to be added to bottleneck, should have dimension [B, len, model_dim]
            mask: attention mask
        Return:
            bottleneck representation of size [B, bottleneck_len, bottleneck_dim] 
        '''
        h = x
        hidden = self.init_hidden.repeat(bottleneck.shape[0],1,1)
        bottleneck = self.contextfc(bottleneck)
        if aux is not None:
            bottleneck = torch.concat([bottleneck, aux], dim=1)
        for transformerblock1, transformerblock2 in zip(self.transformerblocks_bottleneck_to_hidden, self.transformerblocks_hidden_to_out):
            hidden = transformerblock1(hidden, bottleneck)
            h = transformerblock2(x, hidden, mask = mask)
        return self.outputfc(x + h)



