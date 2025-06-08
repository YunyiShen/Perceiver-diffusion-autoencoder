import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from .util_layers import SinusoidalMLPPositionalEmbedding
from .mmd import RBF, MMD

class basedaepscore(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion_time_embd = SinusoidalMLPPositionalEmbedding(decoder.model_dim)
    
    def encode(x, cond = None, mask = None):
        raise NotImplementedError
    
    def decode(x, cond = None, mask = None):
        raise NotImplementedError
    
    def forward(x, cond = None, mask = None):



class basedaep(nn.Module):
    def __init__(self, encoder, decoder, ):
