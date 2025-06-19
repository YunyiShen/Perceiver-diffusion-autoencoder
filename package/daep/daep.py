import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from .util_layers import SinusoidalMLPPositionalEmbedding
from .mmd import RBF, MMD
import torch.distributions as dist

class unimodaldaep(nn.Module):
    def __init__(self, encoder, score, MMD = MMD(),
                prior = dist.Laplace, regularize = 0.01, 
                beta_1 = 1e-4, beta_T = 0.02, 
                T = 1000
                ):
        super().__init__()
        self.encoder = encoder
        self.score_model = score
        self.diffusion_time_embd = SinusoidalMLPPositionalEmbedding(score.model_dim)
        self.diffusion_trainer = GaussianDiffusionTrainer(beta_1, beta_T, T)
        self.diffusion_sampler = GaussianDiffusionSampler(beta_1, beta_T, T)
        self.MMD = MMD
        self.latent_len = encoder.bottleneck_length
        self.latent_dim = encoder.bottleneck_dim
        self.prior = prior
        self.prior_param = nn.ParameterList([
            nn.Parameter(torch.zeros(self.latent_len, self.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.ones(self.latent_len, self.latent_dim), requires_grad=True)  # logvar
        ])
        self.regularize = regularize
    
    def encode(x):
        return self.encoder(x)
    
    def score(xt, t, cond = None):
        if cond is not None:
            aux = self.diffusion_time_embd(t)
        else:
            cond = self.diffusion_time_embd(t)
            aux = None
        return self.score_model(xt, cond, aux) # score model take xt, cond and aux, cond is always assume to be not None
    
    def forward(x):
        z = self.encode(x)
        qz_x = self.prior(*self.prior_params).rsample(torch.size([z.shape[0]]))

        mmd_loss = self.regularize * self.MMD(z.reshape[z.shape[0], -1], qz_x.reshape(z.shape[0], -1))
        score = self.score_model(xt, t, z)

        score_matching_loss = self.diffusion_trainer(self.score_model, x, z).mean()

        return mmd_loss + score_matching_loss
