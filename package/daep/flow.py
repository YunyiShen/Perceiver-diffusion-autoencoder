import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def extract(a, t, x_shape):
    # Utility function to index into a buffer and reshape
    out = a.gather(-1, t).reshape(-1, *((1,) * (len(x_shape) - 1)))
    return out

class GaussianFlowMatchingTrainer(nn.Module):
    def __init__(self, beta_1=1e-4, beta_T=0.02, T=1000):
        super().__init__()
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, model, x_0, cond=None):
        if isinstance(x_0, tuple):
            x_0_val = x_0[0]
        else:
            x_0_val = x_0

        B = x_0_val.shape[0]
        device = x_0_val.device
        t = torch.randint(0, self.T, (B,), device=device)

        beta_t = extract(self.betas, t, x_0_val.shape)
        alpha_bar_t = extract(self.alphas_bar, t, x_0_val.shape)
        sqrt_alpha_bar_t = extract(self.sqrt_alphas_bar, t, x_0_val.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_bar, t, x_0_val.shape)

        noise = torch.randn_like(x_0_val)
        x_t_val = sqrt_alpha_bar_t * x_0_val + sqrt_one_minus_alpha_bar_t * noise

        if isinstance(x_0, tuple):
            x_t = list(copy.deepcopy(x_0))
            x_t[0] = x_t_val
        else:
            x_t = x_t_val

        # Compute target velocity for flow matching
        v_target = - beta_t / sqrt_one_minus_alpha_bar_t * (x_t_val - sqrt_alpha_bar_t * x_0_val)

        v_pred = model(x_t, t.float(), cond)

        return F.mse_loss(v_pred, v_target, reduction='none')