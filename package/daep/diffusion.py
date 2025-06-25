import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm


# from https://github.com/w86763777/pytorch-ddpm

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, beta_1 = 1e-4, beta_T = 0.02, T = 1000):
        super().__init__()

        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, model, x_0, cond = None, name = "flux"):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0[name].shape[0],), device=x_0[name].device)
        noise = torch.randn_like(x_0[name]).to(x_0[name].device)
        x_t = copy.deepcopy(x_0)
        x_t[name] = (
            extract(self.sqrt_alphas_bar, t, x_0[name].shape) * x_0[name] +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0[name].shape) * noise)
        #breakpoint()
        loss = F.mse_loss(model(x_t, t.float()[:, None], cond), noise, reduction='none')
        del x_t
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, beta_1, beta_T, T,
                 mean_type='epsilon', var_type='fixedsmall'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.T = T
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t, name = "flux"):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t[name].shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t[name].shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t[name].shape) * x_t[name]
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t[name].shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps, name = "flux"):
        assert x_t[name].shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t[name].shape) * x_t[name] -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t[name].shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev, name):
        assert x_t[name].shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t[name].shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t[name].shape) * x_t[name]
        )

    def p_mean_variance(self, model, x_t, t, cond = None, name = "flux"):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t[name].shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = model(x_t, t.float()[:, None], cond)
            x_0 = self.predict_xstart_from_xprev(x_t, t.float(), xprev=x_prev, name = name)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = model(x_t, t.float()[:, None], cond)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t, name)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = model(x_t, t.float()[:, None], cond)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps, name = name)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t, name)
        else:
            raise NotImplementedError(self.mean_type)

        return model_mean, model_log_var

    def forward(self, model, x_T, cond = None, name = "flux"):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in tqdm(reversed(range(self.T))):
            t = x_t[name].new_ones([x_T[name].shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(model=model, x_t=x_t, t=t, cond = cond)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t[name])
            else:
                noise = 0
            x_t[name] = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return x_0
    
    def ddim_sample(self, model, x_T, cond=None, name="flux", eta=0.0, steps=None):
        """
        DDIM sampling loop. Deterministic if eta=0.
        Args:
            model: noise-predicting model
            x_T: dictionary with keys like {'flux': tensor}
            eta: 0.0 for deterministic DDIM, >0.0 adds noise
            cond: conditional context
            name: key to use in x_T dictionary
            steps: number of DDIM steps (defaults to full T)
        Returns:
            x_0: dictionary with denoised sample at x_0[name]
        """
        assert self.mean_type == 'epsilon', "DDIM requires epsilon prediction"
        if steps is None:
            steps = self.T

        device = x_T[name].device
        betas = self.betas
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t

        step_indices = torch.linspace(0, self.T - 1, steps, dtype=torch.long, device=device)
        x_t = {k: v.clone() for k, v in x_T.items()}  # deep copy

        for i in tqdm(reversed(range(steps))):
            t = step_indices[i].long()
            t_batch = t.expand(x_t[name].shape[0]).to(device)

            # Predict epsilon (noise)
            eps = model(x_t, t_batch.float().unsqueeze(1), cond)

            alpha_bar_t = extract(alphas_bar, t_batch, x_t[name].shape)
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

            # Predict x0
            x0_pred = (x_t[name] - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar

            if i == 0:
                x_t[name] = x0_pred
                break

            # Next timestep
            t_prev = step_indices[i - 1].long()
            t_prev_batch = t_prev.expand(x_t[name].shape[0]).to(device)
            alpha_bar_prev = extract(alphas_bar, t_prev_batch, x_t[name].shape)

            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) *
                (1 - alpha_bar_t / alpha_bar_prev)
            )

            noise = torch.randn_like(x_t[name]) if eta > 0 else 0.0

            # DDIM update rule
            x_t[name] = (
                torch.sqrt(alpha_bar_prev) * x0_pred +
                torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps +
                sigma * noise
            )

        return x_t


