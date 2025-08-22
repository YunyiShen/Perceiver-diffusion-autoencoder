import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from .util_layers import SinusoidalMLPPositionalEmbedding
from .mmd import RBF, MMD
import torch.distributions as dist
import random
import torch

class modalitywrapper(nn.Module): 
    '''
    Handy function to use unimodal daep to train cross modality inference model, e.g., 
    wrap a photometry encoder with this and a spectra decoder with this
    '''
    def __init__(self, net, modality):
        super().__init__()
        self.net = net
        self.modality = modality
    
    def forward(self, x):
        return self.net(x[self.modality])

class unimodaldaep(nn.Module):
    def __init__(self, encoder, score, MMD = None, name = "flux",
                prior = dist.Laplace, regularize = 0.0001, 
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
        if MMD is not None and prior is not None:
            self.prior = prior
            self.prior_params = nn.ParameterList([
                nn.Parameter(torch.zeros(self.latent_len, self.latent_dim), requires_grad=False),  # mu
                nn.Parameter(torch.ones(self.latent_len, self.latent_dim), requires_grad=True)  # logvar
            ])
        self.regularize = regularize
        self.name = name
    
    def encode(self, x):
        return self.encoder(x)
    
    def score(self, xt, t, cond = None):
        if cond is not None:
            aux = self.diffusion_time_embd(t)
        else:
            cond = self.diffusion_time_embd(t)
            aux = None
        return self.score_model(xt, cond, aux) # score model take xt, cond and aux, cond is always assume to be not None
    
    def forward(self, x):
        z = self.encode(x)
        if self.MMD is not None and self.prior is not None:
            qz_x = self.prior(*self.prior_params).rsample([z.shape[0]]).to(z.device)
            mmd_loss = self.regularize * self.MMD(z.reshape(z.shape[0], -1), qz_x.reshape(z.shape[0], -1))
        else:
            mmd_loss = 0.0
        score_matching_loss = self.diffusion_trainer(self.score, x, z, self.name).mean()

        return mmd_loss + score_matching_loss
    
    def sample(self, z, x_T, ddim = True, ddim_steps = 200):
        self.eval()
        with torch.no_grad():
            if ddim:
                return self.diffusion_sampler.ddim_sample(self.score, x_T, z, self.name, steps=ddim_steps)
            return self.diffusion_sampler(self.score, x_T, z, self.name)
    

    def reconstruct(self, x_0, ddim = True, ddim_steps = 200):
        name = self.name
        z = self.encode(x_0)
        noise = torch.randn_like(x_0[name]).to(x_0[name].device)
        x_t = x_0
        x_t[name] = noise
        return self.sample(z, x_t, ddim, ddim_steps)


def modality_drop(keys, p_drop=0.2, drop_all=False):
    """
    Randomly drops modalities from a batch during training.
    
    Args:
        keys: modality keys
        p_drop: probability of dropping each modality
        drop_all: if False, ensures at least one modality is retained
    Returns:
        a list of kept modalities
    """
    present_modalities = list(keys)

    # Decide for each modality whether to drop
    drop_decisions = {
        m: (random.random() < p_drop)
        for m in present_modalities
    }

    # Optionally ensure at least one modality is retained
    if not drop_all and all(drop_decisions.values()):
        keep_one = random.choice(present_modalities)
        drop_decisions[keep_one] = False

    return [m for m in present_modalities if not drop_decisions[m]]


class multimodaldaep(nn.Module):
    def __init__(self, tokenizers, encoder, scores, measurement_names, modality_weights = None,
                 modality_dropping_during_training = None,
                 persample_dropping_p = 0.,
                 beta_1 = 1e-4, beta_T = 0.02, 
                 T = 1000):
        '''
        Args:
            tokenizers: {modality: tokenizer} that should share the same out put dimension (can be different seqlen)
            encoder: a sngle perceiver encoder
            scores: {modality: score}
            modality_dropping_during_training: a callable making a copy of the data that conditioning will only be based on them, if it is None roll back to per sample modality dropping by random masking in encoder
        '''
        super().__init__()
        assert set(tokenizers.keys()) == set(scores.keys()) and set(tokenizers.keys()) == set(measurement_names.keys()), "modality keys have to match"
        self.modalities = [*tokenizers.keys()]
        modeldims = [score.model_dim for score in scores.values()] + [tokenizer.model_dim for tokenizer in tokenizers.values()] + [encoder.model_dim]
        assert min(modeldims) == max(modeldims), "model_dims have to match for this implementation"
        
        self.tokenizers = nn.ModuleDict(tokenizers)
        self.encoder = encoder
        self.scores = nn.ModuleDict(scores)
        self.names = measurement_names
        self.modality_dropping_during_training = modality_dropping_during_training
        
        self.model_dim = min(modeldims)
        
        self.diffusion_time_embd = SinusoidalMLPPositionalEmbedding(self.model_dim)
        self.diffusion_trainer = GaussianDiffusionTrainer(beta_1, beta_T, T)
        self.diffusion_sampler = GaussianDiffusionSampler(beta_1, beta_T, T)
        self.latent_len = encoder.bottleneck_length
        self.latent_dim = encoder.bottleneck_dim
        
        self.modalityEmbedding = nn.ParameterDict({key: nn.Parameter(torch.randn(1, 1, self.model_dim) * 0.02) for key in tokenizers.keys()})
        
        if modality_weights is None:
            modality_weights = {key: 1.0 for key in self.modalities}
        self.modality_weights = modality_weights
        self.persample_dropping_p = persample_dropping_p
    
    def make_modality_mask(self, tokens, p):
        """
        Args:
            tokens: list of [B, L_m, D] tensors, one per modality
            p: float or list/tuple/tensor of length num_modalities
               Drop probability for each modality.
        Returns:
            masked_tokens: list of masked modality tensors
            attn_mask: [B, 1, 1, total_seq_len] boolean mask (True=keep, False=mask)
        """
        device = tokens[0].device
        B = tokens[0].size(0)
        num_modalities = len(tokens)

        
        # Step 1: initial random drop decisions
        drop_trials = torch.rand(B, num_modalities, device=device) < p  # [B, M]

        # Step 2: ensure at least one modality remains
        all_dropped = drop_trials.all(dim=1)  # [B] True if all modalities dropped
        if all_dropped.any():
            # Randomly choose one modality to keep for these rows
            random_keep = torch.randint(
                low=0, high=num_modalities, size=(all_dropped.sum(),), device=device
            )
            for row_idx, keep_m in zip(all_dropped.nonzero(as_tuple=True)[0], random_keep):
                drop_trials[row_idx, keep_m] = False

        # Step 3: build keep masks per modality
        keep_masks = []
        for m, t in enumerate(tokens):
            keep_mask = ~drop_trials[:, m].unsqueeze(1)  # [B, 1]
            keep_mask = keep_mask.expand(-1, t.size(1))   # [B, L_m]
            keep_masks.append(keep_mask)

        # Step 5: concat keep masks → attention mask
        concat_keep_mask = torch.cat(keep_masks, dim=1)  # [B, total_seq_len]
        masked_tokens = [
            t.masked_fill(~km.unsqueeze(-1), 0.0)
            for t, km in zip(tokens, keep_masks)
        ]

        return masked_tokens, ~concat_keep_mask
    
    def get_modality_p(self, keys):
        """
        Args:
            keys: list of modality names (str), same order as tokens
            self.persample_dropping_p: float/int OR dict mapping modality key -> drop probability
        Returns:
            torch.tensor of shape [num_modalities] with probabilities
        """
        if isinstance(self.persample_dropping_p, (float, int)):
            # Scalar case → same for all
            p_list = [float(self.persample_dropping_p)] * len(keys)
        elif isinstance(self.persample_dropping_p, dict):
            # Dict case → fill missing with 0.0
            p_list = [float(self.persample_dropping_p.get(k, 0.0)) for k in keys]
        else:
            raise TypeError("self.persample_dropping_p must be float, int, or dict")

        return torch.tensor(p_list, dtype=torch.float32)
    
    def encode(self, x, keys = None):
        '''
        Here we assume the x has a multiple layer structure like
        {modality1: {flux: tensor, time: tensor, ...}, ...}
        
        '''
        keys = keys if keys is not None else x.keys()
        tokens = [self.tokenizers[key](x[key]) + self.modalityEmbedding[key] for key in keys]
        
        
        if self.modality_dropping_during_training is None and self.training:
            #breakpoint()
            p = self.get_modality_p(keys).to(tokens[0].device)
            tokens, modality_mask = self.make_modality_mask(tokens, p)
        else:
            modality_mask = None
        
        
        
        
        return self.encoder(torch.concat(tokens, axis = 1), mask = modality_mask)
    
    
    def get_score(self, key):
        def score(xt, t, cond = None):
            if cond is not None:
                aux = self.diffusion_time_embd(t)
            else:
                cond = self.diffusion_time_embd(t)
                aux = None
            return self.scores[key](xt, cond, aux)
        return score
    
    
    def forward(self, x):
        if self.modality_dropping_during_training is not None:
            z = self.encode(x, keys = self.modality_dropping_during_training(x.keys())) # modality dropping
        else:
            z = self.encode(x)
        #breakpoint()
        score_matching_loss = torch.cat([self.modality_weights[key] * \
                                            self.diffusion_trainer(
                                                self.get_score(key), 
                                                x[key], z, 
                                                self.names[key]).mean(axis = 0).flatten()  
                                         for key in x.keys()]).mean()

        return score_matching_loss
    
    
    def sample(self, z, x_T, score, name, ddim = True, ddim_steps = 200):
        self.eval()
        with torch.no_grad():
            if ddim:
                return self.diffusion_sampler.ddim_sample(score, x_T, z, name, steps=ddim_steps)
            return self.diffusion_sampler(score, x_T, z, name)
    
    
    

    def reconstruct(self, x_0, condition_keys = None, out_keys = None, ddim = True, ddim_steps = 200):
        if condition_keys is None:
            condition_keys = x_0.keys()
        if out_keys is None:
            out_keys = x_0.keys()
        z = self.encode(x_0, condition_keys)
        
        x_t = x_0
        res = {}
        
        for key in out_keys:
            noise = torch.randn_like(x_0[key][self.names[key]]).to(x_0[key][self.names[key]].device)
        
            x_t[key][self.names[key]] = noise
            res[key] = self.sample(z, x_t[key], self.get_score(key), self.names[key], ddim, ddim_steps)
        return res
    
    


#### cross modal inference #####

class crossmodaldaep(nn.Module):
    def __init__(self, encoder, score, 
                 source_modality = "photometry",
                 target_modality = "spectra",
                 name = "flux",
                 query_name = "wavelength",
                beta_1 = 1e-4, beta_T = 0.02, 
                T = 1000
                ):
        super().__init__()
        self.encoder = encoder
        self.score_model = score
        self.diffusion_time_embd = SinusoidalMLPPositionalEmbedding(score.model_dim)
        self.diffusion_trainer = GaussianDiffusionTrainer(beta_1, beta_T, T)
        self.diffusion_sampler = GaussianDiffusionSampler(beta_1, beta_T, T)
        self.source_modility = source_modality
        self.target_modality = target_modality
        self.latent_len = encoder.bottleneck_length
        self.latent_dim = encoder.bottleneck_dim
        self.name = name
        self.query_name = query_name
    
    def encode(self, x):
        return self.encoder(x)
    
    def score(self, xt, t, cond = None):
        if cond is not None:
            aux = self.diffusion_time_embd(t)
        else:
            cond = self.diffusion_time_embd(t)
            aux = None
        return self.score_model(xt, cond, aux) # score model take xt, cond and aux, cond is always assume to be not None
    
    def forward(self, x):
        z = self.encode(x[self.source_modility])
        score_matching_loss = self.diffusion_trainer(self.score, x[self.target_modality], 
                                                     z, self.name).mean()

        return score_matching_loss
    
    def sample(self, z, x_T, ddim = True, ddim_steps = 200):
        self.eval()
        with torch.no_grad():
            if ddim:
                return self.diffusion_sampler.ddim_sample(self.score, x_T, z, self.name, steps=ddim_steps)
            return self.diffusion_sampler(self.score, x_T, z, self.name)
    

    def inference(self, x_0, ddim = True, ddim_steps = 200):
        name = self.query_name
        z = self.encode(x_0[self.source_modility])
        noise = torch.randn_like(x_0[self.target_modality][name]).to(x_0[self.target_modality][name].device)
        x_t = x_0
        x_t[self.target_modality][self.name] = noise
        return self.sample(z, x_t[self.target_modality], ddim, ddim_steps)



