import torch
import torch.nn as nn
import torch.nn.functional as F

from daep.diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from daep.util_layers import SinusoidalMLPPositionalEmbedding, learnable_fourier_encoding
from daep.mmd import RBF, MMD, robust_mean_squared_error, mean_squared_error
import torch.distributions as dist
import random
import torch



class unimodaldaep(nn.Module):
    def __init__(self, encoder, score, MMD = None, name = "flux",
                prior = dist.Laplace, regularize = 0.0001, 
                beta_1 = 1e-4, beta_T = 0.02, 
                T = 1000, output_uncertainty = False,
                sinpos = True, fourier = False
                ):
        super().__init__()
        self.encoder = encoder
        self.score_model = score
        self.use_fourier = fourier
        self.use_sinpos = sinpos
        self.diffusion_time_embd_fourier = learnable_fourier_encoding(score.model_dim)
        self.diffusion_time_embd_sinpos = SinusoidalMLPPositionalEmbedding(score.model_dim)
        self.diffusion_trainer = GaussianDiffusionTrainer(beta_1, beta_T, T)
        self.diffusion_sampler = GaussianDiffusionSampler(beta_1, beta_T, T)
        self.MMD = MMD
        self.latent_len = encoder.bottleneck_length
        self.latent_dim = encoder.bottleneck_dim
        self.output_uncertainty = output_uncertainty
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
            if self.use_fourier and self.use_sinpos:
                aux = self.diffusion_time_embd_fourier(t) + self.diffusion_time_embd_sinpos(t)
            elif self.use_fourier:
                aux = self.diffusion_time_embd_fourier(t)
            elif self.use_sinpos:
                aux = self.diffusion_time_embd_sinpos(t)
            else:
                aux = t
        else:
            if self.use_fourier and self.use_sinpos:
                cond = self.diffusion_time_embd_fourier(t) + self.diffusion_time_embd_sinpos(t)
            elif self.use_fourier:
                cond = self.diffusion_time_embd_fourier(t)
            elif self.use_sinpos:
                cond = self.diffusion_time_embd_sinpos(t)
            else:
                cond = t
            aux = None
        return self.score_model(xt, cond, aux) # score model take xt, cond and aux, cond is always assume to be not None
    
    def forward(self, x):
        z = self.encode(x)
        if self.MMD is not None and self.prior is not None:
            qz_x = self.prior(*self.prior_params).rsample([z.shape[0]]).to(z.device)
            mmd_loss = self.regularize * self.MMD(z.reshape(z.shape[0], -1), qz_x.reshape(z.shape[0], -1))
        else:
            mmd_loss = 0.0
        
        # if self.output_uncertainty:
        #     # Use uncertainty-aware loss if uncertainties are available
        #     print(f"Trying to use uncertainty-aware loss")
        #     try:
        #         score_output = self.diffusion_trainer(self.score, x, z, self.name)
        #         if isinstance(score_output, tuple):
        #             pred, logvar = score_output
        #             print(f"logvar: {logvar}")
        #             uncertainty = x[self.name + '_err']
        #             target = x[self.name]  # The actual target values
        #             score_matching_loss = robust_mean_squared_error(target, pred, logvar, uncertainty).mean()
        #         else:
        #             score_matching_loss = score_output.mean()
        #     except KeyError:
        #         score_matching_loss = self.diffusion_trainer(self.score, x, z, self.name).mean()
        # else:
        # Compute the mean of the score matching loss while ignoring NaN values for numerical stability
        score_matching_losses = self.diffusion_trainer(self.score, x, z, self.name)
        # Print the number of NaN values in the score_matching_losses tensor for debugging purposes
        # num_nans = torch.isnan(score_matching_losses).sum().item()
        # if num_nans == 0:
        #     print(f"No NaNs in score_matching_losses")
        # print(f"Number of NaNs in score_matching_losses: {torch.isnan(score_matching_losses).sum().item()}")
        score_matching_loss = torch.nanmean(score_matching_losses)

        return mmd_loss + score_matching_loss
    
    def sample(self, z, x_T, ddim = True, ddim_steps = 200):
        self.eval()
        with torch.no_grad():
            if ddim:
                return self.diffusion_sampler.ddim_sample(self.score, x_T, z, self.name, steps=ddim_steps, output_uncertainty=self.output_uncertainty)
            return self.diffusion_sampler(self.score, x_T, z, self.name)
    

    def reconstruct(self, x_0, ddim = True, ddim_steps = 200):
        name = self.name
        z = self.encode(x_0)
        noise = torch.randn_like(x_0[name]).to(x_0[name].device)
        x_t = x_0
        x_t[name] = noise
        
        # Use the main sample method which now handles uncertainty output
        result = self.sample(z, x_t, ddim, ddim_steps)
        
        if self.output_uncertainty and isinstance(result, tuple):
            prediction, uncertainty = result
            return prediction, uncertainty
        else:
            return result


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
                 modality_dropping_during_training = lambda x: x,
                 beta_1 = 1e-4, beta_T = 0.02, 
                 T = 1000, output_uncertainty = False,
                 sinpos = True, fourier = False):
        '''
        Args:
            tokenizers: {modality: tokenizer} that should share the same out put dimension (can be different seqlen)
            encoder: a sngle perceiver encoder
            scores: {modality: score}
            modality_dropping_during_training: a callable making a copy of the data that conditioning will only be based on them
            output_uncertainty: if True, output both prediction and log-variance uncertainty for each modality
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
        self.output_uncertainty = output_uncertainty
        
        self.model_dim = min(modeldims)
        
        self.diffusion_time_embd_fourier = learnable_fourier_encoding(self.model_dim)
        self.diffusion_time_embd_sinpos = SinusoidalMLPPositionalEmbedding(self.model_dim)
        self.use_fourier = fourier
        self.use_sinpos = sinpos
        self.diffusion_trainer = GaussianDiffusionTrainer(beta_1, beta_T, T)
        self.diffusion_sampler = GaussianDiffusionSampler(beta_1, beta_T, T)
        self.latent_len = encoder.bottleneck_length
        self.latent_dim = encoder.bottleneck_dim
        
        self.modalityEmbedding = nn.ParameterDict({key: nn.Parameter(torch.randn(1, 1, self.model_dim) * 0.02) for key in tokenizers.keys()})
        
        if modality_weights is None:
            modality_weights = {key: 1.0 for key in self.modalities}
        self.modality_weights = modality_weights
    
    def encode(self, x, keys = None):
        '''
        Here we assume the x has a multiple layer structure like
        {modality1: {flux: tensor, time: tensor, ...}, ...}
        
        '''
        keys = keys if keys is not None else x.keys()
        tokens = [self.tokenizers[key](x[key]) + self.modalityEmbedding[key] for key in keys]
        
        
        
        return self.encoder(torch.concat(tokens, axis = 1))
    
    
    def get_score(self, key):
        def score(xt, t, cond = None):
            if cond is not None:
                if self.use_fourier and self.use_sinpos:
                    aux = self.diffusion_time_embd_fourier(t) + self.diffusion_time_embd_sinpos(t)
                elif self.use_fourier:
                    aux = self.diffusion_time_embd_fourier(t)
                elif self.use_sinpos:
                    aux = self.diffusion_time_embd_sinpos(t)
                else:
                    aux = t
            else:
                if self.use_fourier and self.use_sinpos:
                    cond = self.diffusion_time_embd_fourier(t) + self.diffusion_time_embd_sinpos(t)
                elif self.use_fourier:
                    cond = self.diffusion_time_embd_fourier(t)
                elif self.use_sinpos:
                    cond = self.diffusion_time_embd_sinpos(t)
                else:
                    cond = t
                aux = None
            return self.scores[key](xt, cond, aux)
        return score
    
    
    def forward(self, x):
        z = self.encode(x, keys = self.modality_dropping_during_training(x.keys())) # modality dropping
        #breakpoint()
        
        if self.output_uncertainty:
            # Use uncertainty-aware loss if uncertainties are available
            losses = []
            for key in x.keys():
                score_output = self.diffusion_trainer(self.get_score(key), x[key], z, self.names[key])
                if isinstance(score_output, tuple):
                    pred, logvar = score_output
                    try:
                        uncertainty = x[key + '_err']
                        target = x[key][self.names[key]]  # The actual target values
                        loss = robust_mean_squared_error(target, pred, logvar, uncertainty).mean(axis=0).flatten()
                    except KeyError:
                        loss = score_output.mean(axis=0).flatten()
                else:
                    loss = score_output.mean(axis=0).flatten()
                losses.append(self.modality_weights[key] * loss)
            score_matching_loss = torch.cat(losses).mean()
        else:
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
                return self.diffusion_sampler.ddim_sample(score, x_T, z, name, steps=ddim_steps, output_uncertainty=self.output_uncertainty)
            return self.diffusion_sampler(score, x_T, z, name)
    
    
    

    def reconstruct(self, x_0, condition_keys = None, out_keys = None, ddim = True, ddim_steps = 200):
        if condition_keys is None:
            condition_keys = x_0.keys()
        if out_keys is None:
            out_keys = x_0.keys()
        z = self.encode(x_0, condition_keys)
        
        x_t = x_0
        res = {}
        uncertainties = {} if self.output_uncertainty else None
        
        for key in out_keys:
            noise = torch.randn_like(x_0[key][self.names[key]]).to(x_0[key][self.names[key]].device)
        
            x_t[key][self.names[key]] = noise
            
            if self.output_uncertainty:
                result = self.sample(z, x_t[key], self.get_score(key), self.names[key], ddim, ddim_steps)
                if isinstance(result, tuple):
                    prediction, uncertainty = result
                    res[key] = prediction
                    uncertainties[key] = uncertainty
                else:
                    res[key] = result
            else:
                res[key] = self.sample(z, x_t[key], self.get_score(key), self.names[key], ddim, ddim_steps)
        
        if self.output_uncertainty and uncertainties:
            return res, uncertainties
        else:
            return res




class unimodaldaepclassifier(nn.Module):
    """
    Unimodal DAE classifier that uses an encoder and classifier for classification tasks.
    
    This class replaces the diffusion-based reconstruction with a classification head
    for predicting class labels from encoded representations.
    
    Parameters
    ----------
    encoder : nn.Module
        The encoder network that processes input data into latent representations
    classifier : nn.Module
        The classifier network that predicts class labels from latent representations
    MMD : callable, optional
        Maximum Mean Discrepancy function for regularization
    name : str, default="flux"
        Name identifier for the modality
    prior : torch.distributions.Distribution, optional
        Prior distribution for MMD regularization
    regularize : float, default=0.0001
        Regularization weight for MMD loss
    num_classes : int, optional
        Number of classes for classification (if classifier doesn't specify output dim)
    """
    def __init__(self, encoder, classifier, MMD = None, name = "flux",
                prior = dist.Laplace, regularize = 0.0001, 
                num_classes = None):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.MMD = MMD
        self.latent_len = encoder.bottleneck_length
        self.latent_dim = encoder.bottleneck_dim
        self.regularize = regularize
        self.name = name
        
        # Set up MMD regularization if provided
        if MMD is not None and prior is not None:
            self.prior = prior
            self.prior_params = nn.ParameterList([
                nn.Parameter(torch.zeros(self.latent_len, self.latent_dim), requires_grad=False),  # mu
                nn.Parameter(torch.ones(self.latent_len, self.latent_dim), requires_grad=True)  # logvar
            ])
    
    def forward(self, x, targets=None):
        """
        Forward pass through encoder and classifier.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor
        targets : torch.Tensor, optional
            Target class labels for loss computation
            
        Returns
        -------
        torch.Tensor or tuple
            If targets provided: (classification_loss, mmd_loss, total_loss)
            If no targets: class predictions
        """
        z = self.encoder(x)
        
        # Handle different classifier input requirements
        if hasattr(self.classifier, 'mlp'):  # MLP classifier
            # Flatten the encoder output for MLP: (batch_size, bottleneck_len, bottleneck_dim) -> (batch_size, bottleneck_len * bottleneck_dim)
            z_flat = z.view(z.size(0), -1)
            class_output = self.classifier(z_flat)
        else:
            # Other classifiers (Transformer, CNN) can handle 3D input directly
            class_output = self.classifier(z)
        
        # Compute MMD regularization loss if enabled
        if self.MMD is not None and hasattr(self, 'prior'):
            qz_x = self.prior(*self.prior_params).rsample([z.shape[0]]).to(z.device)
            mmd_loss = self.regularize * self.MMD(z.reshape(z.shape[0], -1), qz_x.reshape(z.shape[0], -1))
        else:
            mmd_loss = 0.0
        
        if targets is not None:
            # Convert one-hot encoded targets to class indices for cross_entropy
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                # Targets are one-hot encoded, convert to class indices
                targets = targets.argmax(dim=1)
            
            classification_loss = F.cross_entropy(class_output, targets)
            total_loss = classification_loss + mmd_loss
            return classification_loss, mmd_loss, total_loss
        else:
            # Return raw logits for loss calculation, apply softmax only for final predictions
            return class_output
    
    def predict(self, x):
        """
        Returns predicted class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


class multimodaldaepclassifier(nn.Module):
    """
    Multimodal DAE classifier that combines multiple modalities for classification.
    
    This class uses multiple tokenizers and a shared encoder to process multimodal
    data and perform classification tasks.
    
    Parameters
    ----------
    tokenizers : dict
        Dictionary mapping modality keys to tokenizer networks
    encoder : nn.Module
        Shared encoder network that processes concatenated tokens
    classifier : nn.Module
        Classifier network that predicts class labels from encoded representations
    measurement_names : dict
        Dictionary mapping modality keys to measurement names
    modality_dropping_during_training : callable, optional
        Function that determines which modalities to use during training
    num_classes : int, optional
        Number of classes for classification
    """
    def __init__(self, tokenizers, encoder, classifier, measurement_names, 
                 modality_dropping_during_training = lambda x: x,
                 num_classes = None):
        super().__init__()
        assert set(tokenizers.keys()) == set(measurement_names.keys()), "modality keys have to match"
        self.modalities = [*tokenizers.keys()]
        modeldims = [tokenizer.model_dim for tokenizer in tokenizers.values()] + [encoder.model_dim]
        assert min(modeldims) == max(modeldims), "model_dims have to match for this implementation"
        
        self.tokenizers = nn.ModuleDict(tokenizers)
        self.encoder = encoder
        self.classifier = classifier
        self.names = measurement_names
        self.modality_dropping_during_training = modality_dropping_during_training
        
        self.model_dim = min(modeldims)
        self.latent_len = encoder.bottleneck_length
        self.latent_dim = encoder.bottleneck_dim
        
        # Modality embeddings to distinguish between different input modalities
        self.modalityEmbedding = nn.ParameterDict({
            key: nn.Parameter(torch.randn(1, 1, self.model_dim) * 0.02) 
            for key in tokenizers.keys()
        })
    
    def encode(self, x, keys = None):
        """
        Encode multimodal input data into latent representations.
        """
        keys = keys if keys is not None else x.keys()
        tokens = [self.tokenizers[key](x[key]) + self.modalityEmbedding[key] for key in keys]
        return self.encoder(torch.concat(tokens, axis=1))
    
    def forward(self, x, targets=None):
        """
        Forward pass through tokenizers, encoder, and classifier.
        
        Parameters
        ----------
        x : dict
            Dictionary with modality keys containing input data
        targets : torch.Tensor, optional
            Target class labels for loss computation
            
        Returns
        -------
        torch.Tensor or tuple
            If targets provided: (classification_loss, total_loss)
            If no targets: class predictions
        """
        # Apply modality dropping during training
        used_keys = self.modality_dropping_during_training(x.keys())
        z = self.encode(x, keys=used_keys)
        class_output = self.classifier(z)
        
        # If targets are provided, compute classification loss
        if targets is not None:
            # Convert one-hot encoded targets to class indices for cross_entropy
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                # Targets are one-hot encoded, convert to class indices
                targets = targets.argmax(dim=1)
            
            # Use cross_entropy for both binary and multi-class classification.
            classification_loss = F.cross_entropy(class_output, targets)
            
            return classification_loss, classification_loss
        else:
            # Return predictions (apply softmax)
            return F.softmax(class_output, dim=-1)
    
    def predict(self, x, condition_keys=None):
        """
        Returns predicted class probabilities
        """
        self.eval()
        with torch.no_grad():
            if condition_keys is None:
                condition_keys = x.keys()
            return self.forward({k: x[k] for k in condition_keys})



