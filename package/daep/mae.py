import torch
import torch.nn as nn
import torch.optim as optim
import copy

class unimodalmae(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.3, name = "flux"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.name = name
        self.loss_fn = nn.MSELoss(reduce = None)

    
    def encode(self, x):
        return self.encoder(x)
    
    def reconstruct(self, x, mask_ratio = None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        masked = torch.rand_like(x[self.name]) < mask_ratio
        if x.get("mask") is None:
            x['mask'] = torch.full(x[self.name].shape, False, dtype=torch.bool).to(x[self.name].device)
        x_masked = copy.deepcopy(x)
        x_masked['mask'] = torch.logical_or(x['mask'], masked)
        all_masked = x_masked['mask'].all(dim = -1)
        if all_masked.any():
            for i in torch.where(all_masked)[0]:
                x_masked['mask'][i][0] = False # fall back if all masked
                masked[i][0] = False
        
        x_masked[self.name][x_masked['mask']] = 0.0
        z = self.encoder(x)  
        recloc = torch.logical_and(~x['mask'], masked)
        x_recon = self.decoder(x_masked, z, recloc, None) 
        #breakpoint()
        x_recon[~recloc] = x[self.name][~recloc]
        x_masked[self.name] = x_recon
        x_masked["mask"] = x.get("mask")
        return x_masked
    
    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        """
        masked = torch.rand_like(x[self.name]) < self.mask_ratio
        if x.get("mask") is None:
            x['mask'] = torch.full(x[self.name].shape, False, dtype=torch.bool).to(x[self.name].device)
        x_masked = copy.deepcopy(x)
        x_masked['mask'] = torch.logical_or(x['mask'], masked)
        all_masked = x_masked['mask'].all(dim = -1)
        if all_masked.any():
            for i in torch.where(all_masked)[0]:
                x_masked['mask'][i][0] = False # fall back if all masked
                masked[i][0] = False
        
        x_masked[self.name][x_masked['mask']] = 0.0
        z = self.encoder(x_masked)  
        recloc = torch.logical_and(~x['mask'], masked)
        x_recon = self.decoder(x_masked, z, recloc, None) 
        #breakpoint()
        loss = self.loss_fn(x_recon, x[self.name])
        
        loss = (loss * recloc).sum()/recloc.sum()
        if torch.isnan(loss):
            breakpoint()

        return loss




class multimodalmae(nn.Module):
    def __init__(self, tokenizers, encoder, decoder, 
                 measurement_names, modality_weights = None,
                 modality_dropping_during_training = None,
                 persample_dropping_p = 0., mask_ratio=0.3):
        
        super().__init__()
        assert set(tokenizers.keys()) == set(decoder.keys()) and set(tokenizers.keys()) == set(measurement_names.keys()), "modality keys have to match"
        self.modalities = [*tokenizers.keys()]
        modeldims = [score.model_dim for score in decoder.values()] + [tokenizer.model_dim for tokenizer in tokenizers.values()] + [encoder.model_dim]
        assert min(modeldims) == max(modeldims), "model_dims have to match for this implementation"
        
        self.tokenizers = nn.ModuleDict(tokenizers)
        self.encoder = encoder
        self.decoder = nn.ModuleDict(decoder)
        self.names = measurement_names
        self.modality_dropping_during_training = modality_dropping_during_training
        
        self.model_dim = min(modeldims)
        
        self.latent_len = encoder.bottleneck_length
        self.latent_dim = encoder.bottleneck_dim
        
        self.modalityEmbedding = nn.ParameterDict({key: nn.Parameter(torch.randn(1, 1, self.model_dim) * 0.02) for key in tokenizers.keys()})
        
        if modality_weights is None:
            modality_weights = {key: 1.0 for key in self.modalities}
        self.modality_weights = modality_weights
        self.persample_dropping_p = persample_dropping_p
        
        self.mask_ratio = mask_ratio
        self.loss_fn = nn.MSELoss(reduce = None)

    
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
    
    
    def reconstruct(self, x, condition_keys = None, out_keys = None, mask_ratio = None):
        if condition_keys is None:
            condition_keys = x.keys()
        if out_keys is None:
            out_keys = x.keys()
            
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        for key in condition_keys:
            if x[key].get("mask") is None:
                x[key]['mask'] = torch.full(x[key][self.names[key]].shape, False, dtype=torch.bool).to(x[key][self.names[key]].device)
        x_masked = {key:copy.deepcopy(x[key]) for key in out_keys}
        z = self.encoder(x, condition_keys)  
        for key in out_keys:
            masked = torch.rand_like(x[key][self.names[key]]) < mask_ratio
            x_masked[key]['mask'] = torch.logical_or(x[key]['mask'], masked)
            all_masked = x_masked[key]['mask'].all(dim = -1)
            if all_masked.any():
                for i in torch.where(all_masked)[0]:
                    x_masked[key]['mask'][i][0] = False # fall back if all masked
                    masked[i][0] = False
        
            x_masked[key][self.names[key]][x_masked['mask']] = 0.0
            
            recloc = torch.logical_and(~x[key]['mask'], masked)
            x_recon = self.decoder[key](x_masked[key], z, recloc, None)
            #breakpoint()
            x_recon[~recloc] = x[key][self.names[key]][~recloc]
            x_masked[key][self.names[key]] = x_recon
            x_masked[key]["mask"] = x[key].get("mask")
        return x_masked
    
    def forward(self, x):
        """
        x: [batch, seq_len, dim]
        """
        keys = x.keys()
        
        
        
        for key in keys:
            if x[key].get("mask") is None:
                x[key]['mask'] = torch.full(x[key][self.names[key]].shape, False, dtype=torch.bool).to(x[key][self.names[key]].device)
        
        x_masked = copy.deepcopy(x)
        recloc = {}
        
        for key in keys:
            masked = torch.rand_like(x[key][self.names[key]]) < self.mask_ratio
        
            x_masked[key]['mask'] = torch.logical_or(x[key]['mask'], masked)
            all_masked = x_masked[key]['mask'].all(dim = -1)
            if all_masked.any():
                for i in torch.where(all_masked)[0]:
                    x_masked[key]['mask'][i][0] = False # fall back if all masked
                    masked[i][0] = False
        
            x_masked[key][self.names[key]][x_masked[key]['mask']] = 0.0
            recloc[key] = torch.logical_and(~x[key]['mask'], masked)
        
        
        z = self.encoder(x_masked)  
        loss = 0.0
        locs = 0.0
        for key in keys:
            x_recon = self.decoder[key](x_masked[key], z, recloc[key], None) 
            #breakpoint()
            loss += self.modality_weights[key] * (self.loss_fn(x_recon, x[key][self.names[key]]) * recloc[key]).sum()
            locs += recloc[key].sum()
        
        loss = loss/locs
        
        
        
        if torch.isnan(loss):
            breakpoint()

        return loss