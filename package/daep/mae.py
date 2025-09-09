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
    
    def reconstruct(self, x):
        z = self.encoder(x) 
        x_recon = copy.deepcopy(x)  
        x_recon[self.name] = self.decoder(x_recon, z, None)

        return x_recon
    
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
