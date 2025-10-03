import torch
import torch.nn.functional as F
import torch.nn as nn

def multimodal_contrastive_loss(emb_dict, tau=0.07):
    """
    emb_dict: dict {modality_name: tensor [B, d]}
              Each tensor contains embeddings for one modality.
              All modalities must share the same batch size B.
    tau: temperature
    """
    modalities = list(emb_dict.keys())
    embeddings = [F.normalize(emb_dict[m], dim=-1) for m in modalities]
    
    B = embeddings[0].size(0)
    device = embeddings[0].device
    targets = torch.arange(B, device=device)
    
    losses = []
    M = len(embeddings)
    
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            sim = embeddings[i] @ embeddings[j].T  # [B, B]
            loss = F.cross_entropy(sim / tau, targets)
            losses.append(loss)
    
    return torch.stack(losses).mean()


class contrastiveae(nn.Module):
    def __init__(self, aes, tau = 0.07, modality_weights = None, contrastive_weight = 1.):
        super().__init__()
        self.modalities = aes.keys()
        self.tau = tau
        self.aes = aes
        if modality_weights is None:
            modality_weights = {key: 1.0 for key in self.modalities}
        
        self.modality_weights = modality_weights
        self.contrastive_weight = contrastive_weight

    def forward(self, x):
        recloss = 0.0
        enc = {}
        for key in self.modalities:
            recloss += self.modality_weights[key] * self.aes(x[key])
            enc[key] = self.aes.encode(x[key])
        
        contrastive_loss = multimodal_contrastive_loss(enc, self.tau)
        return self.contrastive_weight*contrastive_loss + recloss