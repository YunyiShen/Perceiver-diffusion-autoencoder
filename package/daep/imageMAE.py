import torch
import torch.nn as nn
import copy

class ImgMAE(nn.Module):
    def __init__(self, tokenizer, encoder, decoder, detokenizer, mask_ratio=0.3, name="flux"):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.detokenizer = detokenizer
        self.mask_ratio = mask_ratio
        self.name = name
        self.loss_fn = nn.MSELoss(reduction="none")

        # learnable mask token
        self.register_parameter("mask_token", nn.Parameter(torch.randn(1, 1, tokenizer.model_dim)))

    def encode(self, x):
        tokens = self.tokenizer(x[self.name])  # [B, N, D]
        return self.encoder(tokens)

    def reconstruct(self, x):
        tokens = self.tokenizer(x[self.name])
        z = self.encoder(tokens)
        # replace no tokens for reconstruction (or could mask some if you like)
        recon_tokens = self.decoder(z, tokens, aux=None)
        x_recon = copy.deepcopy(x)
        x_recon[self.name] = self.detokenizer(recon_tokens)
        return x_recon

    def forward(self, x):
        """
        Args:
            x[self.name]: [B, C, H, W] or [B, L, D] for sequences
        """
        # 1. Tokenize
        tokens = self.tokenizer(x[self.name])  # [B, N, D]
        #breakpoint()
        # 2. Generate mask
        B, N, D = tokens.shape
        mask = torch.rand(B, N, device=tokens.device) < self.mask_ratio  # [B, N]

        # 3. Prepare masked tokens
        mask_token_exp = self.mask_token.expand(B, N, D)
        tokens_masked = torch.where(mask.unsqueeze(-1), mask_token_exp + self.tokenizer.pos_embed, tokens)

        # 4. Encode masked tokens
        z = self.encoder(tokens_masked, mask=mask)

        # 5. Decode
        recon_tokens = self.decoder(z, tokens_masked, aux=None)
        img_recon = self.detokenizer(recon_tokens)

        # 6. Compute loss only on masked tokens
        loss = self.loss_fn(x[self.name], img_recon).mean()  # [B, N, D]
        #loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * D + 1e-8)

        return loss
