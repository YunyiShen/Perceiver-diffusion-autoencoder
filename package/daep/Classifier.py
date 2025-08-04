import torch
import torch.nn as nn
import torch.nn.functional as F


class LCC(nn.Module):
    def __init__(self, emb_d, num_heads, layers, dropout_p, ffn_d, num_classes):
        super(LCC,self).__init__()
        self.fc1 = nn.Linear(emb_d, 20)
        self.fc2 = nn.Linear(20,num_classes)
        self.swish = nn.SiLU()
        self.norm0 = nn.LayerNorm(normalized_shape=20)
        self.norm1 = nn.LayerNorm(normalized_shape=num_classes)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout0 = nn.Dropout(dropout_p)

    def forward(self, x,t, mask=None):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.norm0(x)
        x = self.swish(x)
        x = self.dropout0(x)
        x = self.fc2(x)
        x = self.norm1(x)
        x = F.softmax(x, dim=-1).squeeze()
        return x