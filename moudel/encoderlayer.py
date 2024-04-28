from torch.nn import Module
import torch
from torch.nn import MultiheadAttention
from moudel.feedForward import FeedForward


class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 d_hidden=1024,
                 layer_norm_eps=1e-5,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()

        self.MHA = MultiheadAttention(d_model, num_heads=nhead, batch_first=True)
        self.feedforward = FeedForward(d_model, d_hidden)

        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout2 = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        residual = x
        x, score = self.MHA(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = self.dropout1(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = self.layerNormal_2(x + residual)

        return x, score
