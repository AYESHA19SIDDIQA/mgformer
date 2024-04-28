import torch
import torch.nn as nn

from moudel import encoderlayer
from moudel import transformer


class Permute(torch.nn.Module):
    def forward(self, x):
        return x.permute(1, 0)


class ChannelTransformerModel(nn.Module):

    def __init__(self, device, seq_len, emb_size, nhead, nhid, nlayers, dropout=0.1):
        super(ChannelTransformerModel, self).__init__()
        self.trunk_net = nn.Sequential(
            nn.Linear(seq_len, emb_size),
            nn.LayerNorm(emb_size),
        )

        encoder_layers = encoderlayer.Encoder(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = transformer.TransformerEncoders(encoder_layers, nlayers, device)
        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = self.trunk_net(x.permute(0, 2, 1))
        x, attn = self.transformer_encoder(x)
        output = self.layer_norm(x)

        return output, attn
