import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, seq_len, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        max_len = max(5000, seq_len)
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos((position * div_term))
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0: -1]

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
