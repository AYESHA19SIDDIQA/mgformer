import math
import copy
import torch
import torch.nn as nn
from moudel import encoderlayer
from moudel.transformer import TransformerEncoders


class GroupEnconde(nn.Module):
    def __init__(self, group, seq_len, emb_size, nhead, nhid, nlayers, dropout=0.1):
        super(GroupEnconde, self).__init__()
        self.group = group
        self.len = math.ceil(seq_len / self.group)

        self.encodelayers = encoderlayer.Encoder(emb_size, nhead, nhid, dropout)
        self.Transformer = TransformerEncoders(self.encodelayers, nlayers)

        self.groupEncode = nn.ModuleList([copy.deepcopy(self.Transformer) for i in range(self.group)])

    def forward(self, x):
        x2 = []
        out = []
        for i in range(self.group):
            x1 = x[:, self.len * i:self.len * (i + 1), :]
            x2.append(x1)
        # group-transformer
        for i in range(self.group):
            out.append(self.groupEncode[i](x2[i])[0])
        out = torch.cat(out, 1)

        return out


class Multi_Group_encoders(nn.Module):
    def __init__(self, multi_group, seq_len, emb_size, nhead, nhid, nlayers, dropout=0.1):
        super(Multi_Group_encoders, self).__init__()
        # multi_group
        self.group_layers = nn.ModuleList(
            [GroupEnconde(i, seq_len, emb_size, nhead, nhid, nlayers, dropout) for i in multi_group])
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, input):
        output = input

        for layer in self.group_layers:
            res = output
            output = layer(output) + res
            output = self.layernorm(output)

        return output
