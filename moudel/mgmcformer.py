import torch
import torch.nn as nn
from moudel.timewise import PositionalEncoding
from moudel.multi_group_encoder import Multi_Group_encoders
from moudel.channelwise import ChannelTransformerModel


class MgMcFORMER(nn.Module):
    def __init__(self, multi_group, nclasses, seq_len, input_size, emb_size, nhid, emb_size_c, nhid_c, nhead, nlayers,
                 device='cuda:0', dropout=0.1):
        super(MgMcFORMER, self).__init__()
        # embeding + PositionEncoding
        self.trunk_net = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.LayerNorm(emb_size),
            PositionalEncoding(seq_len, emb_size, dropout),
            nn.LayerNorm(emb_size)
        )
        # Multi_group
        self.multi_group = Multi_Group_encoders(multi_group=multi_group, seq_len=seq_len, emb_size=emb_size, \
                                                nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout)
        # C_A
        self.c_a = ChannelTransformerModel(seq_len=seq_len, emb_size=emb_size_c, nhead=nhead, nhid=nhid_c,
                                           nlayers=nlayers, device=device, dropout=dropout)

        self.retrunk_net = nn.Sequential(
            nn.Linear(emb_size_c, seq_len),
            nn.LeakyReLU(),
            nn.LayerNorm(seq_len),
        )

        # class
        self.class_net = nn.Sequential(
            nn.Linear(seq_len * emb_size, nhid_c),
            nn.LeakyReLU(),
            nn.LayerNorm(nhid_c),
            nn.Dropout(p=0.3),
            nn.Linear(nhid_c, nhid_c),
            nn.LeakyReLU(),
            nn.LayerNorm(nhid_c),
            nn.Dropout(p=0.3),
            nn.Linear(nhid_c, nclasses)
        )

        self.laynorm = nn.LayerNorm(emb_size)

    def forward(self, x, mask=None):
        x = self.trunk_net(x)

        out = self.multi_group(x)

        res = out
        if mask is not None:
            out = torch.where(mask, torch.as_tensor(0.0, device='cuda:0'), out)

        out, att = self.c_a(out)

        out = self.retrunk_net(out).permute(0, 2, 1)
        out = self.laynorm(out + res)

        out = self.class_net(out.view([out.shape[0], -1]))

        return out, att
