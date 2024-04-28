import torch, copy
import torch.nn as nn

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoders(nn.Module):
    def __init__(self, encoder_layer, num_layers, device='cuda:0', norm = None):
        super(TransformerEncoders, self).__init__()
        self.device = device
        self.layers=_get_clones(encoder_layer, num_layers)

        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):

        output = src
        attn_output = torch.zeros((src.shape[0], src.shape[1], src.shape[1]),device=self.device)  # batch, seq_len, seq_len
        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_output += attn  # attention is all you need use list
        # attn_output=attn_output//len(self.layers)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_output






























# pe)