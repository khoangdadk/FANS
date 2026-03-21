import torch
import torch.nn as nn

from ...nn import MLP



class AffineTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.aff_hid_dim = config["aff_hid_dim"]
        self.n_aff_hid_layers = config["n_aff_hid_layers"]
        self.input_dim = config["input_dim"]
        self.shift_scale_net = MLP([self.input_dim] + [self.aff_hid_dim]*self.n_aff_hid_layers + [2])

    def forward(self, input):
        x, cond_o, mask, logdet = input
        shift_scale_param = self.shift_scale_net(cond_o) # shape (b,d,2)
        shift_ = shift_scale_param[..., 0] # shape (b,d)
        scale_ = shift_scale_param[..., 1] # shape (b,d)
        shift = shift_ * mask
        scale = scale_ * mask
        
        z = x * torch.exp(scale) + shift # shape (b,d)
        logdet_ = torch.sum(scale, dim=1) # shape (b,)
        logdet += logdet_

        return z, mask, logdet
 