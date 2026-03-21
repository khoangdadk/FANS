import numpy as np

import torch
import torch.nn as nn

from ...nn import MLP
from ..conditioner import MAAMConditioner



# Base Distribution: Gaussian Density Network
class GDN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.component = config["component"]
        self.n_components = config["n_components"]
        self.gdn_conditioner = config["conditioner"]
        self.param_net_hid_dim = config["param_net_hid_dim"]
        self.n_param_net_hid_layers = config["n_param_net_hid_layers"]
        if self.gdn_conditioner == "gru":
            self.rnn = nn.GRU(1, config["gru"]["rnn_hid_dim"], num_layers=config["gru"]["n_rnn_layers"], batch_first=True, dtype=torch.float64)
            self.param_net_input_dim = config["gru"]["rnn_hid_dim"]
        elif self.gdn_conditioner == "maam":
            self.maam = MAAMConditioner(config["maam"]) # output (b,d,ol)
            self.param_net_input_dim = config["maam"]["num_outlayers"]
        else: 
            raise NotImplementedError("GDN conditioner not implemented yet")
        # shared param DNN: map o -> GMM params theta(o)
        self.mixture_param_net = MLP([self.param_net_input_dim] + [self.param_net_hid_dim]*self.n_param_net_hid_layers + [3 * self.n_components], init_zeros=True)

    def forward(self, input):
        # z: shape (b,d)
        z, mask = input
        if self.gdn_conditioner == "gru":
            # z_: shape (b,d+1,1), add margin element z_u^0 = -1, h^0 = 0
            z_margin = torch.tensor([[0.0]] * z.size(0), device=z.device, requires_grad=False)
            z_ = torch.cat((z_margin, z), dim=1).unsqueeze(2)
            o, _ = self.rnn(z_)  # RNN output: shape (b,d+1,hid)
            # o_1: z_0, o_2: z_0, z_1,...
            o = o[:, :-1, :] # only get first bdim dimensions: shape (b,d,hid)
        elif self.gdn_conditioner == "maam":
            _, o, _, _ = self.maam((z, mask, None))
        o = o * mask.unsqueeze(-1) # (b,d,ol = 3 * n_components)

        # GMM params theta(o)
        gmm_param = self.mixture_param_net(o) # shape (b,d,3 * n_components)
        mu_ = gmm_param[..., :self.n_components] # shape (b,d,n_components)
        lstd_ = gmm_param[..., self.n_components: 2 * self.n_components] # shape (b,d,n_components)
        logits_ = gmm_param[..., -self.n_components:]
        mu = mu_ * mask.unsqueeze(-1)
        lstd = lstd_ * mask.unsqueeze(-1)
        logits = logits_ * mask.unsqueeze(-1)

        return mu, lstd, logits
      
    def log_prob(self, input):
        z, mask = input
        # z: shape (b,d)
        mu, lstd, logits = self.forward(input)
        std = lstd.exp()
        std = torch.clip(std, min=1e-6, max=None) # std > 0

        if self.component == 'gaussian':
            log_norm_consts = -lstd - 0.5 * np.log(2.0 * np.pi) # shape (b,d,n_components)
            log_kernel = -0.5 * torch.square((z.unsqueeze(-1) - mu) / std) # shape (b,d,n_components)
        elif self.component == 'laplace':
            log_norm_consts = -lstd - np.log(2.0)
            log_kernel = -torch.abs(z.unsqueeze(-1) - mu) / std
        elif self.component == 'logistic':
            log_norm_consts = -lstd
            diff = (z.unsqueeze(-1) - mu) / std
            log_kernel = -torch.nn.Softplus(diff) - torch.nn.Softplus(-diff)
        else:
            raise NotImplementedError
        
        log_exp_terms = log_kernel + log_norm_consts + logits
        log_exp_terms = log_exp_terms * mask.unsqueeze(-1)

        # logsumexp trick for stable computation 
        log_lls = torch.logsumexp(log_exp_terms, -1) - torch.logsumexp(logits, -1) # shape (b,d)

        return log_lls