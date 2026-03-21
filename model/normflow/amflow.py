import torch
import torch.nn as nn

# conditioner
from .conditioner import GRUConditioner, MAAMConditioner, BNAFConditioner, NAFConditioner
from .transformer import AffineTransformer, SigmoidalTransformer
from .base_dist import GDN, PlainDistribution
 
class FlipFlow(nn.Module):
    def __init__(self, conditioner_name):
        super().__init__()
        self.conditioner_name = conditioner_name

    def forward(self, input):
        x, mask, logdet = input
        if self.conditioner_name == "gru":
            x_flip = x[:, torch.arange(x.size(1)-1, -1, -1)]
            x_ = x_flip.gather(1, torch.sort((x_flip != 0) * 1, dim=1, descending=True)[1])
        else:
            x_ = torch.zeros_like(x)
            x_[mask == 1] = x_flip[x_flip != 0]
        return x_, mask, logdet


class AMFlow(nn.Module):
    def __init__(self, config):
        super().__init__()

        self._config = config
        self._conditioner_config = config["conditioner"]
        self._transformer_config = config["transformer"]
        self._base_dist_config = config["base_dist"]
        self.n_flows = config["normflow"]["n_flows"]
        self.flip = config["normflow"]["flip"]

        if config["conditioner_name"] == "gru":
            self.conditioner = GRUConditioner
            self._transformer_config["input_dim"] = self._conditioner_config["rnn_hid_dim"]
        elif config["conditioner_name"] == "maam":
            self.conditioner = MAAMConditioner
            self._transformer_config["input_dim"] = self._conditioner_config["num_outlayers"]
        elif config["conditioner_name"] == "bnaf":
            self.conditioner = BNAFConditioner
        elif config["conditioner_name"] == "naf":
            self.conditioner = NAFConditioner
            self._transformer_config["input_dim"] = self._conditioner_config["num_outlayers"]
        else:
            raise NotImplementedError("Conditioner not implemented yet")
        
        if config["transformer_name"] == "affine":
            self.transformer = AffineTransformer
        elif config["transformer_name"] == "sigmoidal":
            self.transformer = SigmoidalTransformer
        else:
            raise NotImplementedError("Transformer not implemented yet")
        
        if config["base_dist_name"] == "gdn":
            self.base_dist = GDN
        elif config["base_dist_name"] == "plain":
            self.base_dist = PlainDistribution
        else:
            raise NotImplementedError("Transformer not implemented yet")
        
        flows = []
        for f in range(self.n_flows):
            if config["conditioner_name"] == "bnaf":    
                flows.append(self.conditioner(self._conditioner_config, 
                                              res="gated" if f < self.n_flows-1 else False))
            else:
                flows.append(self.conditioner(self._conditioner_config))
                flows.append(self.transformer(self._transformer_config))

            if self.flip and f < self.n_flows-1:
                flows.append(FlipFlow())
        self.flows = nn.Sequential(*flows)
    
    def get_config(self): 
        return self._config
    
    def forward(self, input):
        x, mask = input
        # z = f(x), logpx(x) = logpz(f(x)) + logdet(f)
        logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        z, mask, logdet = self.flows((x, mask, logdet))

        return z, mask, logdet
    
    def log_prob(self, input):
        z, mask, logdet = self.forward(input)
        # Compute autoregressive likelihood: logpz(f(x)) = logpz(z) = sum_logpz(z_i|z_(i-1),...,z_0)
        # if self._config["base_dist_name"] == "gdn" and self._base_dist_config["conditioner"] == "gru":
        #     mask = mask.gather(1, torch.sort((mask != 0) * 1, dim=1, descending=True)[1])
        #     z = z.gather(1, torch.sort((z != 0) * 1, dim=1, descending=True)[1])
        logpz = self.base_dist(self._base_dist_config).log_prob((z, mask)) # shape: (b,d)
        logpz = logpz * mask
        sum_logpz = torch.sum(logpz, -1) 

        return sum_logpz + logdet # shape: (b,)
            
 
    # def log_prob(self, input):
    #     x, mask = input
    #     # Compute autoregressive likelihood: logpx(x) = sum_logpx(x_i|x_(i-1),...,x_0)
    #     if self._config["base_dist_name"] == "gdn" and self._base_dist_config["conditioner"] == "gru":
    #         mask = mask.gather(1, torch.sort((mask != 0) * 1, dim=1, descending=True)[1])
    #         x = x.gather(1, torch.sort((x != 0) * 1, dim=1, descending=True)[1])
    #     logpx = self.base_dist(self._base_dist_config).log_prob((x, mask)) # shape: (b,d)
    #     logpx = logpx * mask
    #     sum_logpx = torch.sum(logpx, -1) 

        # return sum_logpx # shape: (b,)
    