import numpy as np
import torch
from torch.distributions import Distribution
from torch.autograd import Variable
from ...utils import DELTA
 

 
class PlainDistribution(Distribution):
    def __init__(self, config):
        super(PlainDistribution, self).__init__(validate_args=False)
        self.name = config["name"]
    
    def log_prob(self, input):
        z, _ = input
        if self.name == "gaussian":
            mean = Variable(torch.FloatTensor(z.size(0), z.size(1)).zero_()).to(z.device, dtype=z.dtype) # (b,d)
            var = mean + 1.0
            log_norm = -(z - mean) ** 2 / (2.0 * var + DELTA) - 0.5 * torch.log(torch.tensor(2.0) * np.pi)
            return log_norm
        else:
            raise NotImplementedError("base distribution not implemented yet")

 