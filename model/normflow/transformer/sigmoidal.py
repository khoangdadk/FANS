import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from ...utils import softmax, softplus, log, logsigmoid, log_sum_exp, DELTA
 


class SigmoidalTransformer(nn.Module):
    def __init__(self, config):
        super(SigmoidalTransformer, self).__init__()
        self.mollify = config["mollify"]
        self.num_ds_dim = config["num_ds_dim"]
        self.num_ds_layers = config["num_ds_layers"]
        self.input_dim = config["input_dim"]
        self.out_to_dsparams = nn.Conv1d(self.input_dim, 3*self.num_ds_dim*self.num_ds_layers, 1).to(torch.float64) 
        self.reset_parameters()    

        self.act_a = lambda x: softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: softmax(x, dim=2)
    
    def reset_parameters(self):
        self.out_to_dsparams.weight.data.uniform_(-0.001, 0.001)
        self.out_to_dsparams.bias.data.uniform_(0.0, 0.0)
        inv = np.log(np.exp(1-DELTA)-1) 
        for l in range(self.num_ds_layers):
            nc = self.num_ds_dim
            nparams = nc * 3 
            s = l*nparams
            self.out_to_dsparams.bias.data[s:s+nc].uniform_(inv,inv)
    
    def forward_each_layer(self, input):
        x, mask, logdet, params = input
        ndim = self.num_ds_dim
        a_ = self.act_a(params[:,:,0*ndim:1*ndim])
        b_ = self.act_b(params[:,:,1*ndim:2*ndim])
        w = self.act_w(params[:,:,2*ndim:3*ndim])
        
        a = a_ * (1-self.mollify) + 1.0 * self.mollify
        b = b_ * (1-self.mollify) + 0.0 * self.mollify
        
        pre_sigm = a * x[:,:,None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w*sigm, dim=2)
        x_pre_clipped = x_pre * (1-DELTA) + DELTA * 0.5

        x_ = log(x_pre_clipped) - log(1-x_pre_clipped)

        logj = F.log_softmax(params[:,:,2*ndim:3*ndim], dim=2) + logsigmoid(pre_sigm) + logsigmoid(-pre_sigm) + log(a)
        logj = log_sum_exp(logj, 2).sum(2)
        logdet_ = logj + np.log(1-DELTA) - (log(x_pre_clipped) + log(-x_pre_clipped+1)) # (b,d)
        
        # mask logdet
        masked_logdet_ = logdet_ * mask
        logdet = masked_logdet_.sum(1) + logdet

        return x_, logdet
         
    def forward(self, input):
        x, cond_o, mask, logdet = input

        cond_o = cond_o.permute(0,2,1)
        dsparams = self.out_to_dsparams(cond_o)
        dsparams = dsparams.permute(0,2,1) # (b,d,dsparam)
        dsparams = dsparams * mask.unsqueeze(-1)
        nparams = self.num_ds_dim*3
        
        z = (x * mask).view(x.size(0), -1)
        for i in range(self.num_ds_layers):
            params = dsparams[:,:,i*nparams:(i+1)*nparams]
            z, logdet = self.forward_each_layer((z, mask, logdet, params))

        return z, mask, logdet  
