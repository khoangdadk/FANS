# https://github.com/CW-Huang/NAF/

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import numpy as np
from functools import reduce



delta = 1e-6



def get_rank(max_rank, num_out):
    rank_out = np.array([])
    while len(rank_out) < num_out:
        rank_out = np.concatenate([rank_out, np.arange(max_rank)])
    excess = len(rank_out) - num_out
    remove_ind = np.random.choice(max_rank,excess,False)
    rank_out = np.delete(rank_out,remove_ind)
    np.random.shuffle(rank_out)
    return rank_out.astype('float64')
    


def get_mask_from_ranks(r1, r2):
    return (r2[:, None] >= r1[None, :]).astype('float64')



def get_masks_all(ds, fixed_order=False, derank=1):
    # ds: list of dimensions dx, d1, d2, ... dh, dx, 
    #                       (2 in/output + h hidden layers)
    # derank only used for self connection, dim > 1
    dx = ds[0]
    ms = list()
    rx = get_rank(dx, dx)
    if fixed_order:
        rx = np.sort(rx)
    r1 = rx
    if dx != 1:
        for d in ds[1:-1]:
            r2 = get_rank(dx-derank, d)
            ms.append(get_mask_from_ranks(r1, r2))
            r1 = r2
        r2 = rx - derank
        ms.append(get_mask_from_ranks(r1, r2))
    else:
        ms = [np.zeros([ds[i+1],ds[i]]).astype('float64') for \
              i in range(len(ds)-1)]
    if derank==1:
        assert np.all(np.diag(reduce(np.dot,ms[::-1])) == 0), 'wrong masks'
    
    return ms, rx



def get_masks(dim, dh, num_layers, num_outlayers, fixed_order=False, derank=1):
    ms, rx = get_masks_all([dim,]+[dh for i in range(num_layers)]+[dim,],
                           fixed_order, derank)
    ml = ms[-1]
    ml_ = (ml.transpose(1,0)[:,:,None]*([np.asarray(1, dtype=np.float64),] *\
                           num_outlayers)).reshape(
                           dh, dim*num_outlayers).transpose(1,0)
    ms[-1]  = ml_
    return ms, rx



class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function
        
    def forward(self, input):
        return self.function(input)



class CWNlinear(nn.Module):
    def __init__(self, in_features, out_features, num_outlayers, mask=None, norm=True, is_V=False):
        super(CWNlinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('mask',mask)
        self.norm = norm
        self.is_V = is_V
        self.num_outlayers = num_outlayers
        self.direction = Parameter(torch.Tensor(out_features, in_features).to(dtype=torch.float64))
        self.cscale = nn.Linear(1, out_features, dtype=torch.float64)
        self.cbias = nn.Linear(1, out_features, dtype=torch.float64)
        self.reset_parameters()
        self.cscale.weight.data.normal_(0, 0.001)
        self.cbias.weight.data.normal_(0, 0.001)

    def reset_parameters(self):
        self.direction.data.normal_(0, 0.001)
        
    def forward(self, input):
        x, mask = input
        zeros = Variable(torch.FloatTensor(x.size(0), 1).zero_()).to(x.device, dtype=x.dtype)
        scale = self.cscale(zeros)
        bias = self.cbias(zeros)
        if self.norm:
            dir_ = self.direction
            direction = dir_.div(dir_.pow(2).sum(1).sqrt()[:,None])
            weight = direction
        else:
            weight = self.direction
        if self.mask is not None:
            weight = weight * Variable(self.mask)

        if self.is_V == False:
            return scale * F.linear(x, weight, None) + bias, mask
        else:
            out_mask = mask.repeat_interleave(self.num_outlayers, dim=1)
            masked_scale = out_mask * scale
            masked_bias = out_mask * bias # (b,out)
            weighted_out = F.linear(x, weight, None)
            masked_weighted_out = out_mask * weighted_out
            return masked_scale * masked_weighted_out + masked_bias, mask



class cMADE(nn.Module):
    def __init__(self, 
                 dim, 
                 hid_dim, 
                 num_layers,
                 num_outlayers, 
                 activation=nn.ELU(),
                 derank=1):
        super(cMADE, self).__init__()
        
        oper = CWNlinear
        
        self.dim = dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.num_outlayers = num_outlayers
        self.activation = Lambda(lambda x: (nn.ELU()(x[0]), *x[1:]))
        
        ms, rx = get_masks(dim, hid_dim, num_layers, num_outlayers,
                        derank)
        ms = [m for m in map(torch.from_numpy, ms)]
        self.rx = rx
        
        sequels = list()
        sequels.append(oper(dim, hid_dim, self.num_outlayers, ms[0], norm=False, is_V=False))
        sequels.append(self.activation)
        # for i in range(num_layers-1):
        #     sequels.append(oper(hid_dim, hid_dim, self.num_outlayers, ms[i], norm=False, is_V=False))
        #     sequels.append(self.activation)    
        self.input_to_hidden = nn.Sequential(*sequels)
        self.hidden_to_output = oper(hid_dim, dim*num_outlayers, self.num_outlayers, ms[-1], norm=True, is_V=True)
        
    def forward(self, input):
        hid = self.input_to_hidden(input)
        out, mask = self.hidden_to_output(hid)
        return out.view(-1, self.dim, self.num_outlayers)
    


class NAFConditioner(nn.Module):
    def __init__(self, config):
        super(NAFConditioner, self).__init__()

        self.dim = config["dim"]
        self.hid_dim = config["hid_dim"] 
        self.num_hid_layers = config["num_hid_layers"]
        self.num_outlayers = config["num_outlayers"]
        self.activation = nn.ELU()
        
        self.mdl = cMADE(
                        self.dim, 
                        self.hid_dim, 
                        self.num_hid_layers, 
                        self.num_outlayers, 
                        self.activation
                    )
        
    def forward(self, input):
        x, mask, logdet = input
        out = self.mdl((x,mask))
        return x, out, mask, logdet