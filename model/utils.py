import numpy as np

import torch
import torch.nn as nn
    

DELTA = 1e-6

class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function
    def forward(self, input):
        return self.function(input)
    

def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out 


log = lambda x: torch.log(x*1e2) - np.log(1e2)
softplus = lambda x: log(1.0 + torch.exp(x)) + DELTA # softplus(x) = log(1 + exp(x))
logsigmoid = lambda x: -softplus(-x) # logsigmoid(x) = log(1/(1 + exp(-x)))


def log_sum_exp(A, axis=-1, sum_op=torch.sum):    
    maximum = lambda x: x.max(axis)[0]    
    A_max = oper(A, maximum, axis, True)
    summation = lambda x: sum_op(torch.exp(x - A_max), axis)
    B = torch.log(oper(A, summation, axis, True)) + A_max    
    return B


def oper(array,oper, axis=-1, keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for _,s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def normalize_px(px, obs_mask):
    d = obs_mask.size(1)
    mask_prob = obs_mask.sum(1)*1.0/d # (b,)
    px_norm = px * mask_prob
    return px_norm