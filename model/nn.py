import torch
import torch.nn as nn
from torch.autograd import Variable



class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/nets/mlp.py
    """
 
    def __init__(
        self,
        layers,
        leaky=0.0,
        init_zeros=False,
        dropout=None,
    ):
        """
        layers: list of layer sizes from start to end
        leaky: slope of the leaky part of the ReLU, if 0.0, standard ReLU is used
        init_zeros: Flag, if true, weights and biases of last layer are initialized with zeros 
        (helpful for deep models, see [arXiv 1807.03039](https://arxiv.org/abs/1807.03039))
        dropout: Float, if specified, dropout is done before last layer; if None, no dropout is done
        """
        super().__init__()
        net = nn.ModuleList([])
        for k in range(len(layers) - 2):
            net.append(nn.Linear(layers[k], layers[k + 1], dtype=torch.float64))
            net.append(nn.LeakyReLU(leaky))
        if dropout is not None:
            net.append(nn.Dropout(p=dropout))
        net.append(nn.Linear(layers[-2], layers[-1], dtype=torch.float64))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


# # arbitrary marginal masked linear layer
# class AMMaskedLinear(nn.Module):
#     def __init__(self, 
#                  d, 
#                  in_features, 
#                  out_features, 
#                  units, 
#                  dh_per_unit, 
#                  num_outlayers, 
#                  perm_low=None, 
#                  perm_high=None, 
#                  is_W1=False, 
#                  is_V=False):
#         super(AMMaskedLinear, self).__init__()
#         self.d = d
#         self.in_features = in_features
#         self.out_features = out_features
#         self.units = units
#         self.dh_per_unit = dh_per_unit
#         self.num_outlayers = num_outlayers
#         self.is_W1 = is_W1
#         self.is_V = is_V
#         self.perm_low = perm_low
#         self.perm_high = perm_high
#         self.rank = torch.arange(1, self.dh_per_unit+1).repeat_interleave(self.units)
#         self.direction = nn.Parameter(torch.Tensor(out_features, in_features).to(torch.float64))
#         # if not self.is_V:
#         #     self.cscale = nn.Linear(1, (self.d-1)*self.units, dtype=torch.float64) # (b,out=(d-1)*units)
#         #     self.cbias = nn.Linear(1, (self.d-1)*self.units, dtype=torch.float64) # (b,out=(d-1)*units)
#         # else:
#         #     self.cscale = nn.Linear(1, out_features, dtype=torch.float64)
#         #     self.cbias = nn.Linear(1, out_features, dtype=torch.float64)
#         self.cscale = nn.Linear(1, out_features, dtype=torch.float64)
#         self.cbias = nn.Linear(1, out_features, dtype=torch.float64)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.direction.data.normal_(0, 0.001)
#         self.cscale.weight.data.normal_(0, 0.001)
#         self.cbias.weight.data.normal_(0, 0.001)
        
#     def forward(self, input):
#         x, mask, pre_mask, hidden_rank, pre_masked_idx = input # x (b,d), mask (b,d)

#         zeros = Variable(torch.FloatTensor(x.size(0), 1).zero_()).to(x.device, dtype=x.dtype)
#         scale = self.cscale(zeros) # (b,out)
#         bias = self.cbias(zeros) # (b,out)
#         weight = self.direction # (out,in)
#         rank = self.rank.to(device=x.device)

#         if self.is_V:
#             out_mask = Variable(mask.repeat_interleave(self.num_outlayers, dim=1))
#             final_layer_hidden_rank = ((mask - pre_mask) * (self.d + 1)).int() + hidden_rank 
#             masked_rs_high = final_layer_hidden_rank.repeat_interleave(self.num_outlayers, dim=1)
#             masked_scale = out_mask * scale # (b,out)
#             masked_bias = out_mask * bias # (b,out)
#         else:
#             # ordered_masked_rs_high = rank * (rank == hidden_rank.unsqueeze(-1)).any(dim=1) 
#             # out_indices = torch.zeros_like(ordered_masked_rs_high, device=x.device)
#             # ordered_masked_rs_high_bool = ordered_masked_rs_high != 0
#             # out_indices[torch.nonzero(ordered_masked_rs_high_bool, as_tuple=True)] = pre_masked_idx[pre_masked_idx != 0]
#             # out_indices = ((out_indices-1) * self.units + torch.arange(self.units, device=x.device).repeat(self.dh_per_unit)) * ordered_masked_rs_high_bool.int()
            
#             # masked_rs_high = ordered_masked_rs_high[:, self.perm_high]
#             # rs_high_mask = (masked_rs_high != 0).to(dtype=x.dtype, device=x.device)
#             # out_indices = out_indices[:, self.perm_high]
#             # batch_indices = torch.arange(scale.size(0), device=x.device).unsqueeze(1).expand(-1, masked_rs_high.size(1))
#             # masked_scale = scale[batch_indices, out_indices] * rs_high_mask
#             # masked_bias = bias[batch_indices, out_indices] * rs_high_mask

#             r_high_ = rank[self.perm_high]
#             out_mask = Variable((r_high_ == hidden_rank.unsqueeze(-1)).any(dim=1)).to(dtype=torch.float64)
#             masked_scale = out_mask * scale # (b,out)
#             masked_bias = out_mask * bias # (b,out)
#             masked_rs_high = r_high_ * (r_high_ == hidden_rank.unsqueeze(-1)).any(dim=1) # (in,) * (b,in) -> (b,in)
#         if self.is_W1:
#             masked_rs_low = hidden_rank
#         else:
#             r_low_ = rank[self.perm_low]
#             masked_rs_low = r_low_ * (r_low_ == hidden_rank.unsqueeze(-1)).any(dim=1) # (in,) * (b,in) -> (b,in)

#         if self.is_V:
#             weight_mask = ((masked_rs_low.unsqueeze(1) < masked_rs_high.unsqueeze(-1)) \
#                             & (masked_rs_low.unsqueeze(1) != 0) \
#                             & (masked_rs_high.unsqueeze(-1) != 0)).to(dtype=x.dtype)
#         else:
#             weight_mask = ((masked_rs_low.unsqueeze(1) <= masked_rs_high.unsqueeze(-1)) \
#                             & (masked_rs_low.unsqueeze(1) != 0) \
#                             & (masked_rs_high.unsqueeze(-1) != 0)).to(dtype=x.dtype) # (b,out,in)
            
#         weight_mask = Variable(weight_mask)
#         masked_weight = weight_mask * weight # (b,out,in)
#         x_ = masked_scale * (masked_weight @ x.unsqueeze(-1)).squeeze(-1) + masked_bias # (b,out)
        
#         return x_, mask, pre_mask, hidden_rank, pre_masked_idx


# arbitrary marginal masked linear layer
class AMMaskedLinear(nn.Module):
    def __init__(self, 
                 d, 
                 in_features, 
                 out_features, 
                 units, 
                 dh_per_unit, 
                 num_outlayers, 
                 r_low=None, 
                 r_high=None, 
                 is_W1=False, 
                 is_V=False):
        super(AMMaskedLinear, self).__init__()
        self.d = d
        self.in_features = in_features
        self.out_features = out_features
        self.units = units
        self.dh_per_unit = dh_per_unit
        self.num_outlayers = num_outlayers
        self.is_W1 = is_W1
        self.is_V = is_V
        self.r_low = r_low
        self.r_high = r_high
        self.direction = nn.Parameter(torch.Tensor(out_features, in_features).to(torch.float64))
        # if not self.is_V:
        #     self.cscale = nn.Linear(1, (self.d-1)*self.units, dtype=torch.float64) # (b,out=(d-1)*units)
        #     self.cbias = nn.Linear(1, (self.d-1)*self.units, dtype=torch.float64) # (b,out=(d-1)*units)
        # else:
        #     self.cscale = nn.Linear(1, out_features, dtype=torch.float64)
        #     self.cbias = nn.Linear(1, out_features, dtype=torch.float64)
        self.cscale = nn.Linear(1, out_features).to(torch.float64)
        self.cbias = nn.Linear(1, out_features).to(torch.float64)
        self.reset_parameters()

    def reset_parameters(self):
        self.direction.data.normal_(0, 0.001)
        self.cscale.weight.data.normal_(0, 0.001)
        self.cbias.weight.data.normal_(0, 0.001)
        
    def forward(self, input):
        x, mask, pre_mask, hidden_rank = input # x (b,d), mask (b,d)

        zeros = Variable(torch.FloatTensor(x.size(0), 1).zero_()).to(x.device, dtype=x.dtype)
        scale = self.cscale(zeros) # (b,out)
        bias = self.cbias(zeros) # (b,out)
        weight = self.direction # (out,in)

        r_low_ = self.r_low.to(device=x.device) if self.r_low != None else None
        r_high_ = self.r_high.to(device=x.device) if self.r_high != None else None

        if self.is_V:
            out_mask = mask.repeat_interleave(self.num_outlayers, dim=1)
            final_layer_hidden_rank = ((mask - pre_mask) * (self.d + 1)).int() + hidden_rank 
            masked_rs_high = final_layer_hidden_rank.repeat_interleave(self.num_outlayers, dim=1)
        else:
            out_mask = (r_high_ == hidden_rank.unsqueeze(-1)).any(dim=1)
            masked_rs_high = r_high_ * out_mask # (in,) * (b,in) -> (b,in)
        if self.is_W1:
            masked_rs_low = hidden_rank
        else:
            masked_rs_low = r_low_ * (r_low_ == hidden_rank.unsqueeze(-1)).any(dim=1) # (in,) * (b,in) -> (b,in)

        if self.is_V:
            weight_mask = ((masked_rs_low.unsqueeze(1) < masked_rs_high.unsqueeze(-1)) \
                            & (masked_rs_low.unsqueeze(1) != 0) \
                            & (masked_rs_high.unsqueeze(-1) != 0)).to(dtype=x.dtype)
        else:
            weight_mask = ((masked_rs_low.unsqueeze(1) <= masked_rs_high.unsqueeze(-1)) \
                            & (masked_rs_low.unsqueeze(1) != 0) \
                            & (masked_rs_high.unsqueeze(-1) != 0)).to(dtype=x.dtype) # (b,out,in)
        
        out_mask = Variable(out_mask.to(dtype=mask.dtype))
        weight_mask = Variable(weight_mask)
        masked_scale = out_mask * scale # (b,out)
        masked_bias = out_mask * bias # (b,out)
        masked_weight = weight_mask * weight # (b,out,in)
        x_ = masked_scale * (masked_weight @ x.unsqueeze(-1)).squeeze(-1) + masked_bias # (b,out)
        
        return x_, mask, pre_mask, hidden_rank