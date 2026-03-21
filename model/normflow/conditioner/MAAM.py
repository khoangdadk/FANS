import numpy as np

import torch
import torch.nn as nn

from ...utils import Lambda
from ...nn import AMMaskedLinear
    

     
class MAAMConditioner(nn.Module):
    def __init__(self, config): 
        super(MAAMConditioner, self).__init__()
        self.dim = config["dim"]
        self.dh_per_unit = config["dh_per_unit"]
        self.hid_dim = config["hid_dim"] 
        self.units = config["units"] 
        self.num_hid_layers = config["num_hid_layers"]
        self.num_outlayers = config["num_outlayers"]
        self.activation = Lambda(lambda x: (nn.ELU()(x[0]), *x[1:]))
        
        # get node rank 
        rs = get_rank_partial(self.dim, self.hid_dim, self.dh_per_unit, self.num_hid_layers)
        rs = [torch.from_numpy(r) for r in rs]

        self.input_to_first_hid = nn.Sequential(AMMaskedLinear(d=self.dim, 
                                                               in_features=self.dim, 
                                                               out_features=self.hid_dim, 
                                                               units=self.units, 
                                                               dh_per_unit=self.dh_per_unit,
                                                               num_outlayers=self.num_outlayers,
                                                               r_low=None, 
                                                               r_high=rs[0], 
                                                               is_W1=True, 
                                                               is_V=False), 
                                                               self.activation)
        hid_sequels = list()
        for i in range(self.num_hid_layers-1):
            hid_sequels.append(AMMaskedLinear(d=self.dim, 
                                              in_features=self.hid_dim, 
                                              out_features=self.hid_dim, 
                                              units=self.units, 
                                              dh_per_unit=self.dh_per_unit,
                                              num_outlayers=self.num_outlayers,
                                              r_low=rs[i], 
                                              r_high=rs[i+1], 
                                              is_W1=False, 
                                              is_V=False))
            hid_sequels.append(self.activation)
        self.hids = nn.Sequential(*hid_sequels)
        self.last_hid_to_output = AMMaskedLinear(d=self.dim, 
                                                 in_features=self.hid_dim, 
                                                 out_features=self.num_outlayers*self.dim, 
                                                 units=self.units,
                                                 dh_per_unit=self.dh_per_unit,
                                                 num_outlayers=self.num_outlayers,
                                                 r_low=rs[-1], 
                                                 r_high=None, 
                                                 is_W1=False, 
                                                 is_V=True)
    
    def push_fwd_collider_rank(self, mask, rank_in_unit, pos_in_nonzeros):
        rank = torch.cummax(rank_in_unit - pos_in_nonzeros, dim=-1)[0] + pos_in_nonzeros
        return mask * rank

    def push_bwd_collider_rank(self, mask, rank_in_unit, last_nonzero_pos, rev_pos_in_nonzeros):
        excess_alt = torch.minimum(rank_in_unit[torch.arange(rank_in_unit.shape[0], device=mask.device), last_nonzero_pos], 
                                   torch.tensor(self.dh_per_unit))
        rank = torch.minimum(rank_in_unit, excess_alt.unsqueeze(1) - rev_pos_in_nonzeros)
        return mask * rank

    def forward(self, input):
        x, mask, logdet = input
        # create hidden mask for arbitrary margin
 
        # create pre-masks
        pre_mask = mask.scatter(1, (mask.cumsum(1).argmax(1)).unsqueeze(1), 0)

        # rescale from (d-1)-scale to m-scale m=dh_per_unit: rank' = ceil(m*rank/(d-1))
        rank_in_unit = torch.ceil(self.dh_per_unit*torch.arange(1, self.dim+1, device=mask.device)/(self.dim-1))
        pre_masked_rank_in_unit = pre_mask * rank_in_unit

        pre_marginal_nonzero_cnt = pre_mask.sum(1)
        pre_marginal_last_nonzero_pos = self.dim - 1 - pre_mask.flip(1).argmax(1)
        pre_marginal_pos_in_nonzeros = pre_mask.cumsum(1) * pre_mask - 1 
        pre_marginal_rev_pos_in_nonzeros = pre_marginal_nonzero_cnt.unsqueeze(1) - 1 - pre_marginal_pos_in_nonzeros

        hidden_rank = self.push_fwd_collider_rank(pre_mask, 
                                                  pre_masked_rank_in_unit, 
                                                  pre_marginal_pos_in_nonzeros)
        hidden_rank = self.push_bwd_collider_rank(pre_mask, 
                                                  hidden_rank, 
                                                  pre_marginal_last_nonzero_pos, 
                                                  pre_marginal_rev_pos_in_nonzeros)
        hidden_rank = hidden_rank.int()

        first_hid_out = self.input_to_first_hid((x, mask, pre_mask, hidden_rank))
        last_hid_out = self.hids(first_hid_out)
        masked_o = self.last_hid_to_output(last_hid_out)[0] # (b,ol*d)
        masked_o = masked_o.view(-1, x.size(1), self.num_outlayers) # (b,d,ol)
        
        return x, masked_o, mask, logdet
    


def get_hidden_perm(d, units, dh_per_unit, num_hid_layers):
    assert dh_per_unit <= d - 1
    def get_perm_each_layer():
        perm = torch.randperm(units*dh_per_unit)
        return perm
    return [get_perm_each_layer() for _ in range(num_hid_layers)]
    


def get_rank_partial(d, dh, dh_per_unit, num_hid_layers):
    assert dh_per_unit <= d - 1
    assert dh_per_unit <= dh
    def get_rank_each_layer():
        rl = np.array([])
        while len(rl) < dh:
            rl = np.concatenate([rl, np.arange(dh_per_unit)])
        excess = len(rl) - dh
        remove_ind = np.random.choice(dh_per_unit, excess, False)
        rl = np.delete(rl, remove_ind)
        np.random.shuffle(rl)
        rl += 1
        rl = rl.astype('float64')
        return rl
    return [get_rank_each_layer() for _ in range(num_hid_layers)]
        
 
 
def get_rank(d, dh, num_hid_layers, num_outlayers):
    def get_rank_each_layer(is_outlayer=False):
        if is_outlayer:
            rl = np.repeat(np.arange(1, d+1, dtype='float64'), num_outlayers)
        else:
            rl = np.array([])
            while len(rl) < dh:
                rl = np.concatenate([rl, np.arange(d-1)])
            excess = len(rl) - dh
            remove_ind = np.random.choice(d-1, excess, False)
            rl = np.delete(rl, remove_ind)
            np.random.shuffle(rl)
            rl += 2
            rl = rl.astype('float64')
        return rl
    rs = [np.arange(1, d+1, dtype='float64')]
    rs += [get_rank_each_layer(is_outlayer=False) for _ in range(num_hid_layers)]
    rs.append(get_rank_each_layer(is_outlayer=True))

    return rs