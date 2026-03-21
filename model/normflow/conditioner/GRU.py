import torch
import torch.nn as nn

 
 
class GRUConditioner(nn.Module):
    def __init__(self, config):
        super(GRUConditioner).__init__()
        self.rnn_hid_dim = config["rnn_hid_dim"]
        self.n_rnn_layers = config["n_rnn_layers"]
        self.rnn = nn.GRU(1, self.rnn_hid_dim, num_layers=self.n_rnn_layers, batch_first=True)

    def forward(self, input):
        # x: shape (b,d)
        # rnn conditioner
        # x_: shape (b,d+1,1), add margin element x_u^0 = -1, h^0 = 0
        x, mask, logdet = input
        x_margin = torch.tensor([[0.0]] * x.size(0), device=x.device, requires_grad=False) # (b,1)
        x_ = torch.cat((x_margin, x), dim=1).unsqueeze(2) # (b,d+1)
        o, _ = self.rnn(x_)  # RNN output: shape (b,d+1,hid)
        o = o[:, :-1, :] # only get first bdim dimensions: shape (b,d,hid)
        # o0 -> none, o1 -> x0, o2 -> x0, x1
        masked_o = o * mask.unsqueeze(-1) # (b,d,hid) * (b,d) -> (b,d,hid)

        return x, masked_o, mask, logdet
 