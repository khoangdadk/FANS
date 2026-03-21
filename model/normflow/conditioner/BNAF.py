# https://github.com/nicola-decao/BNAF

import math

import torch
import torch.nn as nn


 
class BNAFConditioner(nn.Module):
    """
    https://github.com/nicola-decao/BNAF
    Class that extends ``torch.nn.Sequential`` for constructing a Block Neural Normalizing Flow.
    """
    def __init__(self, config, res):
        """
        Parameters
        ----------
        *args : ``Iterable[torch.nn.Module]``, required.
            The modules to use.
        res : ``str``, optional (default = None).
            Which kind of residual connection to use. ``res = None`` is no residual
            connection, ``res = 'normal'`` is ``x + f(x)`` and ``res = 'gated'`` is
            ``a * x + (1 - a) * f(x)`` where ``a`` is a learnable parameter.
        """
        super(BNAFConditioner, self).__init__()
        self.dim = config["dim"]
        self.hidden_dim = config["hidden_dim"]
        self.config = config

        layers = [BNAFMaskedWeight(self.dim, self.dim * self.hidden_dim, dim=self.dim), Tanh()]
        for _ in range(self.config["layers"] - 1):
            layers.append(BNAFMaskedWeight(self.dim * self.hidden_dim, self.dim * self.hidden_dim, dim=self.dim))
            layers.append(Tanh())
        layers += [BNAFMaskedWeight(self.dim * self.hidden_dim, self.dim, dim=self.dim)]
        self.bnaf = nn.Sequential(*layers)
        self.res = res
        if self.res == "gated":
            self.gate = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(1)))

    def forward(self, input):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """
        x, mask, logdet = input
        grad = None

        bnaf_output = (x, mask)
        for module in self.bnaf:
            bnaf_output, grad = module(bnaf_output, grad)
            grad = grad if len(grad.shape) == 4 else grad.view(grad.shape + [1, 1])

        output, _ = bnaf_output
        
        grad_f = grad.squeeze()
        if grad.shape[1] == 1:
            grad_f = grad_f.unsqueeze(1)
        
        grad_f = grad_f.view(x.size(0), x.size(1))

        if self.res == "normal":
            output = x + output
            grad_f = torch.nn.functional.softplus(grad_f)
        elif self.res == "gated":
            output = self.gate.sigmoid() * output + (1 - self.gate.sigmoid()) * x
            grad_f = torch.nn.functional.softplus(grad_f + self.gate) - torch.nn.functional.softplus(self.gate)
        
        output *= mask
        grad_f *= mask
        
        logdet += grad_f.sum(-1)
        
        return output, mask, logdet

    def _get_name(self):
        return "BNAFConditioner(res={})".format(self.res)
    


class BNAFMaskedWeight(nn.Module):
    """
    Module that implements a linear layer with block matrices with positive diagonal blocks.
    Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.
    """
    def __init__(self, in_features, out_features, dim, bias=True):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features per each dimension ``dim``.
        out_features : ``int``, required.
            The number of output features per each dimension ``dim``.
        dim : ``int``, required.
            The number of dimensions of the input of the flow.
        bias : ``bool``, optional (default = True).
            Whether to add a parametrizable bias.
        """
        super(BNAFMaskedWeight, self).__init__()
        self.in_features, self.out_features, self.dim = in_features, out_features, dim

        weight = torch.zeros(out_features, in_features, dtype=torch.float64)
        for i in range(dim):
            weight[
                i * out_features // dim : (i + 1) * out_features // dim,
                0 : (i + 1) * in_features // dim,
            ] = torch.nn.init.xavier_uniform_(
                torch.Tensor(out_features // dim, (i + 1) * in_features // dim)
            )
        self._weight = torch.nn.Parameter(weight)
        self._diag_weight = torch.nn.Parameter(
            torch.nn.init.uniform_(torch.Tensor(out_features, 1).to(dtype=torch.float64)).log()
        )
        self.bias = (
            torch.nn.Parameter(
                torch.nn.init.uniform_(
                    torch.Tensor(out_features).to(dtype=torch.float64),
                    -1 / math.sqrt(out_features),
                    1 / math.sqrt(out_features),
                )
            )
            if bias
            else 0
        )
        
        mask_d = torch.zeros_like(weight)
        for i in range(dim):
            mask_d[
                i * (out_features // dim) : (i + 1) * (out_features // dim),
                i * (in_features // dim) : (i + 1) * (in_features // dim),
            ] = 1
        self.register_buffer("mask_d", mask_d)

        mask_o = torch.ones_like(weight)
        for i in range(dim):
            mask_o[
                i * (out_features // dim) : (i + 1) * (out_features // dim),
                i * (in_features // dim) :,
            ] = 0

        self.register_buffer("mask_o", mask_o)

    def get_weights(self):
        """
        Computes the weight matrix using masks and weight normalization.
        It also compute the log diagonal blocks of it.
        """
        w = torch.exp(self._weight) * self.mask_d + self._weight * self.mask_o
        w_squared_norm = (w ** 2).sum(-1, keepdim=True)
        w = self._diag_weight.exp() * w / w_squared_norm.sqrt()
        wpl = self._diag_weight + self._weight - 0.5 * torch.log(w_squared_norm)

        return w.t(), wpl.t()[self.mask_d.bool().t()].view(
            self.dim, self.in_features // self.dim, self.out_features // self.dim
        )

    def forward(self, input, grad=None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal block of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
        transformations combined with this transformation.
        """
        x, mask = input
        w, wpl = self.get_weights()
        g = wpl.transpose(-2, -1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # (b,d,out,1), out: dim per variable
        output = x.matmul(w) + self.bias
        mask_ = mask.repeat_interleave(output.size(1)//mask.size(1), dim=1)
        output = output * mask_
        if grad is not None:
            grad_ = torch.logsumexp(g.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3), -1)
        else:
            grad_ = g
        grad_ = grad_ * mask.unsqueeze(-1).unsqueeze(-1)

        return (
            (output, mask),
            grad_
        )

    def __repr__(self):
        return "BNAFMaskedWeight(in_features={}, out_features={}, dim={}, bias={})".format(
            self.in_features,
            self.out_features,
            self.dim,
            not isinstance(self.bias, int),
        )



class Tanh(nn.Tanh):
    """
    Class that extends ``torch.nn.Tanh`` additionally computing the log diagonal
    blocks of the Jacobian.
    """
    def forward(self, input, grad=None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal blocks of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
        transformations combined with this transformation.
        """
        x, mask = input
        g = -2 * (x - math.log(2) + torch.nn.functional.softplus(-2 * x))
        output = torch.tanh(x)
        mask_ = mask.repeat_interleave(output.size(1)//mask.size(1), dim=1)
        output = output * mask_
        if grad is not None:
            grad_ = g.view(grad.shape) + grad
        else:
            grad_ = g
        grad_ = grad_ * mask.unsqueeze(-1).unsqueeze(-1)

        return (
            (output, mask),
            grad_,
        )