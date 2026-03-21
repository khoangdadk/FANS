# https://github.com/lupalab/ace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MixtureSameFamily



class ProposalNetwork(nn.Module):
    def __init__(self, num_features, context_units=64, mixture_components=10, 
                 residual_blocks=4, hidden_units=512, activation="relu", dropout=0.0):
        super(ProposalNetwork, self).__init__()
        self.num_features = num_features
        self.context_units = context_units
        self.mixture_components = mixture_components
        self.hidden_units = hidden_units
        self.activation = getattr(F, activation)
        
        # Input layers
        self.input_layer = nn.Linear(num_features * 2, hidden_units)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_units, hidden_units)
            )
            for _ in range(residual_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_units, num_features * (3 * mixture_components + context_units))
        
    def forward(self, x_o, observed_mask):
        h = torch.cat([x_o, observed_mask], dim=-1) # shape: (batch_size, 2*xdim)
        h = self.input_layer(h) # shape: (batch_size, hid_dim)
        
        for block in self.residual_blocks:
            res = block(h)
            h = h + res # shape: (batch_size, hid_dim)
        
        h = self.activation(h)
        h = self.output_layer(h) # shape: (batch_size, xdim*(3 * mixture_components + context_units))
        h = h.view(-1, self.num_features, 3 * self.mixture_components + self.context_units)
        # shape: (batch_size, xdim, 3 * mixture_components + context_units)
        
        context = h[..., :self.context_units] # shape: (batch_size, xdim, context_units)
        params = h[..., self.context_units:] # shape: (batch_size, xdim, 3 * mixture_components)
        
        return context, params



class EnergyNetwork(nn.Module):
    def __init__(self, num_features, context_units, residual_blocks=4, hidden_units=128, 
                 activation="relu", dropout=0.0, energy_clip=30.0):
        super(EnergyNetwork, self).__init__()
        self.num_features = num_features
        self.context_units = context_units
        self.hidden_units = hidden_units
        self.activation = getattr(F, activation)
        self.energy_clip = energy_clip
        
        # Input layer
        self.input_layer = nn.Linear(1 + num_features + context_units, hidden_units)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_units, hidden_units)
            )
            for _ in range(residual_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_units, 1)
        
    def forward(self, x_u_i, u_i, context):
        # context: (batch_size * (num_importance_samples + 1) * xdim, context_units)
        # x_u_i, u_i: (batch_size * (num_importance_samples + 1) * xdim,)
        u_i_one_hot = F.one_hot(u_i, num_classes=self.num_features).float()
        # (batch_size * (num_importance_samples + 1) * xdim, xdim)
        # like: tensor([[1., 0., 0.],
        # [0., 1., 0.],
        # [0., 0., 1.],
        # [1., 0., 0.],
        # [0., 1., 0.],
        # [0., 0., 1.]])

        # will recheck this
        h = torch.cat([x_u_i.unsqueeze(-1), u_i_one_hot, context], dim=-1)
        # (batch_size * (num_importance_samples + 1) * xdim, 1 + xdim + context_units)

        h = self.input_layer(h) # (batch_size * (num_importance_samples + 1) * xdim, hid_dim)
        
        for block in self.residual_blocks:
            res = block(h)
            h = h + res
        
        h = self.activation(h)
        h = self.output_layer(h) # (batch_size * (num_importance_samples + 1) * xdim, 1)
        energies = F.softplus(h) # (batch_size * (num_importance_samples + 1) * xdim, 1)
        energies = torch.clamp(energies, 0.0, self.energy_clip)
        negative_energies = -energies
        
        return negative_energies



class ACEModel(nn.Module):
    def __init__(self, config, context_units=64, mixture_components=10, 
                 proposal_residual_blocks=4, proposal_hidden_units=512,
                 energy_residual_blocks=4, energy_hidden_units=128,
                 activation="relu", dropout=0.0, energy_clip=30.0,
                 energy_regularization=0.0):
        super(ACEModel, self).__init__()
        self.num_features = config["dim"]
        self.context_units = context_units
        self.energy_regularization = energy_regularization
        self.mixture_components = mixture_components
        
        self.proposal_network = ProposalNetwork(
            self.num_features, context_units, mixture_components, proposal_residual_blocks, 
            proposal_hidden_units, activation, dropout
        )
        
        self.energy_network = EnergyNetwork(
            self.num_features, context_units, energy_residual_blocks, energy_hidden_units, 
            activation, dropout, energy_clip
        )
        
        self.alpha = torch.tensor(1.0, requires_grad=False)

    
    def forward(self, x, observed_mask, missing_mask=None, num_importance_samples=10):
        x_o, x_u, observed_mask, query = self._process_inputs(x, observed_mask, missing_mask)

        # Step 1: proposal 
        context, params = self.proposal_network(x_o, observed_mask)
    
        logits = params[..., :self.mixture_components] # shape: (batch_size, xdim, mixture_components)
        means = params[..., self.mixture_components:-self.mixture_components] # shape: (batch_size, xdim, mixture_components)
        scales = F.softplus(params[..., -self.mixture_components:]) + 1e-3 # shape: (batch_size, xdim, mixture_components)
        
        components_dist = Normal(means, scales)
        proposal_dist = MixtureSameFamily(Categorical(logits=logits), components_dist)
        proposal_ll = proposal_dist.log_prob(x_u) * query # shape: (batch_size, xdim)

        # Step 2: energy
        proposal_samples = proposal_dist.sample((num_importance_samples,)) # shape: (num_importance_samples, batch_size, xdim)
        proposal_samples_proposal_ll = torch.stack([proposal_dist.log_prob(part) for part in proposal_samples]) # shape: (num_importance_samples, batch_size, xdim)
        
        proposal_samples = proposal_samples.permute(1, 0, 2) # shape: (batch_size, num_importance_samples, xdim)
        proposal_samples_proposal_ll = proposal_samples_proposal_ll.permute(1, 0, 2)
        proposal_samples *= query.unsqueeze(1) # query: shape (batch_size, xdim) -> (batch_size, 1, xdim)
        proposal_samples_proposal_ll *= query.unsqueeze(1)        
        
        x_u_i, u_i, tiled_context = self._get_energy_inputs(
            x_u, proposal_samples, num_importance_samples, context
        )
        negative_energies = self.energy_network(x_u_i, u_i, tiled_context) # (batch_size * (num_importance_samples + 1) * xdim, 1)

        negative_energies = negative_energies.view(
            -1,
            1 + num_importance_samples,
            self.num_features,
        )  # (batch_size, (num_importance_samples + 1), xdim)

        negative_energies *= query.unsqueeze(1)

        unnorm_energy_ll = negative_energies[:, 0] # (batch_size, xdim)
        proposal_samples_unnorm_energy_ll = negative_energies[:, 1:] # (batch_size, num_importance_samples, xdim)

        proposal_samples_log_ratios = proposal_samples_unnorm_energy_ll - proposal_samples_proposal_ll # (batch_size, num_importance_samples, xdim)

        log_normalizers = (
            torch.logsumexp(proposal_samples_log_ratios, dim=1)
            - torch.log(torch.tensor(num_importance_samples, dtype=torch.float32))
        ) * query # (batch_size, xdim)

        energy_ll = unnorm_energy_ll - log_normalizers
        
        return energy_ll, proposal_ll
    
    def _process_inputs(self, x, observed_mask, missing_mask):
        query = 1.0 - observed_mask

        if missing_mask is not None:
            missing_mask = missing_mask
            query *= 1.0 - missing_mask
            observed_mask *= 1.0 - missing_mask

        x_o = x * observed_mask
        x_u = x * query

        return x_o, x_u, observed_mask, query
    
    def _get_energy_inputs(self, x_u, proposal_samples, num_importance_samples, context):
        # x_u: shape (batch_size, xdim), proposal_samples: (batch_size, num_importance_samples, xdim)
        x_u_and_samples = torch.cat([x_u.unsqueeze(1), proposal_samples], dim=1) # (batch_size, num_importance_samples + 1, xdim)
        u_i = torch.arange(self.num_features, device=x_u.device).unsqueeze(0) # (1, xdim)
        u_i = u_i.repeat(x_u.size(0), 1 + num_importance_samples, 1) # (batch_size, 1 + num_importance_samples, xdim)
        
        tiled_context = context.unsqueeze(1).repeat(1, 1 + num_importance_samples, 1, 1)
        # shape: (batch_size, 1 + num_importance_samples, xdim, context_units)
        
        x_u_i = x_u_and_samples.view(-1) # (batch_size * (num_importance_samples + 1) * xdim,)
        u_i = u_i.view(-1) # (batch_size * (num_importance_samples + 1) * xdim,)
        tiled_context = tiled_context.view(-1, self.context_units) # (batch_size * (num_importance_samples + 1) * xdim, context_units)
        
        return x_u_i, u_i, tiled_context
    
