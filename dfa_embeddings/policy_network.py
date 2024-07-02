import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[], alphabet_type='deterministic', activation=nn.Tanh()):
        super().__init__()
        
        layer_dims = [in_dim] + hiddens
        self.layers = nn.ModuleList()
        
        # Create hidden layers
        for i in range(len(hiddens)):
            self.layers.append(fc(layer_dims[i], layer_dims[i+1], activation))
        
        self.alphabet_type = alphabet_type

        if self.alphabet_type == 'probabilistic':
            # Continuous action space (mu and std)
            self.encoder_mu = nn.Linear(layer_dims[-1], out_dim)
            self.encoder_std = nn.Linear(layer_dims[-1], out_dim)
            self.softplus = nn.Softplus()
        elif self.alphabet_type == 'deterministic':
            # Discrete action space (logits)
            self.encoder = nn.Linear(layer_dims[-1], out_dim)
        else:
            raise ValueError("alphabet_type must be either 'probabilistic' or 'deterministic'")

    def forward(self, obs):
        x = obs
        for layer in self.layers:
            x = layer(x)
        if self.alphabet_type == 'probabilistic':
            mu = self.encoder_mu(x)  # Ensure mu is between 0 and 1
            std = self.softplus(self.encoder_std(x)) + 1e-3
            return Normal(mu, std)
        elif self.alphabet_type == 'deterministic':
            logits = self.encoder(x)
            return Categorical(logits=F.log_softmax(logits, dim=1))

def fc(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        activation
    )
