import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from gymnasium.spaces import Box, Discrete


class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[], scales=None, activation=nn.Tanh()):
        super().__init__()

        layer_dims = [in_dim] + hiddens
        self.num_layers = len(layer_dims)
        self.enc_ = nn.Sequential(*[fc(in_dim, out_dim, activation=activation)
            for (in_dim, out_dim) in zip(layer_dims, layer_dims[1:])])

        self.discrete_ = nn.Sequential(
            nn.Linear(layer_dims[-1], out_dim)
        )

    def forward(self, obs):
        x = self.enc_(obs)
        x = self.discrete_(x)
        return Categorical(logits=F.log_softmax(x, dim=1))


def fc(in_dim, out_dim, activation=nn.Tanh()):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        activation
    )
