import dgl
import torch
import numpy as np
import torch.nn as nn
from dgl.nn.pytorch.conv import GATv2Conv

class GATv2(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()

        hidden_dim = kwargs.get('hidden_dim', 64)
        num_layers = kwargs.get('num_layers', 8)
        n_heads = kwargs.get('n_heads', 4)

        self.num_layers = num_layers

        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.gatv2 = GATv2Conv(2*hidden_dim, hidden_dim, n_heads, activation=torch.tanh)
        self.g_embed = nn.Linear(hidden_dim, output_dim)

    def forward(self, g):
        g = np.array(g).reshape((1, -1)).tolist()[0]
        g = dgl.batch(g)
        h_0 = self.linear_in(g.ndata["feat"].float().squeeze(dim=1))
        h = h_0
        for _ in range(self.num_layers):
            h = self.gatv2(g, torch.cat([h, h_0], dim=1)).sum(dim=1)
        g.ndata['h'] = h
        g.ndata["is_root"] = g.ndata["is_root"].float()
        hg = dgl.sum_nodes(g, 'h', weight='is_root')
        return self.g_embed(hg)

