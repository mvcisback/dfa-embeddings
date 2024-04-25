# Based on: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/graphs/gatv2/__init__.py
import torch
from torch import nn


class GraphAttentionV2Layer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_heads: int = 1,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """
        # Number of nodes
        n_nodes = h.shape[0]

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)
        e = e.masked_fill(adj_mat == 0, float('-inf'))
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)

        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)


class DFAEncoder(nn.Module):
    def __init__(self,
                 n_tokens: int,
                 output_dim: int,
                 hidden_dim: int,
                 depth: int,
                 n_heads: int = 1):
        super().__init__()
        self.n_tokens = n_tokens
        self.output_dim = output_dim
        graph_layer = lambda: GraphAttentionV2Layer(in_features=hidden_dim,
                                                    out_features=hidden_dim,
                                                    n_heads=n_heads)
        self.pre = nn.Linear(3 + self.n_tokens, hidden_dim)
        self.graph_layers = [graph_layer() for _ in range(depth)]
        self.layer_norm =  nn.LayerNorm(hidden_dim)
        self.post = nn.Linear(hidden_dim, output_dim)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        h = self.pre(h)
        for layer in self.graph_layers:
            h = h + layer(h, adj_mat) 
            h = self.layer_norm(h)
        return self.post(h[0])  # index 0 is assumed to be the start.


class DFATranformerEncoder(nn.Module):
    def __init__(self,
                 n_tokens: int,
                 output_dim: int,
                 hidden_dim: int,
                 depth: int,
                 n_heads: int = 1):
        super().__init__()
        self.n_tokens = n_tokens
        self.output_dim = output_dim
        self.pre = nn.Linear(3 + self.n_tokens, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads),
            num_layers=depth,
        )
        self.post = nn.Linear(hidden_dim, output_dim)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        h = self.pre(h)
        # TODO: Apply adj_matrix or reachable matrix as mask.
        mask = None
        if adj_mat.shape[0] > 1:
            adj_mat = ~adj_mat.bool().squeeze()
            mask = torch.zeros_like(adj_mat)
            mask[adj_mat] = -float('inf')
        h = self.transformer(h, mask)
        return self.post(h[0])  # index 0 is assumed to be the start.



class ActionPredictor(nn.Module):
    def __init__(self, dfa_encoder: DFAEncoder):
        super().__init__()
        self.dfa_encoder = dfa_encoder
        n_tokens = self.dfa_encoder.n_tokens
        # Last token represents EOS.
        self.decoder = nn.Linear(2 * dfa_encoder.output_dim, n_tokens + 1)

    def forward(self, h1: torch.Tensor, adj_mat1: torch.Tensor,
                      h2: torch.Tensor, adj_mat2: torch.Tensor):
        h1 = self.dfa_encoder(h1, adj_mat1)
        h2 = self.dfa_encoder(h2, adj_mat2)
        return self.decoder(torch.cat([h1, h2]))
