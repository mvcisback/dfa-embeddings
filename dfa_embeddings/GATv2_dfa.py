import dgl
import torch
import numpy as np
import torch.nn as nn
from dgl.nn.pytorch.conv import GATv2Conv
# from torch_geometric.nn.conv import GATv2Conv

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity

class GATv2_dfa(nn.Module):
    def __init__(self, input_dim, output_dim, batch_size, **kwargs):
        super().__init__()

        hidden_dim = kwargs.get('hidden_dim', 64)
        num_layers = kwargs.get('num_layers', 8)
        n_heads = kwargs.get('n_heads', 4)

        self.batch_size = batch_size

        self.num_layers = num_layers

        # self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.gatv2_first = CustomGATv2Conv((3+5, 3), hidden_dim, 1)
        self.linear_in_condition = nn.Linear(5, hidden_dim)
        # self.gatv2 = GATv2Conv(2*hidden_dim, hidden_dim, n_heads, edge_dim=hidden_dim)
        self.gatv2 = CustomGATv2Conv((3*hidden_dim, 2*hidden_dim), hidden_dim, 1)
        # self.gatv2 = CustomGATv2Conv((hidden_dim+3+5, hidden_dim + 3), hidden_dim, n_heads)
        # self.gatv2 = CustomGATv2Conv(2*hidden_dim, hidden_dim, n_heads)
        # self.map = nn.Linear(n_heads*hidden_dim, hidden_dim) # initialize as diagonal
        self.g_embed = nn.Linear(hidden_dim, output_dim)

        # self.is_root_weights = nn.Linear(output_dim*2, 1)
        # self.is_root_weights = nn.Linear(output_dim+1, 1)

        # self.project = nn.Linear(1, output_dim)

        # self.log_softmax = nn.LogSoftmax(dim=0)

        self.n_heads = n_heads

        # self.eps = 1e-7

    def forward(self, g):
        g = np.array(g).reshape((1, -1)).tolist()[0]
        g = dgl.batch(g)
        edge_attr = g.edata["condition"].squeeze(dim=1)
        feat_0 = g.ndata["feat"].squeeze(dim=1)
        h_0 = self.gatv2_first(g, feat_0, edge_attr=edge_attr).squeeze(dim=1)
        # h_0 = self.linear_in(g.ndata["feat"].squeeze(dim=1))
        h = h_0
        h_shape = h.shape
        edge_attr = self.linear_in_condition(edge_attr)
        
        # probs = g.edata["prob"]
        for _ in range(self.num_layers):
            # h = self.map(self.gatv2(g, torch.cat([h, h_0], dim=1), edge_attr=edge_attr).reshape(h_shape[0], self.n_heads*h_shape[1]))
            # h = self.gatv2(g, torch.cat([h, h_0], dim=1), edge_attr=edge_attr).sum(dim=1)
            # h = self.gatv2(g, torch.cat([h, feat_0], dim=1), edge_attr=edge_attr).squeeze(dim=1)
            h = self.gatv2(g, torch.cat([h, h_0], dim=1), edge_attr=edge_attr).squeeze(dim=1)
        # h = self.g_embed(h)
        g.ndata['h'] = h
        # hg = []
        # for g_i in dgl.unbatch(g):
        #     h_i = g_i.ndata['h']
        #     is_root = g_i.ndata['is_root']
        #     # is_root_proj = self.project(is_root)
        #     # scores = self.is_root_weights(torch.cat([h_i, is_root_proj], dim=1))
        #     scores = self.is_root_weights(torch.cat([h_i, torch.log(is_root + self.eps)], dim=1)) # Put log prob
        #     attention = torch.exp(self.log_softmax(scores)) * is_root
        #     z = torch.matmul(attention.T, h_i)
        #     hg.append(z)
        # hg = torch.cat(hg, dim=0)
        # hg = dgl.sum_nodes(g, 'h', weight='is_root')
        # return hg
        hg = dgl.sum_nodes(g, 'h', weight='is_root')
        return self.g_embed(hg)

# import dgl
# import torch
# import numpy as np
# import torch.nn as nn
# from dgl.nn.pytorch.conv import GATv2Conv

# class GATv2_dfa(nn.Module):
#     def __init__(self, input_dim, output_dim, **kwargs):
#         super().__init__()

#         hidden_dim = kwargs.get('hidden_dim', 64)
#         num_layers = kwargs.get('num_layers', 8)
#         n_heads = kwargs.get('n_heads', 4)

#         self.num_layers = num_layers

#         self.linear_in = nn.Linear(input_dim, hidden_dim)
#         self.gatv2 = GATv2Conv(2*hidden_dim, hidden_dim, n_heads, activation=torch.tanh)
#         self.g_embed = nn.Linear(hidden_dim, output_dim)

#     def forward(self, g):
#         g = np.array(g).reshape((1, -1)).tolist()[0]
#         g = dgl.batch(g)
#         h_0 = self.linear_in(g.ndata["feat"].squeeze(dim=1))
#         h = h_0
#         for _ in range(self.num_layers):
#             h = self.gatv2(g, torch.cat([h, h_0], dim=1)).sum(dim=1)
#         h = self.g_embed(h)
#         g.ndata['h'] = h
#         # hg = dgl.sum_nodes(g, 'h', weight='is_root')
#         return g
#         # g.ndata['h'] = h
#         # g.ndata["is_root"] = g.ndata["is_root"].float()
#         # hg = dgl.sum_nodes(g, 'h', weight='is_root')
#         # return self.g_embed(hg)

class CustomGATv2Conv(GATv2Conv):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=True, share_weights=False):
        super(CustomGATv2Conv, self).__init__(in_feats, out_feats, num_heads, feat_drop, attn_drop, negative_slope, residual, activation, allow_zero_in_degree, bias, share_weights)
        pass

    def _concat_node_messages_with_edge_attr(self, edges):
        return {'h_src': torch.cat((edges.src['h_src'], edges.data['edge_attr']), dim=1)}

    def _compute_message(self, edges):
        return {'m': edges.data['el']*edges.data['a']}


    def forward(self, graph, feat, edge_attr=None, get_attention=False):
        r"""
        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                feat_dst = self.fc_dst(h_dst).view(
                    -1, self._num_heads, self._out_feats
                )
            else:
                h_dst = self.feat_drop(feat)
                h_src = feat
                # print(h_src.shape)
                # input()
                graph.srcdata.update({"h_src": h_src})
                graph.edata["edge_attr"] = edge_attr
                graph.apply_edges(self._concat_node_messages_with_edge_attr)
                h_src = graph.edata["h_src"]
                h_src = self.feat_drop(h_src)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_dst).view(
                        -1, self._num_heads, self._out_feats
                    )
                if graph.is_block:
                    feat_dst = feat_dst[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
            graph.edata.update({"el": feat_src})
            graph.dstdata.update({'er': feat_dst})
            graph.apply_edges(fn.e_add_v("el", "er", "e"))
            e = self.leaky_relu(
                graph.edata.pop("e")
            )  # (num_src_edge, num_heads, out_dim)
            e = (
                (e * self.attn).sum(dim=-1).unsqueeze(dim=2)
            )  # (num_edge, num_heads, 1)

            # probs = probs.view(probs.shape[0], 1, 1)
            # e = e + probs

            # compute softmax
            graph.edata["a"] = self.attn_drop(
                edge_softmax(graph, e)
            )  # (num_edge, num_heads)
            # message passing
            graph.update_all(self._compute_message, fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(
                    h_dst.shape[0], -1, self._out_feats
                )
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst
