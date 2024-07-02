import dgl
import torch
import numpy as np
import torch.nn as nn
from dgl.nn.pytorch.conv import GATv2Conv
from dgl.nn import EdgeWeightNorm
# from torch_geometric.nn.conv import GATv2Conv

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity



class GATv2_cdfa(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()

        hidden_dim = kwargs.get('hidden_dim', 64)
        num_layers = kwargs.get('num_layers', 8)
        n_heads = kwargs.get('n_heads', 4)

        self.num_layers = num_layers

        self.linear_in = nn.Linear(input_dim, hidden_dim)
        # self.gatv2 = GATv2Conv(2*hidden_dim, hidden_dim, n_heads, edge_dim=6, fill_value=None, add_self_loops=False)
        # self.map = nn.Linear(n_heads*hidden_dim, hidden_dim)
        self.gatv2 = GATv2Conv(2*hidden_dim, hidden_dim, n_heads)
        self.g_embed = nn.Linear(hidden_dim, output_dim)

        self.n_heads = n_heads

    def _update_h_given_weights(self, g, weights):
        n = len(weights)
        graphs = dgl.unbatch(g)
        for i in range(n):
            # print(graphs[i].ndata['h'])
            graphs[i].ndata['h'] = torch.matmul(weights[i], graphs[i].ndata['h'])
            # print(graphs[i].ndata['h'])
            # print(graphs[i].ndata['h'].shape, n)
            # input()
        return dgl.batch(graphs)



    def forward(self, g):
        g = np.array(g).reshape((1, -1)).tolist()[0]
        # weights = [_g.ndata["weights"] for _g in g]
        g = dgl.batch(g, ndata=["feat", "is_root"])
        h_0 = self.linear_in(g.ndata["feat"].squeeze(dim=1))


        # g.ndata['h'] = h_0
        # g = self._update_h_given_weights(g, weights)
        # h_0 = g.ndata['h']

        h = h_0

        # input(">>>>")

        for _ in range(self.num_layers):
            # h = self.gatv2(torch.cat([h, h_0], dim=1), edge_index, edge_attr=edge_attr).reshape(h_shape[0], self.n_heads, h_shape[1]).sum(dim=1)
            # h = self.map(self.gatv2(torch.cat([h, h_0], dim=1), edge_index, edge_attr=edge_attr))
            # h = self.gatv2(g, torch.cat([h, h_0], dim=1), edge_probs).sum(dim=1)


            # #### update ####
            # g.ndata['h'] = h
            # h[agg_idx] = dgl.sum_nodes(g, 'h', weight='is_root')
            # ################

            # g.ndata['h'] = h
            # g = self._update_h_given_weights(g, weights)
            # h = g.ndata['h']

            h = self.gatv2(g, torch.cat([h, h_0], dim=1)).sum(dim=1)

        
        # #### update ####
        # g.ndata['h'] = h
        # h[agg_idx] = dgl.sum_nodes(g, 'h', weight='is_root')
        # ################
        # h_agg = h[agg_idx]
        # return self.g_embed(h_agg)
        # h = torch.matmul(weights, h)




        # g.ndata['h'] = h
        # hg = dgl.sum_nodes(g, 'h', weight='is_root')
        # return self.g_embed(hg)

        h = self.g_embed(h)
        g.ndata['h'] = h
        hg = dgl.sum_nodes(g, 'h', weight='is_root')
        return hg

# import dgl
# import torch
# import numpy as np
# import torch.nn as nn
# import dgl.function as fn
# from dgl.nn.pytorch.conv import GATv2Conv

# class GATv2_cdfa(nn.Module):
#     def __init__(self, input_dim, output_dim, **kwargs):
#         super().__init__()

#         hidden_dim = kwargs.get('hidden_dim', 64)
#         num_layers = kwargs.get('num_layers', 2)
#         n_heads = kwargs.get('n_heads', 2)

#         self.num_layers = num_layers

#         self.linear_in = nn.Linear(35, hidden_dim)
#         self.gatv2 = GATv2Conv(2*hidden_dim, hidden_dim, n_heads, activation=torch.tanh)
#         self.g_embed = nn.Linear(hidden_dim, output_dim)

#     def forward(self, g):
#         g = np.array(g).reshape((1, -1)).tolist()[0]
#         g = dgl.batch(g)

#         print(g.ndata["feat"])
#         g.update_all(fn.u_mul_e('feat', 'prob', 'm'), fn.sum('m', 'feat'))

#         print(g.ndata["feat"])
#         input()
#         h_0 = self.linear_in(g.ndata["feat"].float().squeeze(dim=1))
#         h = h_0
#         for _ in range(self.num_layers):
#             h = self.gatv2(g, torch.cat([h, h_0], dim=1)).sum(dim=1)
#         g.ndata['h'] = h
#         g.ndata["is_root"] = g.ndata["is_root"].float()
#         hg = dgl.sum_nodes(g, 'h', weight='is_root')
#         return self.g_embed(hg)


class CustomGATv2Conv(GATv2Conv):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=True, share_weights=False):
        super(CustomGATv2Conv, self).__init__(in_feats, out_feats, num_heads, feat_drop, attn_drop, negative_slope, residual, activation, allow_zero_in_degree, bias, share_weights)
        self.norm = EdgeWeightNorm(norm='right')

    def forward(self, graph, feat, edge_probs=None, get_attention=False):
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
                h_src = p*h_dst
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_dst).view(
                        -1, self._num_heads, self._out_feats
                    )
                print(graph)
                print(h_dst.shape)
                print(feat_dst.shape)
                input()
                if graph.is_block:
                    feat_dst = feat_dst[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
            graph.srcdata.update(
                {"el": feat_src}
            )  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({"er": feat_dst})
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(
                graph.edata.pop("e")
            )  # (num_src_edge, num_heads, out_dim)
            e = (
                (e * self.attn).sum(dim=-1).unsqueeze(dim=2)
            )  # (num_edge, num_heads, 1)
            # compute softmax
            graph.edata["a"] = self.attn_drop(
                edge_softmax(graph, e)
            )  # (num_edge, num_heads)
            # message passing
            graph.update_all(fn.u_mul_e("el", "a", "m"), fn.sum("m", "ft"))
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
