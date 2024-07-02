import ring
import random
import numpy as np
from scipy.special import log_softmax, softmax

import dgl
import torch
import networkx as nx
from copy import deepcopy
from dfa import DFA, dict2dfa
import dfa_embeddings.utils as utils

feature_inds = {k:v for (v, k) in enumerate(["rejecting", "accepting", "dontcare"])}
edge_types = {k:v for (v, k) in enumerate(["transition"])}

class DFABuilder(object):
    def __init__(self, n_tokens=12, alphabet_type='deterministic'):
        super(DFABuilder, self).__init__()
        self.n_tokens = n_tokens
        self.alphabet_type = alphabet_type
        self.feature_size = len(feature_inds)

    # To make the caching work.
    def __ring_key__(self):
        return "DFABuilder"

    def __call__(self, dfa, device=None):
        return self._to_graph_dfa(dfa)

    # @ring.lru(maxsize=400000)
    def _to_graph_dfa(self, dfa):
        dfa_dict, state_belief = dfa
        nxg = self.dfa_dict_to_nxg(dfa_dict)

        # agg_node = "AGG"
        # nxg.add_node(agg_node)
        # nxg.nodes[agg_node]["feat"] = np.array([[0.0] * self.feature_size])
        # nxg.nodes[agg_node]["feat"][0][feature_inds["AGG"]] = 1.0

        # temp = np.log(state_belief)

        # print(temp)
        # print(softmax(temp))
        # input()


        # for n in dfa_dict.keys():
            # node = str(n)
            # nxg.add_edge(node, agg_node, type=edge_types["transition"], condition=np.array([np.zeros(self.n_tokens)]))

        # nx.set_node_attributes(nxg, np.array([0.0]), "is_root")
        # nxg.nodes[agg_node]["is_root"] = np.array([1.0])

        nxg = self._write_state_beliefs_to_nxg(state_belief, nxg)

        # if state_belief.nonzero()[0][0] > 0:
        #     for i, node in enumerate(nxg.nodes):
        #         print(node, nxg.nodes[node])

        #     for edge in nxg.edges():
        #         print(edge, nxg.edges[edge])

        #     utils.draw(nxg)
        #     input()

        return utils.nx2dgl(nxg)

    @ring.lru(maxsize=1000000)
    def dfa_dict_to_nxg(self, dfa_dict):
        nxg = nx.DiGraph()
        # new_node_name_counter = 0
        # new_node_name_base_str = "temp_"

        for s in dfa_dict.keys():
            start = str(s)
            nxg.add_node(start)
            # nxg.nodes[start]["id"] = s
            nxg.nodes[start]["feat"] = np.array([[0.0] * self.feature_size])
            is_accepting, transitions = dfa_dict[s]
            if is_accepting: # is accepting?
                nxg.nodes[start]["feat"][0][feature_inds["accepting"]] = 1.0
            elif sum(s != e for e in transitions.values()) == 0: # is rejecting?
                nxg.nodes[start]["feat"][0][feature_inds["rejecting"]] = 1.0
            else:
                nxg.nodes[start]["feat"][0][feature_inds["dontcare"]] = 1.0
            embeddings = {}
            for a, e in transitions.items():
                # if s == e:
                #     continue # We define self loops later when composing graphs
                end = str(e)
                if end not in embeddings.keys():
                    embeddings[end] = np.zeros(self.n_tokens)
                embeddings[end][a] = 1.0
            for end in embeddings.keys():
                nxg.add_edge(end, start, type=edge_types["transition"], condition=np.array([embeddings[end]]))

        return nxg

    def _write_state_beliefs_to_nxg(self, state_belief, nxg):
        nx.set_node_attributes(nxg, np.array([0.0]), "is_root")
        for node in nxg.nodes:
            # try:
            s = int(node)
            nxg.nodes[node]["is_root"] = np.array([state_belief[s]])
            # except:
            #     pass
            # nxg.add_edge(node, node, type=edge_types["transition"], condition=np.array([np.zeros(self.n_tokens)]))
        return nxg
    #     nx.set_node_attributes(nxg, np.array([0.0]), "is_root")
    #     if self.alphabet_type == 'probabilistic':
    #         for node in nxg.nodes:
    #             try:
    #                 s = int(node)
    #                 nxg.nodes[node]["is_root"] = np.array([state_belief[s]])
    #             except:
    #                 pass
    #             # nxg.add_edge(node, node, type=edge_types["transition"], condition=np.array([np.zeros(self.n_tokens)]))
    #     else:
    #         node = str(state_belief)
    #         nxg.nodes[node]["is_root"] = np.array([1.0])
    #         # for node in nxg.nodes:
    #         #     nxg.add_edge(node, node, type=edge_types["transition"], condition=np.array([np.zeros(self.n_tokens)]))
    #     return nxg


# import ring
# import random
# import numpy as np

# import dgl
# import torch
# import networkx as nx
# from copy import deepcopy
# from dfa import DFA, dict2dfa
# import dfa_embeddings.utils as utils

# feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4}
# edge_types = {k:v for (v, k) in enumerate(["self", "normal-to-temp", "temp-to-normal"])}

# class DFABuilder(object):
#     def __init__(self, n_tokens=12, alphabet_type='deterministic'):
#         super(DFABuilder, self).__init__()
#         self.n_tokens = n_tokens
#         self.alphabet_type = alphabet_type
#         self.feature_size = n_tokens + len(feature_inds)

#     # To make the caching work.
#     def __ring_key__(self):
#         return "DFABuilder"

#     def __call__(self, dfa, device=None):
#         return self._to_graph_dfa(dfa)

#     # @ring.lru(maxsize=400000)
#     def _to_graph_dfa(self, dfa):
#         dfa_dict, state_belief = dfa
#         nxg = self.dfa_dict_to_nxg(dfa_dict)
#         nxg = self._write_state_beliefs_to_nxg(state_belief, nxg)
#         return utils.nx2dgl(nxg)

#     @ring.lru(maxsize=1000000)
#     def dfa_dict_to_nxg(self, dfa_dict):
#         nxg = nx.DiGraph()
#         new_node_name_counter = 0
#         new_node_name_base_str = "temp_"

#         for s in dfa_dict.keys():
#             start = str(s)
#             nxg.add_node(start)
#             nxg.nodes[start]["id"] = s
#             nxg.nodes[start]["feat"] = np.array([[0.0] * self.feature_size])
#             nxg.nodes[start]["feat"][0][feature_inds["normal"]] = 1.0
#             is_accepting, transitions = dfa_dict[s]
#             if is_accepting: # is accepting?
#                 nxg.nodes[start]["feat"][0][feature_inds["accepting"]] = 1.0
#             elif sum(s != e for e in transitions.values()) == 0: # is rejecting?
#                 nxg.nodes[start]["feat"][0][feature_inds["rejecting"]] = 1.0
#             embeddings = {}
#             for a, e in transitions.items():
#                 if s == e:
#                     continue # We define self loops later when composing graphs
#                 end = str(e)
#                 if end not in embeddings.keys():
#                     embeddings[end] = np.zeros(self.feature_size)
#                     embeddings[end][feature_inds["temp"]] = 1.0 # Since it is a temp node
#                 embeddings[end][a] = 1.0
#             for end in embeddings.keys():
#                 new_node_name = new_node_name_base_str + str(new_node_name_counter)
#                 new_node_name_counter += 1
#                 nxg.add_node(new_node_name, feat=np.array([embeddings[end]]))
#                 nxg.nodes[new_node_name]["id"] = -1
#                 nxg.add_edge(end, new_node_name, type=edge_types["normal-to-temp"])
#                 nxg.add_edge(new_node_name, start, type=edge_types["temp-to-normal"])

#         return nxg

#     def _write_state_beliefs_to_nxg(self, state_belief, nxg):
#         nx.set_node_attributes(nxg, np.array([0.0]), "is_root")
#         if self.alphabet_type == 'probabilistic':
#             for node in nxg.nodes:
#                 try:
#                     s = int(node)
#                     nxg.nodes[node]["is_root"] = np.array([state_belief[s]])
#                 except:
#                     pass
#                 nxg.add_edge(node, node, type=edge_types["self"])
#         else:
#             node = str(state_belief)
#             nxg.nodes[node]["is_root"] = np.array([1.0])
#             for node in nxg.nodes:
#                 nxg.add_edge(node, node, type=edge_types["self"])
#         return nxg
