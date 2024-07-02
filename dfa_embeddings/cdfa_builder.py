# import ring
import random
import numpy as np

import dgl
import torch
import networkx as nx
from copy import deepcopy
from dfa import DFA, dict2dfa
import dfa_embeddings.utils as utils

edge_types = {k:v for (v, k) in enumerate(["self", "normal-to-temp", "temp-to-normal", "AGG", "OR", "AND"])}
feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "AGG":-5, "OR": -6, "AND": -7}

class cDFABuilder(object):
    def __init__(self, n_tokens=12, alphabet_type='deterministic'):
        super(cDFABuilder, self).__init__()
        # self.dfa2vec = dfa2vec
        self.n_tokens = n_tokens
        self.alphabet_type = alphabet_type
        self.feature_size = n_tokens + len(feature_inds)

    # # To make the caching work.
    # def __ring_key__(self):
    #     return "cDFABuilder"

    def __call__(self, dfa_goal, device=None):
        return self._to_graph_cdfa(dfa_goal)

    # @ring.lru(maxsize=400000)
    def _to_graph_cdfa(self, dfa_goal):
        nxg_goal = []
        nxg_goal_or_nodes = []
        rename_goal = []
        agg_node_weights = {}
        # _print = False
        for i, dfa_clause in enumerate(dfa_goal):
            nxg_clause = []
            nxg_init_nodes = []
            rename_clause = []
            for j, dfa in enumerate(dfa_clause):
                dfa_dict, state_belief = dfa
                nxg = self.dfa_dict_to_nxg(dfa_dict).copy() # Must copy this because of the cache. With no copy, you'll see unusual behavior.
                agg_node = "AGG"
                nxg.add_node(agg_node, feat=np.array([[0.0] * self.feature_size]))
                nxg.nodes[agg_node]["feat"][0][feature_inds["AGG"]] = 1.0
                node_prefix = str(i) + "_" + str(j) + "_"
                for n in dfa_dict.keys():
                    node = str(n)
                    if node_prefix + agg_node not in agg_node_weights:
                        agg_node_weights[node_prefix + agg_node] = []
                    agg_node_weights[node_prefix + agg_node].append((node_prefix + node, state_belief[n]))
                    assert state_belief[n] == 1 or state_belief[n] == 0
                    if state_belief[n] == 1:
                        # if n > 0:
                        #     _print = True
                        nxg.add_edge(node, agg_node, type=edge_types["AGG"])

                nxg_init_nodes.append(str(j) + "_" + agg_node)
                nxg_clause.append(nxg)
                rename_clause.append(str(j) + "_")

            if nxg_clause != []:
                composed_nxg_clause = nx.union_all(nxg_clause, rename=rename_clause)
                or_node = "OR"
                composed_nxg_clause.add_node(or_node, feat=np.array([[0.0] * self.feature_size]))
                composed_nxg_clause.nodes[or_node]["feat"][0][feature_inds["OR"]] = 1.0
                for nxg_init_node in nxg_init_nodes:
                    composed_nxg_clause.add_edge(nxg_init_node, or_node, type=edge_types["OR"])
                nxg_goal.append(composed_nxg_clause)
                rename_goal.append(str(i) + "_")
                nxg_goal_or_nodes.append(str(i) + "_" + or_node)

        if nxg_goal != []:
            composed_nxg_goal = nx.union_all(nxg_goal, rename=rename_goal)
        else:
            composed_nxg_goal = nx.DiGraph()

        and_node = "AND"
        composed_nxg_goal.add_node(and_node, feat=np.array([[0.0] * self.feature_size]))
        nx.set_node_attributes(composed_nxg_goal, np.array([0.0]), "is_root")
        composed_nxg_goal.nodes[and_node]["is_root"] = np.array([1.0])
        composed_nxg_goal.nodes[and_node]["feat"][0][feature_inds["AND"]] = 1.0

        for or_node in nxg_goal_or_nodes:
            composed_nxg_goal.add_edge(or_node, and_node, type=edge_types["AND"])

        for node in composed_nxg_goal.nodes:
            # if "AGG" not in node:
            composed_nxg_goal.add_edge(node, node, type=edge_types["self"])

        nxg = composed_nxg_goal

        # nx.set_node_attributes(nxg, np.array([[0.0] * nxg.number_of_nodes()]), "weights")
        n = nxg.number_of_nodes()
        nodes = list(nxg.nodes())
        for i, node in enumerate(nodes):
            nxg.nodes[node]["weights"] = np.array([0.0] * n)
            if node in agg_node_weights:
                for _node, _weight in agg_node_weights[node]:
                    nxg.nodes[node]["weights"][nodes.index(_node)] = _weight
            else:
                nxg.nodes[node]["weights"][i] = 1.0

        # if _print:
        #     for i, node in enumerate(nxg.nodes):
        #         print(node, nxg.nodes[node])

        #     for edge in nxg.edges():
        #         print(edge, nxg.edges[edge])

        #     utils.draw(nxg)
        #     input()

        return utils.nx2dgl(nxg)

    # @ring.lru(maxsize=1000000)
    def dfa_dict_to_nxg(self, dfa_dict):
        nxg = nx.DiGraph()
        new_node_name_counter = 0
        new_node_name_base_str = "temp_"

        for s in dfa_dict.keys():
            start = str(s)
            nxg.add_node(start)
            nxg.nodes[start]["feat"] = np.array([[0.0] * self.feature_size])
            nxg.nodes[start]["feat"][0][feature_inds["normal"]] = 1.0
            is_accepting, transitions = dfa_dict[s]
            if is_accepting: # is accepting?
                nxg.nodes[start]["feat"][0][feature_inds["accepting"]] = 1.0
            elif sum(s != e for e in transitions.values()) == 0: # is rejecting?
                nxg.nodes[start]["feat"][0][feature_inds["rejecting"]] = 1.0
            embeddings = {}
            for a, e in transitions.items():
                if s == e:
                    continue # We define self loops later when composing graphs
                end = str(e)
                if end not in embeddings.keys():
                    embeddings[end] = np.zeros(self.feature_size)
                    embeddings[end][feature_inds["temp"]] = 1.0 # Since it is a temp node
                embeddings[end][a] = 1.0
            for end in embeddings.keys():
                new_node_name = new_node_name_base_str + str(new_node_name_counter)
                new_node_name_counter += 1
                nxg.add_node(new_node_name, feat=np.array([embeddings[end]]))
                nxg.add_edge(end, new_node_name, type=edge_types["normal-to-temp"])
                nxg.add_edge(new_node_name, start, type=edge_types["temp-to-normal"])

        return nxg
