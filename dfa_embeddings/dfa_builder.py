import ring
import random
import numpy as np

import dgl
import torch
import networkx as nx
from copy import deepcopy
from dfa import DFA, dict2dfa
from dfa.utils import min_distance_to_accept_by_state

# feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "init": -5, "AND": -6, "OR": -7, "distance_normalized": -8}
# feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "init": -5, "AND": -6, "distance_normalized": -7}
feature_inds = {"rejecting": -1, "accepting": -2, "temp": -3, "normal": -4, "init": -5, "ROOT": -6}
edge_types = {k:v for (v, k) in enumerate(["self", "normal-to-temp", "temp-to-normal", "ROOT"])}
# edge_types = {k:v for (v, k) in enumerate(["self", "normal-to-temp", "temp-to-normal", "AND", "OR"])}

"""
A class that can take an DFA formula and generate the Abstract Syntax Tree (DFA) of it. This
code can generate trees in either Networkx or DGL formats. And uses caching to remember recently
generated trees.
"""
class DFABuilder(object):
    def __init__(self, n_tokens=12, compositional=False):
        super(DFABuilder, self).__init__()
        self.n_tokens = n_tokens
        self.compositional = compositional
        self.feature_size = n_tokens + len(feature_inds)

    # To make the caching work.
    def __ring_key__(self):
        return "DFABuilder"

    def __call__(self, dfa_goal, device=None):
        return self._to_graph(dfa_goal)

    def _to_graph(self, dfa_goal):
        if self.compositional:
            return self._to_graph_one_layer(dfa_goal)
        return self._to_graph_dfa(dfa_goal)

    @ring.lru(maxsize=400000)
    def _to_graph_dfa(self, dfa_dict):
        nxg, init_node = self.dfa_dict_to_nxg(*dfa_dict)

        # root_node = "ROOT"
        # nxg.add_node(root_node, feat=np.array([[0.0] * self.feature_size]))

        # nx.set_node_attributes(nxg, np.array([0.0]), "is_root")

        # nxg.nodes[root_node]["is_root"] = np.array([1.0])
        # nxg.nodes[root_node]["feat"][0][feature_inds["ROOT"]] = 1.0
        # nxg.nodes[root_node]["depth"] = 0

        # for node in nxg.nodes:
        #     if node != root_node and nxg.nodes[node]["feat"][0][feature_inds["temp"]] == 0:
        #         nxg.add_edge(node, root_node, type=edge_types["ROOT"])

        nx.set_node_attributes(nxg, np.array([0.0]), "is_root")
        nxg.nodes[init_node]["is_root"] = np.array([1.0])

        for node in nxg.nodes:
            nxg.add_edge(node, node, type=edge_types["self"])

        return self._get_dgl_graph(nxg)

    @ring.lru(maxsize=400000)
    def _to_graph_one_layer(self, dfa_goal):
        nxg_goal = []
        rename_goal = []
        nxg_init_nodes = []
        for i, dfa_clause in enumerate(dfa_goal):
            for _, dfa_dict in enumerate(dfa_clause):
                nxg, init_node = self.dfa_dict_to_nxg(*dfa_dict)
                nxg_goal.append(nxg)
                rename_goal.append(str(i) + "_")
                nxg_init_nodes.append(str(i) + "_" + init_node)

        if nxg_goal != []:
            composed_nxg_goal = nx.union_all(nxg_goal, rename=rename_goal)
        else:
            composed_nxg_goal = nx.DiGraph()

        and_node = "AND"
        composed_nxg_goal.add_node(and_node, feat=np.array([[0.0] * self.feature_size]))
        nx.set_node_attributes(composed_nxg_goal, np.array([0.0]), "is_root")
        composed_nxg_goal.nodes[and_node]["is_root"] = np.array([1.0])
        composed_nxg_goal.nodes[and_node]["feat"][0][feature_inds["AND"]] = 1.0
        composed_nxg_goal.nodes[and_node]["depth"] = max(composed_nxg_goal.nodes[init_node]["depth"] for init_node in nxg_init_nodes) + 1

        for init_node in nxg_init_nodes:
            composed_nxg_goal.add_edge(init_node, and_node, type=edge_types["AND"])

        for node in composed_nxg_goal.nodes:
            composed_nxg_goal.add_edge(node, node, type=edge_types["self"])

        nxg = composed_nxg_goal

        return self._get_dgl_graph(nxg)

    @ring.lru(maxsize=400000)
    def _to_graph_two_layers(self, dfa_goal):
        nxg_goal = []
        nxg_goal_or_nodes = []
        rename_goal = []
        for i, dfa_clause in enumerate(dfa_goal):
            nxg_clause = []
            nxg_init_nodes = []
            rename_clause = []
            for j, dfa_dict in enumerate(dfa_clause):
                nxg, init_node = self.dfa_dict_to_nxg(*dfa_dict)
                nxg_clause.append(nxg)
                rename_clause.append(str(j) + "_")
                nxg_init_nodes.append(str(j) + "_" + init_node)

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
            composed_nxg_goal.add_edge(node, node, type=edge_types["self"])

        nxg = composed_nxg_goal

        return self._get_dgl_graph(nxg)

    def _get_dgl_graph(self, nxg):

        edges = list(nxg.edges)
        nodes = list(nxg.nodes)
        edge_types_attributes = nx.get_edge_attributes(nxg, "type")

        U, V, _type = zip(*[(nodes.index(edge[0]), nodes.index(edge[1]), edge_types_attributes[edge]) for edge in edges])
        _feat, _is_root, _depth = zip(*[(nxg.nodes[node]["feat"], nxg.nodes[node]["is_root"], nxg.nodes[node]["depth"]) for node in nodes])

        U = torch.from_numpy(np.array(U))
        V = torch.from_numpy(np.array(V))
        _type = torch.from_numpy(np.array(_type))
        _feat = torch.from_numpy(np.array(_feat))
        _is_root = torch.from_numpy(np.array(_is_root))
        _depth = torch.from_numpy(np.array(_depth))

        g = dgl.graph((U, V))
        g.ndata["feat"] = _feat
        g.ndata["is_root"] = _is_root
        g.ndata["depth"] = _depth
        g.edata["type"] = _type

        # g.ndata["PE"] = dgl.lap_pe(g, k=2, padding=True)
        # g.ndata["PE"] = dgl.random_walk_pe(g, k=2)

        return g

    def min_distance_to_accept_by_state_normalized(self, dfa, state):
        from dfa.utils import min_distance_to_accept_by_state
        depths = min_distance_to_accept_by_state(dfa)
        if state in depths:
            return depths[state]/100.0
        return 1.0

    @ring.lru(maxsize=600000)
    def dfa_dict_to_nxg(self, dfa_dict, init_state):

        dfa = dict2dfa(dfa_dict, init_state)
        depths = min_distance_to_accept_by_state(dfa)

        nxg = nx.DiGraph()
        new_node_name_counter = 0
        new_node_name_base_str = "temp_"

        for s in dfa_dict.keys():
            start = str(s)
            nxg.add_node(start)
            nxg.nodes[start]["depth"] = 0
            nxg.nodes[start]["feat"] = np.array([[0.0] * self.feature_size])
            nxg.nodes[start]["feat"][0][feature_inds["normal"]] = 1.0
            # Assumption: We never do more than chain length 7-8 so deviding by 100 is safe.
            # nxg.nodes[start]["feat"][0][feature_inds["distance_normalized"]] = self.min_distance_to_accept_by_state_normalized(dfa, s)
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
                nxg.nodes[new_node_name]["depth"] = 0
                nxg.add_edge(end, new_node_name, type=edge_types["normal-to-temp"])
                nxg.add_edge(new_node_name, start, type=edge_types["temp-to-normal"])

        init_node = str(init_state)
        nxg.nodes[init_node]["feat"][0][feature_inds["init"]] = 1.0
        nxg.nodes[init_node]["depth"] = 2.0*depths[init_state]

        return nxg, init_node

def draw(G):
    from networkx.drawing.nx_agraph import graphviz_layout
    import matplotlib.pyplot as plt

    colors = ["black", "red", "green", "blue", "purple", "orange"]
    edge_color = [colors[i] for i in nx.get_edge_attributes(G,'type').values()]

    pos=graphviz_layout(G, prog='dot')
    # labels = nx.get_node_attributes(G,'token')
    labels = G.nodes
    nx.draw(G, pos, with_labels=True, arrows=True, node_shape='s', edgelist=list(nx.get_edge_attributes(G,'type')), node_size=500, node_color="white", edge_color=edge_color) #edge_color=edge_color
    plt.show()

