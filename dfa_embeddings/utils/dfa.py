import dgl
import torch
import random
import numpy as np
import networkx as nx
from bidict import bidict

from dataclasses import dataclass
from typing import Any, Optional
from dfa import DFA, dfa2dict
from functools import lru_cache

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.set_default_device(device)

def gen_geometric(p=0.5):
    while True:
        yield np.random.geometric(p=p)

def cDFA_sampler(dfa_sampler, and_gen=gen_geometric(p=0.5), or_gen=gen_geometric(p=0.5)):
    while True:
        # yield tuple(tuple(next(dfa_sampler) for _ in range(next(or_gen))) for _ in range(next(and_gen)))
        yield tuple(tuple(next(dfa_sampler) for _ in range(1)) for _ in range(2))

def get_state_belief(dfa, state_belief, truth_assignment):
    n_states = state_belief.size
    n_tokens = truth_assignment.size
    transition_matrix = np.zeros((n_states, n_states))
    for s in range(n_states):
        for a in range(n_tokens):
            e = dfa._transition(s, a)
            transition_matrix[s][e] += truth_assignment[a]
    return np.matmul(state_belief, transition_matrix)

def nx2dgl(nxg):

    edges = list(nxg.edges)
    nodes = list(nxg.nodes)
    edge_type_attributes = nx.get_edge_attributes(nxg, "type")
    edge_condition_attributes = nx.get_edge_attributes(nxg, "condition")
    # edge_prob_attributes = nx.get_edge_attributes(nxg, "prob")

    # U, V, _type, _prob = zip(*[(nodes.index(edge[0]), nodes.index(edge[1]), edge_type_attributes[edge], edge_prob_attributes[edge]) for edge in edges])
    # U, V, _type, _condition, _prob = zip(*[(nodes.index(edge[0]), nodes.index(edge[1]), edge_type_attributes[edge], edge_condition_attributes[edge], edge_prob_attributes[edge]) for edge in edges])
    U, V, _type, _condition = zip(*[(nodes.index(edge[0]), nodes.index(edge[1]), edge_type_attributes[edge], edge_condition_attributes[edge]) for edge in edges])
    # _feat, _weights, _is_root, _is_agg = zip(*[(nxg.nodes[node]["feat"], nxg.nodes[node]["weights"], nxg.nodes[node]["is_root"], nxg.nodes[node]["is_agg"]) for node in nodes])
    # _feat, _weights, _is_root = zip(*[(nxg.nodes[node]["feat"], nxg.nodes[node]["weights"], nxg.nodes[node]["is_root"]) for node in nodes])
    _feat, _is_root = zip(*[(nxg.nodes[node]["feat"], nxg.nodes[node]["is_root"]) for node in nodes])

    U = torch.from_numpy(np.array(U))
    V = torch.from_numpy(np.array(V))
    _type = torch.from_numpy(np.array(_type))
    _condition = torch.from_numpy(np.array(_condition))
    # _prob = torch.from_numpy(np.array(_prob))
    _feat = torch.from_numpy(np.array(_feat))
    # _weights = torch.from_numpy(np.array(_weights))
    _is_root = torch.from_numpy(np.array(_is_root))
    # _is_agg = torch.from_numpy(np.array(_is_agg))

    g = dgl.graph((U, V))
    g.ndata["feat"] = _feat.float()
    # g.ndata["weights"] = _weights.float()
    g.ndata["is_root"] = _is_root.float()
    # g.ndata["is_agg"] = _is_agg.float()
    g.edata["type"] = _type
    g.edata["condition"] = _condition.float()
    # g.edata["prob"] = _prob.float()

    return g

def draw(G):
    from networkx.drawing.nx_agraph import graphviz_layout
    import matplotlib.pyplot as plt

    colors = ["black", "red", "green", "blue", "purple", "orange"]
    edge_color = [colors[i] for i in nx.get_edge_attributes(G,'type').values()]

    pos=graphviz_layout(G, prog='dot')
    labels = G.nodes
    nx.draw(G, pos, with_labels=True, arrows=True, node_shape='s', edgelist=list(nx.get_edge_attributes(G,'type')), node_size=500, node_color="white", edge_color=edge_color) #edge_color=edge_color
    plt.show()
