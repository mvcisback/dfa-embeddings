import torch
import random
import numpy as np
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

@dataclass
class Graph:
    # N = Number of states + Non-Stuttering Transitions.
    # M = Number of tokens.
    # M + 3 node types: accepting/rejecting/token
    # [ psat, is_accepting, is_rejecting, token_1, ... ]
    node_features: Any  # 1 x N x (M + 3).
    adj_matrix: Any     # N x N.


def rand_sat(d: DFA, start: Optional[int]=None,
             n_samples=10_000, eoe_prob: float =1/32):
    if start is None: start = d.start
    tokens = list(d.inputs)
    count = 0.0
    for _ in range(n_samples):
        word = []
        while random.random() < eoe_prob:
            word.append(random.choice(tokens))
            count += int(d.label(word, start=start))
    return count / n_samples


def rand_sat_per_state(d: DFA, **kwargs):
    # Randomly sample words and 
    return {s: rand_sat(d, start=s, **kwargs) for s in d.states()}


@lru_cache
def dfa2graph(d: DFA):
    dfa_dict, start = dfa2dict(d)
    state2psat = rand_sat_per_state(d)
    # Create indexing for tokens.
    inputs = sorted(d.inputs)  # Force unique edge features w. canonical order.
    n_tokens = len(d.inputs)
    token2idx = bidict({t: idx for idx, t in enumerate(d.inputs)})

    # Create indexing for nodes.
    #  * Nodes are of the form state or (state1, state2).
    #  * (state1, state2) is only present for non-stuttering transitions.
    #  * First index is always the start.
    n_states = len(dfa_dict)
    node2idx = bidict()
    node2idx[start] = 0
    idx = 1
    for s in dfa_dict.keys():
        if s == start: continue
        node2idx[s] = idx
        idx += 1

    for state, (_, transitions) in dfa_dict.items():
        for token, state2 in transitions.items():
            if state2 == state: continue
            if (state, state2) in node2idx: continue
            node2idx[(state, state2)] = idx
            idx += 1

    # Fill in adj matrix and node features
    n_nodes = len(node2idx)
    adj = np.zeros((n_nodes, n_nodes))
    node_features = np.zeros((n_nodes, 3 + n_tokens))
    for s1, (label, transitions) in dfa_dict.items():
        idx1 = node2idx[s1]
        adj[idx1, idx1] = 1  # Represents stutter.
        node_features[idx1, int(label)] = 1
        node_features[idx1, 2] = state2psat[s1]
        for token, s2 in transitions.items():
            if s1 == s2: continue
            idx2 = node2idx[s2]
            idx12 = node2idx[s1, s2]

            # Connect s1 to s2.
            adj[idx2, idx12] = 1
            adj[idx12, idx1] = 1

            # Note token leading to s2.
            node_features[idx12, 3 + token2idx[token]] = 1

    adj = adj.reshape(n_nodes, n_nodes, 1)
    return Graph(adj_matrix=torch.Tensor(adj).to(device),
                 node_features=torch.Tensor(node_features).to(device))