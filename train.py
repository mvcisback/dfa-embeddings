from typing import Any
from dataclasses import dataclass

import funcy as fn
import numpy as np
import torch
from bidict import bidict
from dfa import DFA, dfa2dict
from tqdm import tqdm

from dfa_embeddings.dfa_sampler import sample_dfas
from dfa_embeddings.dataloader import gen_problems, target_dist
from dfa_embeddings.model import DFAEncoder, ActionPredictor


@dataclass
class Graph:
    # N = Number of states + Non-Stuttering Transitions.
    # M = Number of tokens.
    # M + 3 node types: accepting/rejecting/token
    # [ is_accepting, is_rejecting, token_1, ... ]
    node_features: Any  # 1 x N x (M + w).
    adj_matrix: Any     # N x N.


def dfa2graph(d: DFA):
    dfa_dict, start = dfa2dict(d)

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
    node_features = np.zeros((n_nodes, 2 + n_tokens))
    for s1, (label, transitions) in dfa_dict.items():
        idx1 = node2idx[s1]
        adj[idx1, idx1] = 1  # Represents stutter.
        node_features[idx1, int(label)] = 1
        for token, s2 in transitions.items():
            if s1 == s2: continue
            idx2 = node2idx[s2]
            idx12 = node2idx[s1, s2]

            # Connect s1 to s2.
            adj[idx2, idx12] = 1
            adj[idx12, idx1] = 1

            # Note token leading to s2.
            node_features[idx12, 2 + token2idx[token]] = 1

    adj = adj.reshape(n_nodes, n_nodes, 1)
    return Graph(adj_matrix=torch.Tensor(adj).to(device),
                 node_features=torch.Tensor(node_features).to(device))


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


torch.set_default_device(device)


def train(n_iters=1_000_000, n_tokens=12):
    dfa_encoder = DFAEncoder(n_tokens=n_tokens,
                             output_dim=8,
                             hidden_dim=16,
                             depth=6,
                             n_heads=2)
    model = ActionPredictor(dfa_encoder)

    # TODO: Switch to adam.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # TODO: clean up data loader API.
    # TODO: make dataloader deterministic via seed.
    def my_dfa_sampler(rng=None):
        yield from sample_dfas(n_tokens=n_tokens)

    dataloader = gen_problems(my_dfa_sampler)

    running_loss = 0
    for iter in tqdm(range(n_iters)):
        problem, answer = next(dataloader)
        optimizer.zero_grad()

        graph1, graph2 = map(dfa2graph, problem)
        # TODO: Handle conjunctive distribution using seperate head.
        distinguish_distr, _ = answer
        distinguish_distr = torch.from_numpy(distinguish_distr).to(device)

        # TODO: model should directly take in dfa!
        prediction = model(graph1.node_features, graph1.adj_matrix,
                           graph2.node_features, graph2.adj_matrix)

        loss = ((distinguish_distr - prediction)**2).mean()
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if iter % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(iter + 1, last_loss))
            running_loss = 0 

    torch.save(model.state_dict(), "pytorch_model")


if __name__ == "__main__":
    train()
