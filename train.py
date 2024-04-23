from typing import Any
from dataclasses import dataclass

import funcy as fn
import numpy as np
import torch
from bidict import bidict
from dfa import DFA, dfa2dict

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
    return Graph(adj_matrix=torch.Tensor(adj),
                 node_features=torch.Tensor(node_features))


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)




def train(n_iters=100):
    dfa_encoder = DFAEncoder(n_tokens=4,
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
        yield from sample_dfas(n_tokens=12)

    dataloader = gen_problems(my_dfa_sampler)
    loss_fn = torch.nn.MSELoss()

    running_loss = 0
    for iter, (problem, answer) in zip(range(n_iters), dataloader):
        optimizer.zero_grad()

        graph1, graph2 = map(dfa2graph, problem)
        # TODO: Handle conjunctive distribution using seperate head.
        distinguish_distr, _ = answer

        # TODO: model should directly take in dfa!
        prediction = model(graph1.node_features, graph1.adj_matrix,
                           graph2.node_features, graph2.adj_matrix)

        loss = loss_fn(prediction, distinguish_distr)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0 


if __name__ == "__main__":
    # TODO: convert to unit tests.
    def transition(state, token):
        match (state, token):
            case (_, "lava"):  return "dead"
            case ("dry", "y"): return "done"
            case ("dry", "b"): return "wet"
            case ("wet", "y"): return "dead"
            case ("wet", "g"): return "dry"
            case (s, _):       return s

    d1 = DFA(start="dry",
             inputs={"r", "g", "b", "y"},
             label=lambda s: s == "done",
             transition=transition)

    graph1 = dfa2graph(d1)
    graph2 = dfa2graph(~d1)

    dfa_encoder = DFAEncoder(n_tokens=4,
                             output_dim=8,
                             hidden_dim=16,
                             depth=6,
                             n_heads=2)
    predictor = ActionPredictor(dfa_encoder)
    prediction = predictor(graph1.node_features, graph1.adj_matrix,
                           graph2.node_features, graph2.adj_matrix)
    target = torch.Tensor(target_dist(d1, ~d1, True))
    loss = torch.nn.MSELoss()
    print(loss(prediction, target))

    train()
    """
    N = len(d.states())
    predict = np.ones(1 + N) / N
    for state in d.states():
        print(state)
        d2 = evolve(d, start=state)
        print(target)
        print(loss(target, predict))

    def my_dfa_sampler(rng=None):
        if rng is None: rng = random.Random(100)
        while True:
            gen_dfas = dfa_mutate.generate_mutations(d)
            dfas = fn.take(10, fn.distinct(gen_dfas))
            rng.shuffle(dfas)
            yield from dfas
    """
