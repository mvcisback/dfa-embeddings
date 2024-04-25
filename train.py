import random
from collections import deque
from functools import lru_cache
from typing import Any, Optional
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
from dfa_embeddings.model import DFATranformerEncoder



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


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


torch.set_default_device(device)


def train(n_iters=1_000_000, n_tokens=12):
    #dfa_encoder = DFATranformerEncoder(n_tokens=n_tokens,
    #                                   output_dim=256,
    #                                   hidden_dim=256,
    #                                   depth=6,
    #                                   n_heads=2)
    dfa_encoder = DFAEncoder(n_tokens=n_tokens,
                             output_dim=128,
                             hidden_dim=128,
                             depth=6,
                             n_heads=4)

    model = ActionPredictor(dfa_encoder)
    model.train(True)

    # TODO: Switch to adam.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # TODO: clean up data loader API.
    # TODO: make dataloader deterministic via seed.
    def my_dfa_sampler(rng=None):
        yield from sample_dfas(n_tokens=n_tokens)

    dataloader = gen_problems(my_dfa_sampler)

    def problem2target(problem):
        distiguishing_dfa = problem[0] ^ problem[1]
        graph1, graph2 = map(dfa2graph, problem)

        tokens = range(n_tokens)
        target = torch.zeros(len(tokens) + 1) 
        
        target[-1] = rand_sat(distiguishing_dfa)
        for idx, token in enumerate(tokens):
            start = distiguishing_dfa.transition((token,))
            target[idx] = rand_sat(distiguishing_dfa, start=start)
        return -torch.log(target + 0.0000001)

    test_set = fn.take(200, dataloader)
    test_set = [(p, problem2target(p)) for p in test_set]

    def eval_problem(model, problem, target):
        graph1, graph2 = map(dfa2graph, problem)

        # TODO: model should directly take in dfa!
        prediction = model(graph1.node_features, graph1.adj_matrix,
                           graph2.node_features, graph2.adj_matrix)

        return ((target - prediction)**2).mean()

    running_loss = 0
    replay_buffer = deque(maxlen=1_000)
    for iter in tqdm(range(n_iters)):
        optimizer.zero_grad()

        if (replay_buffer.maxlen > len(replay_buffer)) or (iter % 100 == 0):
            problem = next(dataloader)
            replay_buffer.append((problem, problem2target(problem)))

        problem, target = random.choice(replay_buffer)
        loss = eval_problem(model, problem, target)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if iter % 500 == 499:
            last_loss = running_loss / 500 # loss per batch
            print('  batch {} loss: {}'.format(iter + 1, last_loss))
            running_loss = 0 
        if iter % 1000 == 999:
            with torch.no_grad():
                test_loss = sum(eval_problem(model, *x) for x in test_set) / len(test_set)
                print('  test loss: {}'.format(test_loss))

    torch.save(model.state_dict(), "pytorch_model")


if __name__ == "__main__":
    train()
