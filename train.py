import random
from collections import deque

import funcy as fn
import torch
from tqdm import tqdm

from dfa_sampler import gen_mutated_sequential_reach_avoid
from dfa_embeddings.dataloader import gen_problems, target_dist
from dfa_embeddings.model import DFAEncoder, ActionPredictor
from dfa_embeddings.model import DFATranformerEncoder
from dfa_embeddings.utils import Graph
from dfa_embeddings.utils import dfa2graph
from dfa_embeddings.utils import rand_sat
from dfa_embeddings.utils import rand_sat_per_state




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
        yield from gen_mutated_sequential_reach_avoid(n_tokens=n_tokens)

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
