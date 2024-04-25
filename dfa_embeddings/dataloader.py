import random
from attr import evolve
from collections import deque, defaultdict
from itertools import chain, product

import dfa
import dfa_mutate
import funcy as fn
import numpy as np
import scipy as sp
from dfa import DFA
from dfa_embeddings.dfa_sampler import sample_dfas


def target_dist(d1: DFA, d2: DFA, distinguish: bool):
    d = (d1 ^ d2) if distinguish else (d1 & d2)
    return _target_dist(d)


def _target_dist(d: DFA):
    N = len(d.inputs)
    # N-tokens + End of string.
    costs = 2 * N * np.ones(N+1)
    temp = np.sqrt(N) / 2

    if d._label(d.start):
        costs[N] = 1.0  # End of string.
        return sp.special.softmax(-costs / temp)

    # Construct inverse transition function.
    inv_transition = defaultdict(set)
    for state in d.states():
        for token in d.inputs:
            state2 = d._transition(state, token)
            inv_transition[state2].add((state, token))

    # BFS from the accepting states to the start state.
    queue = deque([s for s in d.states() if d._label(s)])
    depths = {s: 0 for s in queue}
    while queue:
        state = queue.pop()
        depth = depths[state]
        for (state2, token) in inv_transition[state]:
            if state2 in depths:
                continue  # Visited in BFS -> already have depth.
            depths[state2] = depth + 1
            queue.appendleft(state2)
    one_step_reachable = [d._transition(d.start, t) for t in d.inputs]
    costs[:N] = [depths.get(s, 2 * N - 1) + 1 for s in one_step_reachable]
    return sp.special.softmax(-costs / temp)


def loss(predicted, target):
    return sp.special.rel_entr(target, predicted).sum()


def gen_problems(dfa_sampler, rng=None):
    for d1, d2 in fn.pairwise(dfa_sampler(rng)):
        for start1, start2 in product(d1.states(), d2.states()):
            _d1 = evolve(d1, start=start1)
            _d2 = evolve(d2, start=start2)
            yield (_d1, _d2)
