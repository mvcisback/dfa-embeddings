import random
from attr import evolve
from collections import deque, defaultdict
from itertools import chain

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
        for distinguish in (False, True):
            yield (d1, d2, distinguish), target_dist(d1, d2, distinguish)


if __name__ == "__main__":
    """
    # TODO: convert to unit tests.
    def transition(state, token):
        match (state, token):
            case (_, "lava"):  return "dead"
            case ("dry", "y"): return "done"
            case ("dry", "b"): return "wet"
            case ("wet", "y"): return "dead"
            case ("wet", "g"): return "dry"
            case (s, _):       return s

    d = DFA(start="dry",
            inputs={"r", "g", "b", "y"},
            label=lambda s: s == "done",
            transition=transition)
    N = len(d.states())
    predict = np.ones(1 + N) / N
    for state in d.states():
        print(state)
        d2 = evolve(d, start=state)
        target = _target_dist(d2)
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

    def my_dfa_sampler(rng=None):
        yield from sample_dfas(12)

    problems = gen_problems(my_dfa_sampler)

    print(next(problems))
