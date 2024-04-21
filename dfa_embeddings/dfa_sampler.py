from collections import defaultdict
import random
import signal
from pathlib import Path
from pprint import pprint

import attr
import dfa
import funcy as fn
import numpy as np
from dfa import DFA
from dfa_identify import find_dfas
from dfa_mutate import sample_mutation
from tqdm import tqdm
from scipy.special import softmax


def reach_avoid_sampler(n_tokens, avg_size=5, prob_stutter=0.7):
    assert n_tokens > 1

    n = np.random.geometric(1/avg_size) + 2
    success, fail = n - 2, n - 1

    tokens = list(range(n_tokens))
    while True:
        transitions = {
          success: (True,  {t: success for t in range(n_tokens)}),
          fail:    (False, {t: fail    for t in range(n_tokens)}),
        }
        for state in range(n - 2):
            noop, good, bad = partition = (set(), set(), set())
            random.shuffle(tokens)
            good.add(tokens[0])
            bad.add(tokens[1])
            for token in tokens[2:]:
                if random.random() <= prob_stutter:
                    noop.add(token)
                else:
                    partition[random.randint(1, 2)].add(token)

            _transitions = dict()
            for token in good:
                _transitions[token] = state + 1
            for token in bad:
                _transitions[token] = fail
            for token in noop:
                _transitions[token] = state

            transitions[state] = (False, _transitions)

        yield dfa.dict2dfa(transitions, start=0).minimize()


def enumerate_dfas_given_size(n_tokens: int, size: int):
    alphabet = range(n_tokens)
    dfas = find_dfas(accepting=(),
                     rejecting=(),
                     order_by_stutter=True,
                     allow_unminimized=True,
                     alphabet=alphabet,
                     bounds=(size,size))
    dfas = (d.minimize() for d in dfas)
    dfas = (d for d in dfas if len(d.states()) == size)
    dfas = (attr.evolve(d, outputs={True, False}) for d in dfas)
    yield from fn.distinct(dfas)


def enumerate_and_save_dfas(n_tokens: int, n_epochs: int):
    def handler(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, handler)
    root = Path(f"dfas_{n_tokens}_inputs")
    root.mkdir(exist_ok=True)
    for size in tqdm(range(n_epochs)):
        path = root / f"{size}.txt"
        if path.exists():
            continue
        gen_dfas = enumerate_dfas_given_size(n_tokens, size)
        with path.open("w") as f:
            for _ in tqdm(range(1000), leave=False):
                try:
                    signal.alarm(60 * 10)
                    d = next(gen_dfas, None)
                    signal.alarm(0)
                except TimeoutError:
                    break
                if d is None:
                    break
                f.write(f"{d.to_int()}\n")

            sort_by_size(path)


def sort_by_size(path: Path):
    # Opens text file
    with path.open("r") as f:
        to_sort = []
        for line in f.readlines():
            size = len(bin(int(line)))
            to_sort.append((size, line))
    to_sort.sort()
    with path.open("w") as f:
        for _, line in to_sort:
            f.write(line)


def unroll(d: DFA, k: int):
    """Transforms DFA into a sequential reach-avoid like problem."""
    def transition(s, c):
        if s[0] >= k: return s
        return (s[0] + 1, d._transition(s[1], c))

    return DFA(start=(1, d.start),
               inputs=d.inputs,
               outputs=d.outputs,
               label=lambda s: (s[0] == k) and d._label(s[1]),
               transition=transition).minimize()


def sample_dfas(n_tokens, *,
                unroll_prob=1/5,
                avg_unroll_length=3,
                avg_mutations=3,
                tempature=500):

    path_root = Path(f"dfas_{n_tokens}_inputs")
    if not path_root.exists():
         raise NotImplementedError  # TODO: Implement dynamic DFA sampling

    encodings, weights = [], []
    for path in path_root.glob("*.txt"):
        with path.open('r') as f:
            xs = [int(x) for x in f.readlines()]
            encodings.extend(xs)
            weights.extend([len(bin(x)) - 2 for x in xs])
    weights = softmax(-np.array(weights) / tempature)

    reach_avoids = reach_avoid_sampler(n_tokens)

    alphabet = list(range(n_tokens))
    while True:
        if random.random() <= 1/2:
            candidate = next(reach_avoids)
        else:
            encoding = random.choices(encodings, weights=weights)[0]
            # Permute the alphabet.
            # Note: Helps paper over asymmetries in enumeration.
            random.shuffle(alphabet)
            candidate = DFA.from_int(encoding, inputs=alphabet)

        n_mutations = np.random.geometric(1/(avg_mutations + 1)) - 1
        mutation = candidate
        for _ in range(n_mutations):
            mutation = sample_mutation(mutation)
        mutation = mutation.minimize()
        if len(mutation.states()) > 1:
            candidate = mutation

        yield candidate


def print_without_stutter(d: DFA):
    mapping, start = dfa.dfa2dict(d)
    for state1, (_, transitions) in mapping.items():
        for token in d.inputs:
            if state1 == transitions[token]:
                del transitions[token]
    pprint(mapping)


if __name__ == "__main__":
    #enumerate_and_save_dfas(n_tokens=12, n_epochs=100)
    print_without_stutter(next(sample_dfas(12)))
    #print_without_stutter(next(reach_avoid_sampler(12)))

    print("test")

