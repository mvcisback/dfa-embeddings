import signal
from pathlib import Path

import attr
import dfa
import funcy as fn
import numpy as np
from dfa_identify import find_dfas
from tqdm import tqdm


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


def unroll(d, k):
    def transition(s, c):
        if s[0] >= k: return s
        return (s[0] + 1, d._transition(s[1], c))

    return dfa.DFA(start=(1, d.start),
                   inputs=d.inputs,
                   outputs=d.outputs,
                   label=lambda s: (s[0] == k) and d._label(s),
                   transition=transition)


def sample_dfas(n_tokens, *,
                unroll_prob=1/5,
                avg_unroll_length=3):
    while True:
        # TODO: Sample size and open up corresponding text file.
        # TODO: Sample dfa encoding weighted by size.
        encoding = 0  # TODO

        # Permute the alphabet.
        # Note: Helps paper over asymmetries in enumeration.
        alphabet = list(range(n_tokens))
        random.shuffle(alphabet)
        candidate = dfa.DFA.from_int(encoding, inputs=alphabet)

        # With some probability, create a tree structure to
        # emphasize that DFAs locally look like reach avoid.
        if random.random() <= unroll_prob:
            k = np.random.geometric(1 / avg_unroll_length)
            yield unroll(candidate, k)
        else:
            yield candidate


if __name__ == "__main__":
    enumerate_and_save_dfas(n_tokens=12, n_epochs=100)

