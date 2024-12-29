import itertools

import numpy as np
import jax.numpy as jnp
from jaxtyping import Bool, Int, Int64, Array, PRNGKeyArray
from dfa import DFA
from dfa_sampler import gen_mutated_sequential_reach_avoid as gen_dfas


def dfa2graph(dfa: DFA) -> tuple[Int64[Array, "n n"],
                                 Bool[Array, "n"]]:
    """Translates DFA into an adjacency matrix and accept/reject state vec.

    The adjacency matrix entries are bit flags where the ith bit
    being on means token i triggers a transition.

    The adjacency matrix also has edges reversed compared to DFA and
    should be read as A[s',s] = x means the tokens encoded in x
    transition s to s'.
    """
    assert len(dfa.inputs) >> 8 == 0, "Currently, support at most 2^8 inputs."""

    states = list(dfa.states())
    tokens = sorted(dfa.inputs)
    transitions = itertools.product(states, tokens)

    n = len(states)
    adj = np.zeros((n, n), dtype=np.int64)
    accepting = np.zeros(n, dtype=np.bool_)

    state2idx = {s: i for i, s in enumerate(states)}
    token2idx = {t: i for i, t in enumerate(tokens)}
    for s in states:
        accepting[state2idx[s]] = dfa._label(s)

    for s1, t in itertools.product(states, tokens):
        s2 = dfa._transition(s1, t)
        i1, i2 = map(state2idx.get, (s1, s2))
        adj[i2, i1] |= 1 << token2idx[t]

    return jnp.array(adj), jnp.array(accepting)



def train():
    dfas = gen_dfas()

    dfa2graph(next(dfas))
    pass


if __name__ == '__main__':
    train()
