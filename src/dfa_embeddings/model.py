import itertools
from collections import defaultdict

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from dfa import DFA
from jaxtyping import Array, PRNGKeyArray
from jaxtyping import Bool, UInt64, Float
from gatv2_eqx import GATv2


def dfa2mat(dfa: DFA):
    """Translates DFA into an adjacency matrix and feature vec.

    TODO: Document.
    """
    assert len(dfa.inputs) < 64, "Currently, support at most 63 inputs."""

    states = list(dfa.states())
    tokens = sorted(dfa.inputs)
    transitions = defaultdict(list)
    for s1, t in itertools.product(states, tokens):
        s2 = dfa._transition(s1, t)
        if s1 == s2: continue
        transitions[s1, s2].append(t)

    m = len(states)
    n = m + len(transitions)
    adj = np.eye(n, dtype=np.bool_)
    features = np.zeros(n, dtype=np.uint64)

    state2idx = {s: i for i, s in enumerate(states)}
    token2idx = {t: i for i, t in enumerate(tokens)}
    for s in states:
        features[state2idx[s]] = int(dfa._label(s))

    for i12, ((s1, s2), ts) in enumerate(transitions.items(), start=m):
        i1, i2 = map(state2idx.get, (s1, s2))

        adj[i2, i12] = adj[i12, i1] = True  # s2 -> (s1, t) -> s1.
        for t in ts:
            features[i12] |= np.uint64(1 << (token2idx[t] + 1))

    return adj, features


class DFAEncoder(eqx.Module):
    gnn: GATv2
    tags: eqx.nn.Embedding
    n_tokens: int

    def __init__(self,
                 n_tokens: int,
                 dim: int | None = None,
                 *, key: PRNGKeyArray):
        self.n_tokens = n_tokens

        if dim is None:
            dim = 1 + n_tokens

        key_gnn, key_tag = jax.random.split(key, 2)

        self.gnn = GATv2(dim, key=key_gnn)
        self.tags = eqx.nn.Embedding(num_embeddings=n_tokens+2,
                                     embedding_size=dim,
                                     key=key) 

    def pack(self, dfa: DFA) -> tuple[Bool[Array, "n n"], UInt64[Array, "n"]]:
        adj, features = dfa2mat(dfa)
        return jnp.array(adj), jnp.array(features)

    def unpack_and_tag(self, x) -> Float[Array, "d"]:
        """Unpack bits and map to sum of feature attributes (tags)."""
        indices = jnp.arange(1 + self.n_tokens)  # accepting bit + token bits.
        x = jax.vmap(lambda i: ((x >> i) & 1) * self.tags(i))(indices)
        return einops.reduce(x, "k d -> d", 'sum')

    def encode_packed_dfa(self,
                          adj: Bool[Array, "n n"],
                          nodes: UInt64[Array, "n"],
                          n_iters: int,
                          *, key: PRNGKeyArray) -> Float[Array, "n d"]:
        adj = adj.astype(jnp.float32)
        nodes = einops.rearrange(nodes, "n -> n 1")
        nodes = jax.vmap(self.unpack_and_tag)(nodes)
        return self.gnn(nodes=nodes, adj_mat=adj, n_iters=n_iters, key=key)

    def __call__(self, dfa: DFA,
                 n_iters: int | None = None,
                 *, key: PRNGKeyArray) -> Float[Array, "n d"]:
        adj, nodes = self.pack(dfa)

        if n_iters is None:
            n_iters = nodes.shape[0]

        return self.encode_packed_dfa(adj, nodes, n_iters, key=key)

