from collections import deque
from typing import Literal

import einops
import equinox as eqx
import funcy as fn
import jax
import jax.numpy as jnp
import optax
import networkx as nx
import numpy as np
from dfa import DFA
from dfa_sampler import gen_mutated_sequential_reach_avoid as rad_dfas
from jaxtyping import Bool, Float, UInt64, Array, PRNGKeyArray

from dfa_embeddings.model import dfa2mat, DFAEncoder


type LossTerm = Literal["dist", "psat"]


class Codec(eqx.Module):
    encoder: DFAEncoder
    dist_decoder: eqx.nn.MLP
    psat_decoder: eqx.nn.MLP

    def __init__(self, n_tokens: int, dim: int, *, key: PRNGKeyArray):
        key1, key2, key3 = jax.random.split(key, 3)
        self.encoder = DFAEncoder(n_tokens=n_tokens,
                                  dim=dim,
                                  key=key1)
        self.dist_decoder = eqx.nn.MLP(dim, 1, 5, 3, key=key2)
        self.psat_decoder = eqx.nn.MLP(dim, 1, 5, 3, key=key3)

    def predict(self,
                adj: Bool[Array, "n n"],
                nodes: UInt64[Array, "n"],
                *, key: PRNGKeyArray) -> dict[LossTerm, Float[Array, "n"]]:
        n_iters = 6 # nodes.shape[0]
        x = self.encoder.encode_packed_dfa(adj, nodes, n_iters, key=key)

        out = jax.vmap(self.dist_decoder)(x)
        out = einops.rearrange(out, "n 1 -> n")
        dists = 2 * jax.vmap(jax.nn.sigmoid)(out) - 1

        out = jax.vmap(self.psat_decoder)(x)
        out = einops.rearrange(out, "n 1 -> n")
        psats = jax.vmap(jax.nn.sigmoid)(out)

        return {'dist': dists, 'psat': psats}

    def eval(self, adj, nodes, dists, psats, *, key):
        pred = self.predict(adj, nodes, key=key)

        l_dist = ((pred['dist'] - dists)**2 * (dists > 0)).mean()
        l_psat = ((pred['psat'] - psats)**2).mean()

        return (l_dist + l_psat) / 2

    # TODO: rename.
    @eqx.filter_jit
    def gen_sample(self, adj, nodes):
        pred = self.predict(adj, nodes, key=jax.random.PRNGKey(0))
        dists = pred['dist']

        # Flip transitions and remove self loops.
        adj = adj.T ^ jnp.eye(*adj.shape, dtype=jnp.bool_)

        done = state = 0
        for i in range(adj.shape[0]):
            # Count how many steps were non-accepting.
            done |= nodes[state] & 1
            idx = jnp.argmax(dists * adj[state])
            state = jnp.argmax(adj[idx])
        return done


@eqx.filter_jit
def eval_model(model, adj, nodes, dists, psats, *, key):
    n_iters = 6 #nodes.shape[0]
    x = encoder.encode_packed_dfa(adj, nodes, n_iters, key=key)
    out = jax.vmap(dist_decoder)(x)
    out = einops.rearrange(out, "n 1 -> n")
    out = 2 * jax.vmap(jax.nn.sigmoid)(out) - 1
    l_dist = ((out - dists)**2 * (dists > 0)).mean()
    out = jax.vmap(psat_decoder)(x)
    out = einops.rearrange(out, "n 1 -> n")
    out = jax.vmap(jax.nn.sigmoid)(out)
    l_psat = ((out - psats)**2).mean()
    return (l_dist + l_psat) / 2


# TODO: Rename since this isn't a distance!
def distances_from_accepting(adj, nodes):
    adj, nodes = map(np.asarray, (adj, nodes))
    g = nx.DiGraph(adj)
    # TODO: assert that accepting is a sink and only a single state.
    accept_node = np.argwhere(nodes == 1.0)[0][0]
    x = nx.single_source_shortest_path_length(g, accept_node)
    dists = jnp.array([x.get(i, 2. * nodes.shape[0]) for i in range(adj.shape[0])])
    dists = 1 - dists / nodes.shape[0]
    return dists


def prob_reach_accepting(adj, nodes):
    H = int(((nodes >> 1) != 0).sum())
    row_sums = einops.reduce(adj.T, "row col -> row", "sum")
    accept_idx = np.argwhere(nodes == 1.0)[0][0]
    P_1 = adj.T / row_sums[:, None]
    P_H = jnp.linalg.matrix_power(P_1, 2 * H)
    return P_H[:, accept_idx]


def gen_problems(encoder, n_tokens):
    for dfa in rad_dfas(n_tokens=n_tokens):
        adj, nodes = encoder.pack(dfa)
        dists = distances_from_accepting(adj, nodes)
        psats = prob_reach_accepting(adj, nodes)
        yield dfa, adj, nodes, dists, psats


def pad(target, problem):
    _, adj, nodes, dists, psats = problem
    diff = target - nodes.shape[0]
    assert diff >= 0
    adj = jnp.pad(adj, (0, diff))
    adj += jnp.eye(*adj.shape, dtype=jnp.bool_)
    nodes = jnp.pad(nodes, (0, diff))

    dists = jnp.pad(dists, (0, diff), constant_values=-1.)
    psats = jnp.pad(psats, (0, diff), constant_values=-1.)

    return adj, nodes, dists, psats


def pad_batch(batch, target=15):
    n = max(x[1].shape[0] for x in batch)
    n = max(target, n)
    adj_list, nodes_list, dists_list, psats_list = [], [], [], []
    for problem in batch:
        adj, nodes, dists, psats = pad(n, problem)
        adj_list.append(adj)
        nodes_list.append(nodes)
        dists_list.append(dists)
        psats_list.append(psats)
    return (jnp.stack(adj_list), jnp.stack(nodes_list),
            jnp.stack(dists_list), jnp.stack(psats_list))


def train(n_tokens=16, dim: int | None = 128, seed=0, n_batches=3, n_per_batch=100):
    # Hyperparameters
    LEARNING_RATE = 1e-3

    key = jax.random.PRNGKey(seed)

    model_key, call_key = jax.random.split(key, 2)

    model = Codec(n_tokens=n_tokens, dim=dim, key=model_key)

    optim = optax.adamw(LEARNING_RATE)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    problems = gen_problems(model.encoder, n_tokens=n_tokens)

    print("generating test problems")
    test = fn.take(n_per_batch, problems)
    test = pad_batch(test)

    print(f"generating {n_batches} batches")
    data = []
    for i in range(n_batches):
        print(f"-- {i}")
        batch = fn.take(n_per_batch, problems)
        batch = pad_batch(batch)
        data.append(batch)

    data = fn.cycle(data) 

    def eval_batch(model, batch):
        keys = jax.random.split(key, batch[0].shape[0])
        keys = jnp.stack(keys)
        losses = jax.vmap(model.eval)(*batch, key=keys)
        return losses.mean()

    @eqx.filter_jit
    def make_step(
        model,
        opt_state,
        batch
    ):
        loss_value, grads = eqx.filter_value_and_grad(eval_batch)(model, batch)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        test_loss = eval_batch(model, test)
        psat = jax.vmap(model.gen_sample)(*test[:2]).mean()
        return model, opt_state, {
            'training': loss_value,
            'test': test_loss,
            'psat': psat
        }

    epoch = 0
    best = -float('inf')

    eval_model = eqx.filter_jit(Codec.eval)

    for epoch, batch in enumerate(data):
        print(f'====== epoch {epoch} ======')

        model, opt_state, info = make_step(model, opt_state, batch)
        print(info)
        if info['psat'] > best:
            best = info['psat']
            print('new best!')
            eqx.tree_serialise_leaves(f"checkpoints/checkpoint_{epoch}.eqx", model)


if __name__ == '__main__':
    train()
