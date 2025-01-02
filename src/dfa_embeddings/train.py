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
from jaxtyping import Bool, Int, Int64, Array, PRNGKeyArray

from dfa_embeddings.model import dfa2mat, DFAEncoder


oo = float('inf')


@eqx.filter_jit
def eval_model(model, adj, nodes, dists, psats, *, key):
    encoder, dist_decoder, psat_decoder = model
    n_iters = 2 * nodes.shape[0]
    x = encoder.encode_packed_dfa(adj, nodes, n_iters, key=key)
    out = jax.vmap(dist_decoder)(x)
    l_dist = ((out - dists)**2 * (dists < 0)).mean()
    out = jax.vmap(psat_decoder)(x)
    l_psat = ((out - psats)**2).mean()
    return l_dist + l_psat


def distances_from_accepting(adj, nodes):
    adj, nodes = map(np.asarray, (adj, nodes))
    g = nx.DiGraph(adj)
    # TODO: assert that accepting is a sink and only a single state.
    src = np.argwhere(nodes == 1.0)[0][0]
    x = nx.single_source_shortest_path_length(g, src)
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
        yield adj, nodes, dists, psats


def train(n_tokens=10, dim: int | None = 8, seed=0):
    # Hyperparameters
    LEARNING_RATE = 1e-3

    key = jax.random.PRNGKey(seed)

    init_key, call_key, decoder_key = jax.random.split(key, 3)

    encoder = DFAEncoder(n_tokens=n_tokens,
                         dim=dim,
                         key=init_key)
    dist_decoder = eqx.nn.MLP(dim, 1, 5, 3, key=decoder_key)
    psat_decoder = eqx.nn.MLP(dim, 1, 5, 3, key=decoder_key)
    model = (encoder, dist_decoder, psat_decoder)

    optim = optax.adamw(LEARNING_RATE)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    problems = gen_problems(encoder, n_tokens=n_tokens)
    test = fn.take(100, problems)

    @eqx.filter_jit
    def make_step(
        model,
        opt_state,
        adj,
        nodes,
        dists,
        psats,
        *,
        key
    ):
        loss_value, grads = eqx.filter_value_and_grad(eval_model)(model, adj, nodes, dists, psats, key=key)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    epoch = 0
    best = float('inf')
    while True:
        epoch += 1
        print(f'====== epoch {epoch} ======')
        batch = fn.take(10, problems)
        for i in range(1_000):
            loss = 0.0
            print(f"--- {i}")
            for adj, nodes, dists, psats in batch:
                model, opt_state, train_loss = make_step(model, opt_state, adj, nodes, dists, psats, key=call_key)
                loss += train_loss
            print(f'training: {loss / len(batch)}')

            loss = 0.0
            for adj, nodes, dists, psats in test:
                loss += eval_model(model, adj, nodes, dists=dists, psats=psats, key=call_key)
            print(f'test: {loss / len(test)}')
            if loss < best:
                best = loss
                print('new best!')
                eqx.tree_serialise_leaves(f"checkpoints/checkpoint_{epoch}_{i}.eqx", model)


    #x = dfa2mat(next(dfas))
    pass


if __name__ == '__main__':
    train()
