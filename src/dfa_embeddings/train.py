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
def eval_model(model, adj, nodes, dists, *, key):
    encoder, decoder = model
    n_iters = 2 * nodes.shape[0]
    x = encoder.encode_packed_dfa(adj, nodes, n_iters, key=key)
    out = jax.vmap(decoder)(x)
    return ((out - dists)**2).mean()


def distances_from_accepting(adj, nodes, n):
    adj, nodes = map(np.asarray, (adj, nodes))
    g = nx.DiGraph(adj)
    # TODO: assert that accepting is a sink and only a single state.
    src = np.argwhere(nodes == 1.0)[0][0]
    x = nx.single_source_shortest_path_length(g, src)
    dists = jnp.array([x.get(i, 2. * nodes.shape[0]) for i in range(adj.shape[0])])
    dists = 1 - dists / nodes.shape[0]
    return dists



def train(n_tokens=10, dim: int | None = 8, seed=0):
    # Hyperparameters
    LEARNING_RATE = 1e-3

    key = jax.random.PRNGKey(seed)
    dfas = rad_dfas(n_tokens=n_tokens)

    init_key, call_key, decoder_key = jax.random.split(key, 3)

    encoder = DFAEncoder(n_tokens=n_tokens,
                         dim=dim,
                         key=init_key)
    decoder = eqx.nn.MLP(dim, 1, 5, 3, key=decoder_key)
    model = (encoder, decoder)

    optim = optax.adamw(LEARNING_RATE)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    test = fn.take(100, dfas)
    dfa = test[0]

    @eqx.filter_jit
    def make_step(
        model,
        opt_state,
        adj,
        nodes,
        dists,
        *,
        key
    ):
        loss_value, grads = eqx.filter_value_and_grad(eval_model)(model, adj, nodes, dists, key=key)
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
        batch = fn.take(1000, dfas)
        for i in range(10):
            loss = 0.0
            print(f"--- {i}")
            for dfa in batch:
                adj, nodes = encoder.pack(dfa)
                dists = distances_from_accepting(adj, nodes, n=len(dfa.states()))

                model, opt_state, train_loss = make_step(model, opt_state, adj, nodes, dists, key=call_key)
                loss += train_loss
            print(f'training: {loss / 1000}')

            loss = 0.0
            for dfa in test:
                adj, nodes = encoder.pack(dfa)
                dists = distances_from_accepting(adj, nodes, n=len(dfa.states()))
                loss += eval_model(model, adj, nodes, dists=dists, key=call_key)
            print(f'test: {loss / len(test)}')
            if loss < best:
                best = loss
                print('new best!')
                eqx.tree_serialise_leaves(f"checkpoints/checkpoint_{epoch}_{i}.eqx", model)


    #x = dfa2mat(next(dfas))
    pass


if __name__ == '__main__':
    train()
