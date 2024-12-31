import equinox as eqx
import jax
from jaxtyping import Bool, Int, Int64, Array, PRNGKeyArray
from dfa_sampler import gen_mutated_sequential_reach_avoid as rad_dfas

from dfa_embeddings.model import dfa2mat, DFAEncoder


def eval_model(model, adj, nodes, *, key):
    encoder, decoder = model
    n_iters = nodes.shape[0]
    x = encoder.encode_packed_dfa(adj, nodes, n_iters, key=key)
    out = jax.vmap(decoder)(x)
    return out.sum()  # TODO


def train(n_tokens=10, dim: int | None = 8, seed=0):
    key = jax.random.PRNGKey(seed)
    dfas = rad_dfas(n_tokens=n_tokens)

    init_key, call_key, decoder_key = jax.random.split(key, 3)

    encoder = DFAEncoder(n_tokens=n_tokens,
                         dim=dim,
                         key=init_key)
    decoder = eqx.nn.Linear(dim, 1, key=decoder_key)
    model = (encoder, decoder)

    dfa = next(dfas)

    adj, nodes = encoder.pack(dfa)

    eval_model(model, adj, nodes, key=call_key)
    
    f = eqx.filter_jit(eval_model)
    g = eqx.filter_grad(eval_model)

    x = g(model, adj, nodes, key=call_key)
    print(x)
    #x = dfa2mat(next(dfas))
    pass


if __name__ == '__main__':
    train()
