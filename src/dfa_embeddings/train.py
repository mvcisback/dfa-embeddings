from jaxtyping import Bool, Int, Int64, Array, PRNGKeyArray
from dfa_sampler import gen_mutated_sequential_reach_avoid as gen_dfas

from dfa_embeddings.model import dfa2mat


def train():
    dfas = gen_dfas()

    x = dfa2mat(next(dfas))
    breakpoint()
    pass


if __name__ == '__main__':
    train()
