import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray
from gatv2_eqx import GATv2


class DFAEncoder(eqx.Module):
    gnn: GATv2
    node_encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear
    node_tags: eqx.nn.Embedding

    def __init__(self, key: PRNGKeyArray):
        ...
