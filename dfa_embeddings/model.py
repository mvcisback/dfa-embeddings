"""
This is the description of the deep NN currently being used.
It is a small CNN for the features with an GRU encoding of the LTL task.
The features and LTL are preprocessed by utils.format.get_obss_preprocessor(...) function:
    - In that function, I transformed the LTL tuple representation into a text representation:
    - Input:  ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
    - output: ['until', 'not', 'a', 'and', 'b', 'until', 'not', 'c', 'd']
Each of those tokens get a one-hot embedding representation by the utils.format.Vocabulary class.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import torch_ac

from gymnasium.spaces import Box, Discrete


from dfa_embeddings.GATv2 import GATv2


from dfa_embeddings.policy_network import PolicyNetwork

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(nn.Module, torch_ac.ACModel):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.text_embedding_size = 32
        self.gnn = GATv2(input_dim, self.text_embedding_size).to(self.device)

        # Resize image embedding
        self.embedding_size = self.text_embedding_size

        # Define actor's model
        self.actor = PolicyNetwork(self.embedding_size, output_dim)

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):

        embedding = self.gnn(obs)

        # Actor
        dist = self.actor(embedding)

        # Critic
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

    def load_pretrained_gnn(self, model_state):
        new_model_state = model_state.copy()

        # We delete all keys relating to the actor/critic.
        for key in model_state.keys():
            if key.find("actor") != -1 or key.find("critic") != -1:
                del new_model_state[key]

        self.load_state_dict(new_model_state, strict=False)

        for param in self.gnn.parameters():
            param.requires_grad = False
