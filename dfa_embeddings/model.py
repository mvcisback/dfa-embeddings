import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from gymnasium.spaces import Box, Discrete
import torch_ac
from dfa_embeddings.GATv2_dfa import GATv2_dfa
from dfa_embeddings.GATv2_cdfa import GATv2_cdfa
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
    def __init__(self, input_dim, output_dim, batch_size, is_compositional, alphabet_type='deterministic'):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.text_embedding_size = 32
        if is_compositional:
            self.gnn = GATv2_cdfa(input_dim, self.text_embedding_size).to(self.device)
        else:
            self.gnn = GATv2_dfa(input_dim, self.text_embedding_size, batch_size).to(self.device)

        # Resize image embedding
        self.embedding_size = self.text_embedding_size

        # Define actor's model
        self.actor = PolicyNetwork(self.embedding_size, output_dim, alphabet_type=alphabet_type)

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

    def load_pretrained_gnn(self, model_state, freeze=True):
        new_model_state = model_state.copy()

        # We delete all keys relating to the actor/critic.
        for key in model_state.keys():
            if key.find("actor") != -1 or key.find("critic") != -1:
                del new_model_state[key]

        self.load_state_dict(new_model_state, strict=False)

        if freeze:
            for param in self.gnn.parameters():
                param.requires_grad = False

