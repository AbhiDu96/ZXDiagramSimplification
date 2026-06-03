import torch
from torch import nn
from torch_geometric import nn as gnn


class ActionEmbeddingModel(nn.Module):
    def __init__(self, action_dim, n_actions, hidden_dim, n_heads, n_message_passing, device, model_type):
        super().__init__()
        self.projection = nn.Linear(4, hidden_dim)
        self.transf = nn.ModuleList([
            gnn.conv.TransformerConv(hidden_dim, hidden_dim, n_heads)
            for _ in range(n_message_passing)
        ])
        self.ff = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim)
            )
            for _ in range(n_message_passing)
        ])
        self.value_proj = nn.Linear(hidden_dim, 1)
        self.action_proj = nn.Linear(hidden_dim, n_actions)
        self.pos_proj = nn.Linear(hidden_dim, 1)
        self.n_message_passing = n_message_passing

    def forward(self, data):
        nodes, n_actions = data.action_mask.shape
        p = self.projection(data.x)
        for i in range(self.n_message_passing):
            p = self.transf[i](p, data.edge_index)
            p = self.ff[i](p)
        pos_unmasked = self.pos_proj(p)
        values = self.pos_proj(p)
        action_unmasked = self.action_proj(p)
        return pos_unmasked, action_unmasked, values
