import torch
from torch import nn
from .slow_norm import SlowNorm


class TreeNet(nn.Module):
    def __init__(self, action_dim, n_actions, hidden_dim, n_heads, n_message_passing, device, model_type):
        super().__init__()
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.model_type = model_type
        self.feat = nn.Sequential(
            SlowNorm(8),
            nn.Linear(8, self.hidden_dim),
            nn.LeakyReLU()
        )
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, 2 * self.hidden_dim), nn.GLU(),
                ),
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, 2 * self.hidden_dim), nn.GLU(),
                ),
            ]
        )
        self.priority_prediction = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.value_prediction = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.action_embedder = nn.Embedding(
            self.n_actions + 1, embedding_dim=action_dim, padding_idx=0
        )
        self.n_message_passing = n_message_passing
        self.device = device

    def forward(self, data):
        embs = self.feat(data)
        for m in self.mlp:
            embs = m(embs) + embs
        node_prios = self.priority_prediction(embs).reshape(-1)
        node_values = self.value_prediction(embs).reshape(-1)
        return node_prios, node_values
