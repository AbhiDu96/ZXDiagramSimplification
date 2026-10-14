import torch
from torch import nn

from .slow_norm import SlowNorm


class BundleNet(nn.Module):
    def __init__(self, action_dim, n_actions, hidden_dim, n_heads, n_message_passing, device, model_type):
        super().__init__()
        self.nodenet = DummyActionModel(
            action_dim, n_actions, hidden_dim, n_heads, n_message_passing, device, model_type
        )
        self.treenet = TreeNet(
            action_dim, n_actions, hidden_dim, n_heads, n_message_passing, device, model_type
        )
        
        
class DummyActionModel(nn.Module):
    def __init__(self, action_dim, n_actions, hidden_dim, n_heads, n_message_passing, device, model_type):
        super().__init__()
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        meta_size = int(action_dim * self.n_actions)
        self.meta_size = meta_size
        self.model_type = model_type
        self.n_message_passing = n_message_passing
        self.device = device

    def forward(self, data):
        nodes, n_actions = data.action_mask.shape
        pos_unmasked = torch.ones(nodes, 1, device=self.device)
        action_unmasked = torch.ones(nodes, n_actions, device=self.device)
        values = torch.zeros(data.action_mask.shape, device=self.device)
        return pos_unmasked, action_unmasked, values
    
    
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
