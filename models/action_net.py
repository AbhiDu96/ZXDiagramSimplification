import torch
from torch import nn
from torch_geometric.utils import scatter


class ActionNet(nn.Module):
    def __init__(self, action_dim, n_actions, hidden_dim, n_heads, n_message_passing, device, model_type):
        super().__init__()
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        meta_size = int(action_dim * self.n_actions)
        self.meta_size = meta_size
        self.model_type = model_type
        self.feat = nn.Sequential(nn.Linear(8, self.hidden_dim), nn.LeakyReLU())
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU()
                ),
            ]
        )
        self.position_projection = nn.Linear(meta_size, 1)
        self.action_projection = nn.Linear(meta_size, self.n_actions)
        self.value_projection = nn.Sequential(
            nn.Linear(action_dim, action_dim), nn.LeakyReLU(), nn.Linear(action_dim, 1)
        )
        self.value_projection[-1].weight.data.zero_()
        self.value_projection[-1].bias.data.zero_()
        self.action_embedder = nn.Embedding(
            self.n_actions + 1, embedding_dim=action_dim, padding_idx=0
        )
        self.n_message_passing = n_message_passing
        self.device = device

    def forward(self, data):
        nodes, n_actions = data.action_mask.shape
        embs = self.feat(data.x)
        for m in self.mlp:
            embs = m(embs) + embs
        pos_unmasked = self.position_projection(embs)
        graph = scatter(embs, data.batch, reduce="mean").reshape(
            data.batch.max() + 1, -1
        )
        action_unmasked = self.action_projection(graph)
        action_unmasked = action_unmasked[data.batch]
        values = self.value_projection(
            embs.reshape(nodes, n_actions, -1)
        ).reshape(nodes, n_actions)
        return pos_unmasked, action_unmasked, values
