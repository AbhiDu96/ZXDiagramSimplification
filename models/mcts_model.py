import torch
from torch import nn
from torch_geometric.data import Batch
from .graph_cross_attention import GraphCrossAttention
from torch_geometric import nn as gnn
from .slow_norm import graphnorm


class MCTS_like_model(nn.Module):
    def __init__(self, action_dim, n_actions, hidden_dim, n_heads, n_message_passing, device, model_type):
        super().__init__()
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        meta_size = int(action_dim * self.n_actions)
        self.model_type = model_type
        if model_type == "GAT":
            self.gat = gnn.GATConv(meta_size, meta_size, heads=n_heads, concat=False)
        elif model_type == "ActionAtt":
            self.gat = GraphCrossAttention(n_actions, action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(action_dim),
        )
        self.weight_projection = nn.Sequential(
            nn.Linear(action_dim, 256), nn.LeakyReLU(), nn.Linear(256, 1, bias=False)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(action_dim, 256), nn.LeakyReLU(), nn.Linear(256, 1, bias=False)
        )
        self.action_embedder = nn.Embedding(
            self.n_actions + 1, embedding_dim=action_dim, padding_idx=0
        )
        self.n_message_passing = n_message_passing
        self.device = device

    def forward(self, next_obs):
        data = Batch.from_data_list([o.Graph for o in next_obs]).to(self.device)
        action_mask = torch.cat([o.action_mask for o in next_obs]).to(self.device)
        nodes, n_actions = action_mask.shape
        indices = (
            torch.arange(1, action_mask.shape[-1] + 1, device=self.device)[None, :]
            * action_mask
        )
        embs = self.action_embedder(indices).reshape(nodes, -1)
        embs = self.message_passing_loop(embs, action_mask, data).reshape(
            nodes, self.n_actions, -1
        )
        ws = self.weight_projection(embs).squeeze(-1)
        vs = self.value_projection(embs).squeeze(-1)
        for i in range(max(data.batch)):
            mx = vs[data.batch == i].max()
            vs[data.batch == i] = mx
        weights = data.clone()
        weights.x = ws
        weights.mask = action_mask
        values = data.clone()
        values.x = vs
        values.mask = action_mask
        return weights, values

    def action_masking(self, embs, action_mask):
        N = embs.shape[0]
        embs = embs.reshape(N, self.n_actions, -1)
        embs = embs * action_mask[..., None]
        embs = embs.reshape(N, -1)
        return embs

    def message_passing_loop(self, embs, action_mask, data):
        N = embs.shape[0]
        for i in range(self.n_message_passing):
            embs = graphnorm(embs, data.batch)
            embs_old = embs
            if self.model_type == "GAT":
                embs = self.gat(embs, data.edge_index)
            elif self.model_type == "ActionAtt":
                embs = embs.reshape(N, self.n_actions, -1)
                embs = self.gat(embs, data.edge_index, action_mask).reshape(N, -1)
            embs = (
                self.mlp(embs.reshape(N, self.n_actions, -1)).reshape(N, -1) + embs_old
            )
        return embs

    def to(self, device):
        self.device = device
        super().to(device)
