import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn


class GraphCrossAttention(nn.Module):
    def __init__(self, n_actions, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.queries = nn.Linear(self.action_dim, 1, bias=False)
        self.keys = nn.Linear(self.action_dim, 1, bias=False)
        self.values = nn.Linear(self.action_dim, self.action_dim, bias=False)
        self.aggregator = gnn.SimpleConv()
        self.n_actions = n_actions

    def forward(self, embs, edge_index, action_mask):
        weighted = self.attend(embs, edge_index, action_mask)
        return weighted

    def attend(self, embs, edge_index, action_mask):
        N, n_act, dim = embs.shape
        k = self.keys(embs.mean(-2, keepdim=True))
        q = self.queries(embs)
        v = self.values(embs.mean(-2, keepdim=True))
        alpha = torch.exp(F.leaky_relu(k[edge_index[0]] + q[edge_index[1]]))
        weighted = torch.vmap(
            lambda a: self.aggregator(v.reshape(N, -1), edge_index, edge_weight=a),
            in_dims=1,
            out_dims=1,
        )(alpha)
        return weighted
