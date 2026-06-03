import torch
from torch import nn
from torch_geometric import nn as gnn
from torch_geometric.data import Batch
from .graph_cross_attention import GraphCrossAttention
from .categorical_masked import CategoricalMasked


class ActionModel(nn.Module):
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
        self.policy_projection = nn.Linear(action_dim, 1)
        self.value_projection = nn.Linear(action_dim, 1, bias=False)
        self.action_embedder = nn.Embedding(
            self.n_actions + 1, embedding_dim=action_dim, padding_idx=0
        )
        self.n_message_passing = n_message_passing
        self.device = device

    def forward(self, data, action_mask):
        nodes, n_actions = action_mask.shape
        indices = (
            torch.arange(1, action_mask.shape[-1] + 1, device=self.device)[None, :]
            * action_mask
        )
        embs = self.action_embedder(indices).reshape(nodes, -1)
        embs = self.message_passing_loop(embs, action_mask, data)
        embs = self.action_masking(embs, action_mask).reshape(nodes, self.n_actions, -1)
        mask = torch.where(action_mask == 0, -torch.inf, 0)
        policy = self.policy_projection(embs).squeeze(-1)
        policy = policy + mask
        value = self.value_projection(embs).squeeze(-1)
        return policy, value

    def action_masking(self, embs, action_mask):
        N = embs.shape[0]
        embs = embs.reshape(N, self.n_actions, -1)
        embs = embs * action_mask[..., None]
        embs = embs.reshape(N, -1)
        return embs

    def message_passing_loop(self, embs, action_mask, data):
        N = embs.shape[0]
        for i in range(self.n_message_passing):
            embs = self.action_masking(embs, action_mask)
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

    def get_action_and_value(self, next_obs, actions=None):
        data = Batch.from_data_list([o.Graph for o in next_obs]).to(self.device)
        action_mask = torch.cat([o.action_mask for o in next_obs]).to(self.device)
        policy, value = self(data, action_mask)
        sample_actions = actions is None
        actions = torch.zeros(len(next_obs), 2, device=self.device)
        logprob = torch.zeros(len(next_obs), device=self.device)
        entropy = torch.zeros(len(next_obs), device=self.device)
        values = torch.zeros(len(next_obs), device=self.device)
        for i in range(len(next_obs)):
            p = policy[data.batch == i]
            nnodes, nacts = p.shape
            m = action_mask[data.batch == i]
            probs = CategoricalMasked(logits=p.reshape(-1), masks=m.reshape(-1).bool(), device=self.device)
            if sample_actions:
                samp = probs.sample()
                s = torch.stack(torch.unravel_index(samp, (nnodes, nacts)))
                actions[i] = s
            tmp = actions[i][0] * nacts + actions[i][1]
            logprob[i] = probs.log_prob(tmp)
            entropy[i] = probs.entropy()
            values[i] = value[data.batch == i].mean()
        return actions, logprob, entropy, values

    def get_value(self, next_obs):
        return self.get_action_and_value(next_obs)[-1]
