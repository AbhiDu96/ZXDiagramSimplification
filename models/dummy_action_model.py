import torch
from torch import nn


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
