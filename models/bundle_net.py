from torch import nn
from .tree_net import TreeNet
from .dummy_action_model import DummyActionModel


class BundleNet(nn.Module):
    def __init__(self, action_dim, n_actions, hidden_dim, n_heads, n_message_passing, device, model_type):
        super().__init__()
        self.nodenet = DummyActionModel(
            action_dim, n_actions, hidden_dim, n_heads, n_message_passing, device, model_type
        )
        self.treenet = TreeNet(
            action_dim, n_actions, hidden_dim, n_heads, n_message_passing, device, model_type
        )
