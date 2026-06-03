import torch
from torch import nn


def graphnorm(data, batch):
    for i in range(max(batch)):
        data[batch == i] = (data[batch == i] - data[batch == i].mean()) / (
            data[batch == i].std() + 1e-5
        )
    return data


class SlowNorm(nn.Module):
    def __init__(self, n_channels, momentum=1e-5):
        super().__init__()
        self.n_channels = n_channels
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(self.n_channels))
        self.register_buffer("std", 5 * torch.ones(self.n_channels))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                if x.shape[0] > 1:
                    self.mean = self.mean * (1 - self.momentum) + x.mean(0) * self.momentum
                    self.std = self.std * (1 - self.momentum) + x.std(0) * self.momentum
        return (x - self.mean) / (self.std + 1e-5)
