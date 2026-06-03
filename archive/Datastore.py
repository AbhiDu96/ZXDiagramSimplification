import torch
from dataclasses import dataclass
from typing import Any, List


@dataclass
class GraphData:
    @property
    def n_nodes(self) -> int:
        return

    @property
    def n_actions(self) -> int:
        return

    actions: torch.Tensor
    nodes: torch.Tensor


def largest_graph(obses: GraphData):
    B = len(obses)
    max_obses = 0
    max_actions = 0
    max_t = 0
    for t in range(len(obses)):
        max_t = max(max_t, len(obses[t]))
        for b in range(len(obses[t])):
            max_obses = max(max_obses, obses[t][b].n_nodes)
            max_actions = max(max_actions, obses[t][b].n_actions)
    # now fill the data
    obs = torch.ones(B, max_t, max_obses, max_actions)
    for t in range(len(obses)):
        for b in range(len(obses[t])):
            nnodes = obses[t][b].n_nodes
            nactions = obses[t][b].n_actions
            obs[b, t, :nnodes, :nactions] = obses[t][b].actions
    return obses, max_obses, max_actions


class Datastore:
    def __init__(self):
        self.obs_store = []
        self.act_store = []
        self.logprob_store = []
        self.reward_store = []
        self.dones_store = []
        self.value_store = []
        pass

    def append(self, obs, act, logprobs, reward, dones, values):
        self.obs_store.append(obs)
        self.act_store.append(act)
        self.logprob_store.append(logprobs)
        self.reward_store.append(reward)
        self.dones_store.append(dones)
        self.value_store.append(values)

    def get_padded(self):
        # the datastores are (timesteps, (batch, data))
        obses, max_obses, max_actions = largest_graph(self.obs_store)
        actions = torch.LongTensor(self.act_store)
        logprobs = torch.tensor(self.logprob_store)
        rewards = torch.tensor(self.reward_store)
        dones = torch.tensor(self.dones_store)
        values = torch.tensor(self.value_store)
        return (obses, actions, logprobs, rewards, dones, values)
