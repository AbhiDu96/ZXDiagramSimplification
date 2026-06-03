import torch
import torch_geometric.data as data
import networkx as nx
import torch_geometric
import gymnasium as gym
from typing import Any, SupportsFloat
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import torch_geometric.transforms as T
from torch.multiprocessing import Pool

@dataclass
class GraphMask:
    Graph: data.Data
    action_mask: torch.Tensor
    state_zx_graph: Any


def mask_to_index(mask):
    indices = mask.nonzero()
    return indices


def expand_graph(graph: data.Data) -> data.Data:
    x = graph.x.clone()
    nnodes = x.shape[0]
    edge_features = graph.edge_attr.clone()
    edge_index = graph.edge_index.clone()
    nedges = edge_features.shape[0]

    # first padd both to the same size
    merged = torch.cat([x, edge_features[:, None].expand(nedges, x.shape[-1])], 0)
    # now do the rewiring: we need exactly 2*nedges
    edges = torch.zeros(2, 2 * nedges, dtype=int)
    # from node
    edges[0, :nedges] = edge_index[0]
    # to virtual edges (which are shifted by nnodes)
    edges[1, :nedges] = torch.arange(nedges) + nnodes
    # back from virtual edge
    # to real target node
    edges[0, nedges:] = torch.arange(nedges) + nnodes
    edges[1, nedges:] = edge_index[1]

    return data.Data(x=merged, edge_index=edges)


class GraphMakeDirected(gym.ObservationWrapper):
    def __init__(self,env):
        self.env = env
    def observation(self, observation: Any) -> Any:
        [state, action_masks, statezx, _, _, _] = observation
        n_raw_edges = state.edge_index.shape[1]//2
        state = data.Data(x=state.x,edge_index=state.edge_index[:,:n_raw_edges],edge_attr=state.edge_attr[:n_raw_edges])
        action_masks = action_masks[:,:len(state.x)+n_raw_edges+1]
        return [state, action_masks, statezx, None, None, None]


    def step(self, action,position,**kwargs):
        o, r, t1, t2, i = self.env.step(action, position,**kwargs)
        return self.observation(o), r, t1, t2, i
    def reset(self,*args,**kwargs):
        o,info=self.env.reset(*args,**kwargs)
        return self.observation(o),info


class GraphMaskWrapper(gym.ObservationWrapper):
    def __init__(self, env, device):
        self.env = env
        self.device=device

    def observation(self, observation: Any) -> GraphMask:
        [
            state,
            action_masks,
            state_zx_graph,
            node_masks,
            edge_masks,
            rule_mask,
        ] = observation
        # only take the actual actions
        # also transpose since I want (nnodes, n_actions)
        action_mask = torch.from_numpy(action_masks[:, 1:]).long().T
        # print("action_mask\n",action_mask[:8])
        # now expand the graph with virtual nodes
        # print("state in",state)
        state = expand_graph(state)
        state = T.ToUndirected()(state)
        state.action_mask=action_mask
        # print("state out",state,)
        return GraphMask(state.to(self.device), action_mask.to(self.device), state_zx_graph.clone())

    def step(self, action, position, **kwargs):
        o, r, t1, t2, i = self.env.step(action, position,**kwargs)
        return self.observation(o), r, t1, t2, i
    
    def reset(self,*args,**kwargs):
        o,info=self.env.reset(*args,**kwargs)
        return self.observation(o),info



class RewardTransform(gym.RewardWrapper):
    def __init__(self,env,weight=1,frequency=10000):
        self.frequency = frequency
        self.idx=0
        self.doneflage=False
        self.env=env
        self.weight=weight
    def reward(self,reward):
        return reward*self.weight if self.idx % self.frequency ==0 or self.doneflage else 0
    def step(self,*args,**kwargs):
        self.idx +=1
        o, r, t1, t2, i = self.env.step(*args,**kwargs)
        self.doneflage = t1 or t2
        r = self.reward(r)
        return o, r, t1, t2, i
    def reset(self, *args, **kwargs):
        self.idx=0
        self.doneflage=False
        return self.env.reset(*args,**kwargs)



class SingleVecEnv(gym.Env):
    def __init__(self, envs, cfg):
        self.envs = envs
        self.rewards = np.zeros(len(envs))
        self.n_steps = np.zeros(len(envs))
        self.cfg = cfg

    def step(self, actions: [(int, int)]):
        next_obs, reward, terminations, truncations, infos = (
            [],
            [],
            [],
            [],
            [],
        )
        infos = defaultdict(list)
        for idx, (e, a) in enumerate(zip(self.envs, actions)):
            pos, ac = a.astype(np.int32)
            o, r, t1, t2, i = e.step(ac, position=pos)
            self.rewards[idx] += r
            self.n_steps[idx] += 1
            reward.append(r)
            terminations.append(t1)
            truncations.append(t2)
            if t1 or t2:
                infos["final_info"].append(
                    {"episode": {"r": self.rewards[idx], "l": self.n_steps[idx]}}
                )
                self.rewards[idx] = 0
                self.n_steps[idx] = 0
                infos["env"] = i
                o,_ = e.reset()
                next_obs.append(o)
            else:
                next_obs.append(o)
        return next_obs, reward, terminations, truncations, infos

    def reset(
        self,initial_circuit_graph=None
    ):
        if initial_circuit_graph is None:
            return zip(*map(lambda x: x.reset(), self.envs))
        else:
            return zip(*map(lambda x: x[0].reset(initial_circuit_graph=x[1]), zip(self.envs,initial_circuit_graph)))
    def __getitem__(self,idx):
        return self.envs[idx]


if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    edge_features = torch.tensor([10, 20, 30, 40])
    d = data.Data(x, edge_index=edge_index, edge_attr=edge_features)
    old = torch_geometric.utils.to_networkx(d)
    nx.draw()
    print("d", d)
    exp = expand_graph(d)
    print("exp", exp)
