import torch
from torch_geometric.data import Data
import networkx as nx
import torch_geometric
import gymnasium as gym
from typing import Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import torch_geometric.transforms as T
from torch.multiprocessing import Pool
import ray


@ray.remote
class EnvActor:
    def __init__(self,env,cfg):
        self.env=env
        self.cfg=cfg
        self.n_steps = 0
        self.rewards = 0
    def step(self,ac,pos):
        o, r, t1, t2, i = self.env.step(ac, position=pos)
        self.rewards+=r
        self.n_steps+=1
        n_steps = self.n_steps
        rewards = self.rewards
        if t1 or t2:
            self.n_steps = 0
            self.rewards = 0
            o,_ = self.env.reset()
        return o, r, t1, t2, i, n_steps, rewards
    def reset(self,*args, **kwargs):
        print("ALL RESETTING\n\n")
        return self.env.reset(*args, **kwargs)
    def get_env(self,):
        return self.env
    
class ParallelVecEnv(gym.Env):
    def __init__(self, envs, cfg):
        self.envs = envs
        self.cfg = cfg
        self.pool = [EnvActor.remote(e,cfg) for e in self.envs]  

    def step(self, actions: [(int, int)]):
        actions = np.array(actions).astype(np.int32)
        #res = self.pool.starmap(_vec_step, zip(self.envs,actions[:,1],actions[:,0]))
        res = ray.get([e.step.remote(a,p) for e,a,p in zip(self.pool,actions[:,1],actions[:,0])])
        res = map(list, zip(*res))
        next_obs, reward, terminations, truncations, inf, n_steps, rewards = res
        reward = np.array(reward)
        infos = defaultdict(list)
        for idx,(t1,t2) in enumerate(zip(terminations, truncations)):
            if t1 or t2:
                infos["final_info"].append(
                    {"episode": {"r": rewards[idx], "l": n_steps[idx]},
                     "env":inf[idx]}
                )
        
        return next_obs, reward, terminations, truncations, infos

    def reset(
        self,
    ):
        print("ALL RESETTING\n\n")
        return zip(*ray.get([e.reset.remote() for e in self.pool]))

    
    def __getitem__(self,idx):
        return ray.get(self.pool[idx].get_env.remote())
