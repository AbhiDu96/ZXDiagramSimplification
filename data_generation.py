import torch
import torch_geometric.transforms
import torch_geometric.data
from torch import nn
from torch.nn import functional as F
import torch_geometric
from torch_geometric import nn as gnn
import hydra
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf, read_write
import numpy as np
from zx_env import zx_env
from pytorch_metric_learning import miners, losses, distances, regularizers
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time
import ray
import torch.multiprocessing
from tqdm import trange, tqdm
import os

torch.multiprocessing.set_sharing_strategy("file_system")
import contextlib
from torch_scatter import scatter
from zx_env import extract_circuit
import pickle


def init_pool_processes():
    np.random.seed()
    torch.random.seed()
def dummy(cfg):
    with contextlib.redirect_stdout(None):
        o=gather_data(None,cfg,32)
    return o

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def save_dataset(cfg,iters=100_000):
    items = [int(k.split("_")[-1].split(".")[0]) for k in os.listdir("dataset_similarity")]
    pool=torch.multiprocessing.Pool(64,initializer=init_pool_processes)
    results = pool.imap_unordered(dummy, [cfg for _ in range(iters)])
    for i,obs in tqdm(enumerate(results,start=max(items))):
        #t0=time.time()
        with open(f"dataset_similarity/item_{i}.pkl","wb+") as writer:
            pickle.dump(obs, writer)
        #print("time pickle",time.time()-t0)


def gather_data(start_obs, cfg,depth=128):
    # print("sample new circuit")
    environment = zx_env(
        mutate_probability=cfg.env.mutate_prob,
        mutation_steps=cfg.env.mutation_steps,
        reward_fn=cfg.env.reward_fn,
        n_qubits=cfg.env.n_qubits,
        depth=cfg.env.depth,
        mq_ratio=cfg.env.cnot_fraction,
        t_ratio=cfg.env.t_fraction,
        h_ratio=cfg.env.h_fraction,
        max_steps=cfg.env.max_env_steps,
        negative_reward_mean=cfg.env.extra_noise_mean,
        negative_reward_std=cfg.env.extra_noise_std,
        rules_list=cfg.env.rules_used,
        full_fuse_every_step=cfg.env.full_fuse_every_step,
        reduce_at_reset=cfg.env.reduce_at_reset,
    )
    obses = []
    obs, _ = environment.reset(initital_circuit_graph=start_obs)
    [obs, action_masks, zxgraph, _, _, _] = obs
    action_masks = action_masks[:, 1:]
    obses.append(obs)
    for _ in range(depth):
        choices = (
            (torch.arange(len(action_masks.reshape(-1)))[action_masks.reshape(-1) == 1])
            .long()
            .reshape(-1)
        )
        choice = torch.tensor(np.random.choice(choices.numpy())).long()
        act, pos = torch.unravel_index(choice, action_masks.shape)
        obs, _, _, _, _ = environment.step(position=pos.item(), action=act.item())
        [obs, action_masks, zxgraph, _, _, _] = obs
        action_masks = action_masks[:, 1:]
        obses.append(obs)
    return obses


if __name__ == "__main__":
    save_dataset()
