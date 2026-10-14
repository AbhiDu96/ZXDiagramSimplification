import torch
from torch.nn import functional as F
import hydra
from omegaconf import DictConfig, OmegaConf, read_write
import numpy as np
from zx_env import ZXEnv
from collections import deque
import torch.multiprocessing
from tqdm import trange, tqdm
import os

torch.multiprocessing.set_sharing_strategy("file_system")
import contextlib
import pickle


def init_pool_processes():
    np.random.seed()
    torch.random.seed()
def dummy(cfg):
    with contextlib.redirect_stdout(None):
        o=gather_data(None,cfg,32)
    return o

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def save_dataset(cfg, iters=10):
    os.makedirs("dataset", exist_ok=True)
    items = [int(file.split("_")[-1].split(".")[0]) for file in os.listdir("dataset")]
    pool=torch.multiprocessing.Pool(4,initializer=init_pool_processes)
    results = pool.imap_unordered(dummy, [cfg for _ in range(iters)])
    for i,obs in tqdm(enumerate(results,start = max(items, default = -1) + 1)):
        with open(f"dataset/item_{i}.pkl","wb+") as writer:
            pickle.dump(obs, writer)


def gather_data(start_obs, cfg,depth=128):

    environment = ZXEnv(
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
    obs_list = []
    obs, _ = environment.reset(initial_circuit_graph=start_obs)
    [obs, action_masks, zxgraph, _, _, _] = obs
    action_masks = action_masks[:, 1:]
    obs_list.append(obs)
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
        obs_list.append(obs)
    return obs_list


if __name__ == "__main__":
    save_dataset()
