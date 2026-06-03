import logging
import pyzx as zx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import ray
import hydra
from models import BundleNet
import os
import pickle
import pandas as pd
from benchmark_utils import generate_N_colors, extract_level, deploy_agents, get_benchmark_circuits
from zx_env import extract_circuit
from ppo import make_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def plot(cfg):
    path = cfg.validation.model_path
    logging.info("model_path: %s", cfg.validation.model_path)
    weights = torch.load(path)
    env = make_env(cfg, "cpu")
    agent = BundleNet(
        cfg.model.action_dim,
        env.action_space.n,
        hidden_dim=cfg.model.hidden_dim,
        n_heads=cfg.model.n_heads,
        n_message_passing=cfg.model.n_message_passing,
        device="cpu",
        model_type=cfg.model.model_type,
    )
    agent.load_state_dict(weights)
    res_levels = [[], [], [], []]
    res2 = []
    res3 = []
    res4 = []
    brute = []
    data_path = os.environ.get('data_path', None)
    names, circs = get_benchmark_circuits(data_path)
    for i in range(len(circs)):
        c = deepcopy(circs[i])
        initial_count = c.stats_dict()['twoqubit']
        for j in range(1, 5):
            c_new1 = extract_level(deepcopy(c.to_graph()), relax_level=j)
            res_levels[j - 1].append(c_new1.stats_dict()['twoqubit'])
        g = deepcopy(c.to_graph())
        zx.full_reduce(g)
        c_new2 = zx.extract_circuit(g, up_to_perm=True)
        res2.append(c_new2.stats_dict()['twoqubit'])
        g = deepcopy(c.to_graph())
        zx.full_reduce(g)
        c_new3 = zx.extract_circuit(g, up_to_perm=False)
        res3.append(c_new3.stats_dict()['twoqubit'])
        res4.append(deploy_agents.remote(cfg, deepcopy(c), agent))
        if cfg.validation.n_qubit <= 4:
            from brute_force_CX_opt import optimize_CX_circuit
            c_brute = optimize_CX_circuit(c.copy())
            brute.append(c_brute.stats_dict()['twoqubit'])
    trees = [ray.get(r) for r in res4]
    nodes = [k.get_best_node() for k in trees]
    out = [extract_circuit(n)[0].stats_dict()['twoqubit'] for n in nodes]
    out_no_swaps = [extract_circuit(n, up_to_perm=True)[0].stats_dict()['twoqubit'] for n in nodes]
    labels = ["level1", "level2", "level3", "level4", "f_r - dont count SWAPS", "f_r - count SWAPS as CX", "agent optimized", "agent optimized - don't count SWAPS"]
    reses = res_levels + [res2, res3, out, out_no_swaps]
    if cfg.validation.n_qubit <= 4:
        labels.append("brute force optimum")
        reses.append(brute)
    out_dict = dict()
    out_dict["trees"] = trees
    fig, axes = plt.subplots(int(np.ceil(len(reses) / 2)), 2, figsize=(15, 20))
    for idx, a in enumerate([k for ks in axes for k in ks]):
        if idx >= len(reses):
            break
        logging.info("adding %s: %s", labels[idx], reses[idx])
        out_dict[labels[idx]] = reses[idx]
        a.hist(res2, color='red', alpha=0.5, label='f_r - dont count SWAPS')
        a.hist(res3, color='green', alpha=0.5, label='f_r - count SWAPS as CX')
        a.hist(out, color='black', alpha=0.5, label='agent optimized')
    out_dict["agent_levels"] = [extract_circuit(n)[1] for n in nodes]
    df = pd.DataFrame.from_dict(out_dict, orient="index", columns=names)
    plt.tight_layout()
    t = cfg.exp_name
    if not os.path.exists("bench_results/"):
        os.mkdir("bench_results")
    if not os.path.exists(f"bench_results/results_{t}"):
        os.mkdir(f"bench_results/results_{t}")
    with open(f"bench_results/results_{t}/results.pkl", "wb+") as writer:
        pickle.dump(out_dict, writer)
    df.to_csv(f"bench_results/results_{t}/data.csv")
    plt.savefig(f"bench_results/results_{t}/hist-{t}.png")


if __name__ == "__main__":
    plot()
