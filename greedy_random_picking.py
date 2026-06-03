import logging
import torch
import numpy as np
import os
import pickle
import ray
import hydra
import matplotlib.pyplot as plt
import pyzx as zx
from copy import deepcopy
from TreePolicy import Tree, start_tree, extract_optimal_path
from ppo import make_env
from models import BundleNet
from zx_env import extract_circuit
from benchmark_utils import generate_N_colors, extract_level, Dataset, deploy_agents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class GreedyTree(Tree):
    def __init__(self, prev_Tree=None, multi_range=1):
        super().__init__(prev_Tree, multi_range=multi_range)

    def select_node(self, treenet=None, selected=None, device="cpu", cache=None):
        selected = np.argmax([inf["feats"][3] for inf in self.infos])
        return selected, 0, 0, 0

    def select(self, bundlenet, selected=None, device="cpu", cache=None):
        if selected is None:
            selected = [None, None]
        sel_act, logit_tree, entropy_tree, value_tree = self.select_node(
            bundlenet.treenet, selected[0], device=device, cache=cache
        )
        selected[0] = sel_act
        action, logit_act, entropy_act, value_act = self.select_expansion(
            bundlenet.nodenet, selected[0], selected[1], device=device, cache=cache
        )
        effect = [torch.tensor(sel_act, device=device), action]
        return effect, logit_tree, entropy_tree, value_act + value_tree


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def plot(cfg):
    t = cfg.exp_name
    logging.info("results dir: validation_results/results_%s", t)
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
    n_qubit = 5
    num_gates = 20
    res_levels = [[], [], [], []]
    res2 = []
    res3 = []
    res4 = []
    brute = []
    data_path = os.environ.get('data_path', None)
    ds = Dataset(cfg, 1_000, data_path)
    for i in range(len(ds)):
        c = ds[i]
        initial_count = c.stats_dict()["twoqubit"]
        for j in range(1, 5):
            c_new1 = extract_level(deepcopy(c.to_graph()), relax_level=j, up_to_perm=True)
            res_levels[j - 1].append(c_new1.stats_dict()["twoqubit"])
        g = deepcopy(c.to_graph())
        zx.full_reduce(g)
        c_new2 = zx.extract_circuit(g, up_to_perm=True)
        res2.append(c_new2.stats_dict()["twoqubit"])
        g = deepcopy(c.to_graph())
        zx.full_reduce(g)
        c_new3 = zx.extract_circuit(g, up_to_perm=False)
        res3.append(c_new3.stats_dict()["twoqubit"])
        res4.append(deploy_agents.remote(cfg, deepcopy(c), agent))
        if cfg.validation.n_qubit <= 4:
            from brute_force_CX_opt import optimize_CX_circuit
            c_brute = optimize_CX_circuit(c.copy())
            brute.append(c_brute.stats_dict()["twoqubit"])
    nodes = [ray.get(r) for r in res4]
    out = [extract_circuit(n)[0].stats_dict()["twoqubit"] for n in nodes]
    out_no_swaps = [
        extract_circuit(n, up_to_perm=True)[0].stats_dict()["twoqubit"] for n in nodes
    ]
    labels = [
        "level1",
        "level2",
        "level3",
        "level4",
        "f_r - dont count SWAPS",
        "f_r - count SWAPS as CX",
        "greedy CNOT",
        "greedy CNOT - don't count SWAPS",
    ]
    reses = res_levels + [res2, res3, out, out_no_swaps]
    if cfg.validation.n_qubit <= 4:
        labels.append("brute force optimum")
        reses.append(brute)
    colors = generate_N_colors(len(reses))
    fig, axes = plt.subplots(int(np.ceil(len(reses) / 2)), 2, figsize=(15, 20))
    logging.debug("axes=%s n_reses=%d n_labels=%d", axes, len(reses), len(labels))
    mn = np.min(reses) - 1
    mx = np.max(reses) + 1
    bins = np.linspace(mn, mx)
    out_dict = dict()
    for idx, a in enumerate([k for ks in axes for k in ks]):
        if idx >= len(reses):
            break
        a.set_xlim(left=mn, right=mx)
        a.hist(reses[idx], color=colors[idx], label=labels[idx], bins=bins)
        a.set_title(labels[idx])
        logging.info("adding %s: %s", labels[idx], reses[idx])
        out_dict[labels[idx]] = reses[idx]
        a.legend()
    out_dict["agent_levels"] = [extract_circuit(n)[1] for n in nodes]
    plt.tight_layout()
    t = cfg.exp_name
    if not os.path.exists("validation_results/"):
        os.mkdir("validation_results")
    if not os.path.exists(f"validation_results/results_{t}"):
        os.mkdir(f"validation_results/results_{t}")
    plt.savefig(f"validation_results/results_{t}/hist-{t}.png")
    with open(f"validation_results/results_{t}/results.pkl", "wb+") as writer:
        pickle.dump(out_dict, writer)


if __name__ == "__main__":
    plot()
