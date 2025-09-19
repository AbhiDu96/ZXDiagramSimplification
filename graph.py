import pyzx as zx
import numpy as np
from zx_env import extract_circuit
from zx_env import random_circuit
from zx_env import zx_env
import matplotlib.pyplot as plt
from copy import deepcopy
from ppo import make_env
from TreePolicy import start_tree
import torch
import ray
import hydra
from model import BundleNet
import os
from brute_force_CX_opt import optimize_CX_circuit
import time
import pickle
import copy
from pyzx.simplify import to_graph_like
from zx_env.circuit_utils.circuit_extractor import interior_clifford_simp_on_wire
import time

def generate_N_colors(N):
    colormap = plt.cm.get_cmap("tab10", N)  # You can choose any colormap here
    colors = [colormap(i) for i in range(N)]
    return colors


def random_CX_circuit(n_qubit=5, num_gates=40):
    c = zx.Circuit(qubit_amount=n_qubit)
    for i in range(num_gates):
        target = np.random.choice(range(n_qubit))
        while True:
            control = np.random.choice(range(n_qubit))
            if not control == target:
                break
        c.add_gate("CNOT", control, target)
    return c


class Dataset:
    def __init__(self, cfg, size, path):
        print("="*70)
        print("PATH",path)
        print("="*70)
        if path is None:
            self.dataset = [
                random_circuit(
                    n_qubit=cfg.validation.n_qubit,
                    num_gates=cfg.validation.n_gate,
                    p_two_qubit=cfg.validation.p_two_qubit,
                    p_H=cfg.validation.p_H,
                    p_z=cfg.validation.p_z,
                    p_x=cfg.validation.p_x,
                    many_pi_gates=cfg.validation.many_pi_gates,
                    clifford_plus_T=cfg.validation.clifford_plus_T,
                )
                for _ in range(size)
            ]
        else:
            self.dataset=[]
            with open(path,"rb") as reader:
                random_circuits = pickle.load(reader)
            for i, circ in enumerate(random_circuits):
                g_circ = circ.to_qasm()
                self.dataset.append(zx.Circuit.from_qasm(g_circ))
        pass
    def __getitem__(self,idx):
        return self.dataset[idx]
    def __len__(self):
        return len(self.dataset)

@ray.remote
def deploy_agents(cfg, state, agent):
    env = make_env(cfg, "cpu")
    start = state.to_graph()
    next_obs = None
    for it in range(cfg.validation.search_loops):
        t0=time.time()
        print("LOOP ROUND", it)
        next_obs, info = env.reset(initital_circuit_graph=start)
        next_obs = start_tree(next_obs.Graph, next_obs.state_zx_graph, info=info)
        for step in range(0, cfg.algorithm.num_steps):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                # make a temporary batch from our data:
                effect, total_log, _, value = next_obs.select(agent)

            # TRY NOT TO MODIFY: execute the game and log data.
            new_tree, rew, term, trunc, info = next_obs.expand(effect, env)
            next_done = np.logical_or(term, trunc)
            next_obs, next_done = (
                new_tree,
                torch.Tensor(next_done),
            )
        print(
            "=====" * 5 + "\n",
            f"result of iteration {it}",
            extract_circuit(next_obs.get_best_node())[0].twoqubitcount(),
            "time", time.time()-t0,
            "\n" + "=====" * 5,
        )
        start = extract_circuit(next_obs.get_best_node().clone())[0].to_graph()

    return (
        next_obs.get_best_node()
    )  # extract_circuit(next_obs.get_best_node())[0].twoqubitcount()


def extract_level(graph, relax_level, up_to_perm=False):
    g = copy.deepcopy(graph)
    g.normalize()
    to_graph_like(g)
    g.normalize()
    interior_clifford_simp_on_wire(g, relax_level, quiet=True, stats=None)
    if relax_level == 5:
        zx.full_reduce(g)
        return zx.extract_circuit(g, up_to_perm=up_to_perm)
    try:
        return zx.extract_circuit(copy.deepcopy(g), up_to_perm=up_to_perm)
    except:
        return extract_level(graph, relax_level + 1, up_to_perm=up_to_perm)


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def plot(cfg):
    path = cfg.validation.model_path
    print("model_path is", cfg.validation.model_path)
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
    #print("before", agent.nodenet.value_projection[0].weight)
    agent.load_state_dict(weights)
    #print("after", agent.nodenet.value_projection[0].weight)
    n_qubit = 5
    num_gates = 20
    res_levels = [[], [], [], []]
    res2 = []
    res3 = []
    res4 = []
    brute = []
    data_path = os.environ.get('data_path',None)
    ds = Dataset(cfg,1_000,data_path)
    for i in range(len(ds)):
        # c=random_circuit(n_qubit=5, num_gates=80, p_two_qubit=0.25, p_H=0.25, p_z=0.25, p_x=0.25, many_pi_gates=False,clifford_plus_T=True)
        c=ds[i]
        """c = random_circuit(
            n_qubit=cfg.validation.n_qubit,
            num_gates=cfg.validation.n_gate,
            p_two_qubit=cfg.validation.p_two_qubit,
            p_H=cfg.validation.p_H,
            p_z=cfg.validation.p_z,
            p_x=cfg.validation.p_x,
            many_pi_gates=cfg.validation.many_pi_gates,
            clifford_plus_T=cfg.validation.clifford_plus_T,
        )"""
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
        "agent optimized",
        "agent optimized - don't count SWAPS",
    ]
    reses = res_levels + [res2, res3, out, out_no_swaps]
    if cfg.validation.n_qubit <= 4:
        # colors.append("black")
        labels.append("brute force optimum")
        reses.append(brute)
    colors = generate_N_colors(len(reses))
    fig, axes = plt.subplots(int(np.ceil(len(reses) / 2)), 2, figsize=(15, 20))
    print(axes, len(reses), len(labels))
    mn = np.min(reses) - 1
    mx = np.max(reses) + 1
    bins = np.linspace(
        mn,
        mx,
    )
    out_dict = dict()
    for idx, a in enumerate([k for ks in axes for k in ks]):
        if idx >= len(reses):
            break
        a.set_xlim(left=mn, right=mx)
        a.hist(reses[idx], color=colors[idx], label=labels[idx], bins=bins)
        a.set_title(labels[idx])
        print(idx, "adding in", labels[idx], "with", reses[idx])
        out_dict[labels[idx]] = reses[idx]
        # a.hist(res2, color='red', alpha=0.5, label='f_r - dont count SWAPS')
        # a.hist(res3, color='green', alpha=0.5, label='f_r - count SWAPS as CX')
        # a.hist(out, color='black', alpha=0.5, label='agent optimized')
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
