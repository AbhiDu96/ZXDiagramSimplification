import logging
import pyzx as zx
import numpy as np
import torch
import ray
import copy
import pickle
import os
import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
from copy import deepcopy
from pyzx.simplify import to_graph_like
from zx_env.circuit_utils.circuit_extractor import interior_clifford_simp_on_wire
from zx_env import extract_circuit
from zx_env import random_circuit
from ppo import make_env
from TreePolicy import start_tree


def generate_N_colors(N):
    import matplotlib.pyplot as plt
    colormap = plt.cm.get_cmap("tab10", N)
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
    except Exception:
        return extract_level(graph, relax_level + 1, up_to_perm=up_to_perm)


class Dataset:
    def __init__(self, cfg, size, path=None):
        logging.info("=" * 60)
        logging.info("DATASET path=%s size=%d", path, size)
        logging.info("=" * 60)
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
            self.dataset = []
            with open(path, "rb") as reader:
                random_circuits = pickle.load(reader)
            for circ in random_circuits:
                g_circ = circ.to_qasm()
                self.dataset.append(zx.Circuit.from_qasm(g_circ))

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def get_benchmark_circuits(path="zx_env/bench_mark_circuits/"):
    benchmark_circuit_name = glob.glob(path + "*/*", recursive=True)
    benchmark_graphs = []
    for circ in benchmark_circuit_name:
        benchmark_graphs.append(zx.Circuit.from_quipper_file(circ))
    return benchmark_circuit_name, benchmark_graphs


@ray.remote
def deploy_agents(cfg, state, agent):
    env = make_env(cfg, "cpu")
    start = state.to_graph()
    next_obs = None
    for it in range(cfg.validation.search_loops):
        logging.info("LOOP ROUND %d", it)
        next_obs, info = env.reset(initial_circuit_graph=start)
        next_obs = start_tree(next_obs.Graph, next_obs.state_zx_graph, info=info, multi_range=cfg.multi_range)
        for step in range(0, cfg.algorithm.num_steps):
            with torch.no_grad():
                effect, total_log, _, value = next_obs.select(agent)

            new_tree, rew, term, trunc, info = next_obs.expand(effect, env)
            next_done = np.logical_or(term, trunc)
            next_obs, next_done = (
                new_tree,
                torch.Tensor(next_done),
            )
        result_twoqubit = extract_circuit(next_obs.get_best_node())[0].twoqubitcount()
        logging.info("result of iteration %d: %d two-qubit gates", it, result_twoqubit)
        start = extract_circuit(next_obs.get_best_node().clone())[0].to_graph()

    return next_obs.get_best_node()
