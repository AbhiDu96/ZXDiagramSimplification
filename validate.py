"""Unified validation script for trained ZX-rewrite agents.

Merges the former `graph.py` (random/generated circuits) and
`bench_circuits.py` (benchmark circuit files) into a single entry point,
selected via `validation.mode` in the config:

    uv run validate.py exp_name=my_run validation.model_path=path/to/weights.pth \
        validation.mode=dataset      # random circuits via benchmark_utils.Dataset (default)
    uv run validate.py exp_name=my_run validation.model_path=path/to/weights.pth \
        validation.mode=benchmark    # named benchmark circuits via get_benchmark_circuits

Optionally set the `data_path` environment variable to load circuits from a
pickle file (dataset mode) or a custom benchmark directory (benchmark mode).

Outputs are written to `validation_results/{mode}_{exp_name}/`:
  - hist-{exp_name}.png : grid of two-qubit-gate-count histograms per method
  - results.pkl         : dict of raw result lists per method
  - data.csv            : per-circuit table (benchmark mode only)
"""

import logging
import os
import pickle
from copy import deepcopy

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pyzx as zx
import ray

from benchmark_utils import (
    Dataset,
    deploy_agents,
    extract_level,
    generate_N_colors,
    get_benchmark_circuits,
)
from models import BundleNet, load_agent_checkpoint
from ppo import make_env
from zx_env import extract_circuit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LABELS = [
    "level1",
    "level2",
    "level3",
    "level4",
    "f_r - dont count SWAPS",
    "f_r - count SWAPS as CX",
    "agent optimized",
    "agent optimized - don't count SWAPS",
]


def load_agent(cfg):
    """Load the trained agent: whole pickled model if the file is one,
    otherwise a state_dict rebuilt from the hydra config."""
    path = cfg.validation.model_path
    logging.info("model_path: %s", path)

    def build_agent():
        env = make_env(cfg, "cpu")
        return BundleNet(
            cfg.model.action_dim,
            env.action_space.n,
            hidden_dim=cfg.model.hidden_dim,
            n_heads=cfg.model.n_heads,
            n_message_passing=cfg.model.n_message_passing,
            device="cpu",
            model_type=cfg.model.model_type,
        )

    return load_agent_checkpoint(path, build_agent=build_agent)


def load_circuits(cfg):
    """Return (names, circuits) for the configured validation mode.

    `names` is None in dataset mode (circuits are anonymous).
    """
    mode = cfg.validation.mode
    data_path = os.environ.get("data_path", None)
    if mode == "dataset":
        ds = Dataset(cfg, cfg.validation.n_circuits, data_path)
        return None, [ds[i] for i in range(len(ds))]
    if mode == "benchmark":
        # get_benchmark_circuits has no None handling; only override the default
        # path when the env var is actually set.
        if data_path is not None:
            return get_benchmark_circuits(data_path)
        return get_benchmark_circuits()
    raise ValueError(f"Unknown validation.mode: {mode!r} (expected 'dataset' or 'benchmark')")


def evaluate_baselines(cfg, circuits, agent):
    """Run all comparison methods over the circuits.

    Returns (results, nodes) where `results` maps label -> list of two-qubit
    gate counts and `nodes` is the list of final agent search nodes.
    """
    results = {label: [] for label in LABELS}
    results["initial"] = []
    agent_refs = []

    for c in circuits:
        c = deepcopy(c)
        results["initial"].append(c.stats_dict()["twoqubit"])

        for level in range(1, 5):
            c_level = extract_level(deepcopy(c.to_graph()), relax_level=level, up_to_perm=True)
            results[f"level{level}"].append(c_level.stats_dict()["twoqubit"])

        g = deepcopy(c.to_graph())
        zx.full_reduce(g)
        results["f_r - dont count SWAPS"].append(
            zx.extract_circuit(g, up_to_perm=True).stats_dict()["twoqubit"]
        )

        g = deepcopy(c.to_graph())
        zx.full_reduce(g)
        results["f_r - count SWAPS as CX"].append(
            zx.extract_circuit(g, up_to_perm=False).stats_dict()["twoqubit"]
        )

        agent_refs.append(deploy_agents.remote(cfg, deepcopy(c), agent))

        if cfg.validation.n_qubit <= 4:
            from brute_force_CX_opt import optimize_CX_circuit

            c_brute = optimize_CX_circuit(c.copy())
            results.setdefault("brute force optimum", []).append(
                c_brute.stats_dict()["twoqubit"]
            )

    # deploy_agents already returns the best node of the final search tree.
    nodes = [ray.get(r) for r in agent_refs]
    results["agent optimized"] = [
        extract_circuit(n)[0].stats_dict()["twoqubit"] for n in nodes
    ]
    results["agent optimized - don't count SWAPS"] = [
        extract_circuit(n, up_to_perm=True)[0].stats_dict()["twoqubit"] for n in nodes
    ]
    return results, nodes


def plot_histograms(results, out_path):
    """Grid of per-method histograms with a shared x-axis."""
    plot_labels = [k for k in results if k != "initial"]
    reses = [results[k] for k in plot_labels]
    colors = generate_N_colors(len(reses))

    fig, axes = plt.subplots(int(np.ceil(len(reses) / 2)), 2, figsize=(15, 20))
    mn = np.min(reses) - 1
    mx = np.max(reses) + 1
    bins = np.linspace(mn, mx)

    for idx, a in enumerate([ax for row in axes for ax in row]):
        if idx >= len(reses):
            break
        a.set_xlim(left=mn, right=mx)
        a.hist(reses[idx], color=colors[idx], label=plot_labels[idx], bins=bins)
        a.set_title(plot_labels[idx])
        a.legend()
        logging.info("plotted %s: %s", plot_labels[idx], reses[idx])

    plt.tight_layout()
    plt.savefig(out_path)


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def validate(cfg):
    logging.info("Loading agent...")
    agent = load_agent(cfg)
    logging.info("Loading circuits...")
    names, circuits = load_circuits(cfg)
    logging.info("Evaluating baselines...")
    results, nodes = evaluate_baselines(cfg, circuits, agent)
    logging.info("Extracting agent levels...")
    results["agent_levels"] = [extract_circuit(n)[1] for n in nodes]

    out_dir = os.path.join("validation_results", f"{cfg.validation.mode}_{cfg.exp_name}")
    os.makedirs(out_dir, exist_ok=True)

    plot_histograms(results, os.path.join(out_dir, f"hist-{cfg.exp_name}.png"))

    with open(os.path.join(out_dir, "results.pkl"), "wb+") as writer:
        pickle.dump(results, writer)

    if names is not None:
        # Benchmark mode: per-circuit table with circuit names as columns.
        import pandas as pd

        table = {k: v for k, v in results.items() if k != "agent_levels"}
        df = pd.DataFrame.from_dict(table, orient="index", columns=names)
        df.to_csv(os.path.join(out_dir, "data.csv"))

    logging.info("results written to %s", out_dir)


if __name__ == "__main__":
    validate()
