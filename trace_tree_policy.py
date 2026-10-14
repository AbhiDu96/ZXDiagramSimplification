"""Deterministic end-to-end trace harness for TreePolicy.

Runs the agent-guided tree search (the same loop as zx_compiler.optimizing)
on fixed seeded circuits and logs every decision to a JSON file, so that
refactors of TreePolicy can be verified for exact output parity:

    python trace_tree_policy.py --out traces/baseline.json
    # ... make changes to TreePolicy.py ...
    python trace_tree_policy.py --out traces/refactored.json
    diff traces/baseline.json traces/refactored.json   # must be identical

Timing is printed to stdout (and NOT stored in the JSON, which must stay
bit-identical across runs).

NOTE: this harness intentionally avoids ray and bqskit so it runs with the
base environment only. The env mirrors zx_compiler.make_env (4 rules; the
checkpoint was trained with 5 actions — DummyActionModel adapts to the mask
width, so this is safe, just as in production).
"""

import argparse
import json
import logging
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

import utils
from models import BundleNet, load_agent_checkpoint
from TreePolicy import start_tree, synchronous_forward
from zx_env import ZXEnv, extract_circuit, random_circuit

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = (
    "runs/QACI_graph_optim__20MIO_32_envs_changed_rules_optimized_5qubit"
    "_multirange_4_depth_16__1__1726148732/model-8192000.pth"
)

# fixed circuit distribution for the trace (does not need to match training;
# it only needs to be deterministic)
CIRCUIT_KWARGS = dict(
    n_qubit=5,
    num_gates=50,
    p_two_qubit=0.25,
    p_H=0.25,
    p_z=0.125,
    p_x=0.375,
    clifford_plus_T=True,
)


def build_env():
    # mirrors zx_compiler.make_env("cpu") without importing bqskit
    env = ZXEnv(
        mutate_probability=0,
        mutation_steps=0,
        reward_fn="normalized_cnot_count_reward",
        n_qubits=20,
        depth=500,
        mq_ratio=0.6,
        t_ratio=0.2,
        h_ratio=0.2,
        max_steps=4096,
        negative_reward_mean=0,
        negative_reward_std=0,
        rules_list=["bialgebra", "spider_fusion", "euler", "pi_copy"],
        full_fuse_every_step=False,
        reduce_at_reset=False,
    )
    return utils.GraphMaskWrapper(env, "cpu")


def build_agent():
    # architecture matching the 5-rule "changed_rules" checkpoint
    return BundleNet(16, 5, 256, 4, 8, "cpu", "GAT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Path to write the trace JSON")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--multi-range", type=int, default=4)
    ap.add_argument("--early-exit", action="store_true",
                    help="Reproduce the pre-fix zx_compiler.optimizing behavior "
                         "(return early on term/trunc); production now ignores "
                         "termination signals")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    agent = load_agent_checkpoint(MODEL_PATH, build_agent=build_agent)

    result = {"seeds": []}
    t_select = t_expand = 0.0
    t_start = time.perf_counter()

    trees = []
    for i in range(args.seeds):
        np.random.seed(1000 + i)
        torch.manual_seed(1000 + i)
        circuit = random_circuit(**CIRCUIT_KWARGS)
        env = build_env()

        # time select/expand separately by re-inlining the loop timing
        torch.manual_seed(2000 + i)
        np.random.seed(2000 + i)
        random.seed(2000 + i)
        start = circuit.copy().to_graph()
        obs, info = env.reset(initial_circuit_graph=start)
        tree = start_tree(
            obs.Graph, obs.state_zx_graph, info=info, multi_range=args.multi_range
        )
        events = []
        for step in range(args.steps):
            t0 = time.perf_counter()
            with torch.no_grad():
                effect, total_log, total_entropy, value = tree.select(agent)
            t1 = time.perf_counter()
            new_tree, best_reward, term, trunc, info = tree.expand(effect, env)
            t2 = time.perf_counter()
            t_select += t1 - t0
            t_expand += t2 - t1
            events.append(
                {
                    "step": step,
                    "selected_node": effect.node_idx,
                    "actions": [int(a) for a in effect.actions],
                    "logit": float(total_log),
                    "entropy": float(total_entropy),
                    "value": float(value),
                    "best_reward": float(best_reward),
                    "term": bool(term),
                    "trunc": bool(trunc),
                    "tree_size": len(new_tree.nodes),
                    "new_rewards": [float(r) for r in new_tree.rewards[-args.multi_range:]],
                }
            )
            tree = new_tree
            if args.early_exit and (term or trunc):
                events.append({"step": step, "early_exit": True})
                break
        trees.append(tree)

        result["seeds"].append(
            {
                "seed": i,
                "initial_twoqubit": circuit.twoqubitcount(),
                "final_best_twoqubit": extract_circuit(tree.get_best_node())[0].twoqubitcount(),
                "events": events,
            }
        )

    # also exercise the batched training forward path
    cfg = SimpleNamespace(device="cpu")
    with torch.no_grad():
        outputs, logits, entropies, tree_vals = synchronous_forward(
            cfg, trees, agent, selected=[[0] for _ in trees]
        )
    result["synchronous_forward"] = {
        "logits": [float(x) for x in logits],
        "entropies": [float(x) for x in entropies],
        "tree_vals": [float(x) for x in tree_vals],
    }

    total = time.perf_counter() - t_start
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
        f.write("\n")

    print(f"total={total:.3f}s  select={t_select:.3f}s  expand={t_expand:.3f}s")
    print(f"trace written to {args.out}")


if __name__ == "__main__":
    main()
