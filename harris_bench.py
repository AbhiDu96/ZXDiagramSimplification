import os
import argparse
import logging
import pickle
import torch
import numpy as np
import utils
from zx_env import extract_circuit
from TreePolicy import Tree, start_tree
from bqskit_pass import make_env
from zx_env import random_circuit
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit
import pyzx as zx
from qiskit.compiler import transpile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def optimizing(state, model_path=None):
    if model_path is None:
        model_path = os.environ.get("ZX_MODEL_PATH", "model.pkl")
    logging.info("loading model from %s", model_path)
    with open(model_path, "rb") as reader:
        agent = pickle.load(reader)
    env = make_env("cpu")
    start = state.copy().to_graph()
    next_obs = None
    for it in range(1):
        next_obs, info = env.reset(initial_circuit_graph=start)
        next_obs = start_tree(next_obs.Graph, next_obs.state_zx_graph, info=info, multi_range=16)
        max_rew = 0
        for step in range(128):
            with torch.no_grad():
                effect, total_log, _, value = next_obs.select(agent)

            new_tree, rew, term, trunc, info = next_obs.expand(effect, env)
            if term or trunc:
                logging.info("exiting due to termination signal")
                return next_obs.get_best_node()
            max_rew = max(rew, max_rew)
            next_done = np.logical_or(term, trunc)
            next_obs, next_done = (
                new_tree,
                torch.Tensor(next_done),
            )
        tmp = next_obs.get_best_node()
        '''print(
            "=====" * 5 + "\n",
            f"result of iteration {it}",
            extract_circuit(next_obs.get_best_node())[0].twoqubitcount(),
            "time", time.time()-t0, "max Reward", max_rew,
            "\n" + "=====" * 5,
            "equal??",
            check_equality(state.copy().to_graph(),tmp.copy()),
            tmp
        )'''
        if max_rew == 0.0:
            break
        start = extract_circuit(tmp.copy())[0].to_graph()

    return (
        next_obs.get_best_node()
    ) 


def just_transpile(circuit):
    bqCirc = OPENQASM2Language().decode(circuit.to_qasm())
    just_transpile = transpile(
        bqskit_to_qiskit(bqCirc),
        optimization_level=3,
        basis_gates=["h", "cx", "rx", "rz"],
    )
    just_transpile = qiskit_to_bqskit(just_transpile)
    qasm = OPENQASM2Language().encode(just_transpile)
    circ = zx.Circuit.from_qasm(qasm)
    return circ, circ.twoqubitcount()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, help="Path to model.pkl")
    parser.add_argument("--print-result", action="store_true", help="Print the optimized state")
    args = parser.parse_args()

    state = random_circuit(clifford_plus_T=True)
    optimized_state = optimizing(state, model_path=args.model_path)
    if args.print_result:
        print(optimized_state)