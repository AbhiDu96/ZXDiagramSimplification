"""ZX-based bqskit compiler passes and agent-guided circuit optimization.

Merged from the former `bqskit_pass.py`, `harris_bench.py` and the compiler
workflow helpers in `bench_compilers.py`. Provides:

- bqskit passes: `ConvertZX` (RL agent), `FullReduceZX` (PyZX full_reduce),
  `KorbinianMethodZX` (optional, needs pyzx_korbinian), `PrintCNOTsPass`,
  `OutputGraphImage`
- `optimizing()`: agent-guided tree search over a ZX graph
- `run_zx_workflow()`: shared partition -> ZX pass -> unfold compile loop
- `just_transpile()`: qiskit optimization_level=3 baseline
- convenience wrappers `our_opt` / `korbinian_opt` / `full_reduce_opt`
"""

import importlib
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pyzx as zx
import qiskit.qasm2
import torch
from bqskit.compiler import Compiler, Workflow
from bqskit.compiler.basepass import BasePass
from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit
from bqskit.ir.gates import CNOTGate, CZGate
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.passes import (
    ForEachBlockPass,
    QuickPartitioner,
    ScanningGateRemovalPass,
    UnfoldPass,
)
from qiskit.compiler import transpile

import utils
from models import BundleNet, load_agent_checkpoint
from TreePolicy import start_tree
from zx_env import ZXEnv, extract_circuit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    korbinianbench = importlib.import_module(
        "pyzx_korbinian.pyzx-heuristics.korbinianbench"
    )
except Exception:
    korbinianbench = None

DEFAULT_MODEL_PATH = (
    "runs/QACI_graph_optim__20MIO_32_envs_changed_rules_optimized_5qubit"
    "_multirange_4_depth_16__1__1726148732/saves/model-819200.pth"
)


def make_env(device):
    # NOTE: rules_list has 4 entries here, but the default checkpoint was
    # trained with 5 actions (see validate.py / the run folder name). The
    # search still runs, but one learned action is never available. Align
    # rules_list with your checkpoint for meaningful results.
    n_qubits = 20
    n_depth = 500
    mq_ratio = 0.6
    h_ratio = 0.2
    t_ratio = 0.2
    env = ZXEnv(
        mutate_probability=0,
        mutation_steps=0,
        reward_fn="normalized_cnot_count_reward",
        n_qubits=n_qubits,
        depth=n_depth,
        mq_ratio=mq_ratio,
        t_ratio=t_ratio,
        h_ratio=h_ratio,
        max_steps=4096,
        negative_reward_mean=0,
        negative_reward_std=0,
        rules_list=["bialgebra", "spider_fusion", "euler", "pi_copy"],
        full_fuse_every_step=False,
        reduce_at_reset=False,
    )

    env = utils.GraphMaskWrapper(env, device)
    return env


def _default_build_agent():
    # Architecture matching the 5-rule "changed_rules" checkpoint.
    return BundleNet(16, 5, 256, 4, 8, "cpu", "GAT")


_AGENT_CACHE = {}


def load_agent(model_path=None, build_agent=None):
    """Load (and cache) the agent: whole pickled model if the file is one,
    otherwise state_dict rebuilt via ``build_agent``."""
    model_path = model_path or os.environ.get("ZX_MODEL_PATH", DEFAULT_MODEL_PATH)
    if model_path not in _AGENT_CACHE:
        logging.info("loading model from %s", model_path)
        _AGENT_CACHE[model_path] = load_agent_checkpoint(
            model_path, build_agent=build_agent or _default_build_agent
        )
    return _AGENT_CACHE[model_path]


def optimizing(state, searchdepth=128, model_path=None, agent=None):
    """Run agent-guided tree search on the ZX graph of `state` (a zx.Circuit).

    Returns the best ZX graph node found.
    """
    agent = agent or load_agent(model_path)
    env = make_env("cpu")
    start = state.copy().to_graph()
    next_obs = None
    for it in range(1):
        next_obs, info = env.reset(initial_circuit_graph=start)
        next_obs = start_tree(
            next_obs.Graph, next_obs.state_zx_graph, info=info, multi_range=16
        )
        max_rew = 0
        for step in range(searchdepth):
            with torch.no_grad():
                effect, total_log, _, value = next_obs.select(agent)

            new_tree, rew, term, trunc, info = next_obs.expand(effect, env)
            # TreePolicy deliberately ignores termination signals: tree search
            # can backtrack out of dead ends, so keep expanding until the
            # search-depth budget is exhausted.
            max_rew = max(rew, max_rew)
            next_obs = new_tree
        tmp = next_obs.get_best_node()
        if max_rew == 0.0:
            break
        start = extract_circuit(tmp.copy())[0].to_graph()

    return next_obs.get_best_node()


class PrintCNOTsPass(BasePass):
    async def run(self, circuit, data) -> None:
        logging.info(f"BQSKit step, current CNOT count:  {circuit.count(CNOTGate())}")


def cnot_count_scorer(ops):
    score = 0.0
    for op in ops:
        if op.gate == CNOTGate():
            score += 10
        else:
            score += 1
    return score


class OutputGraphImage(BasePass):

    async def run(self, circuit, data) -> None:
        qs = bqskit_to_qiskit(circuit)
        pyzx = zx.Circuit.from_qasm(qiskit.qasm2.dumps(qs))
        fig = zx.draw(pyzx)
        fig.savefig(f"out-{time.time()}.png")
        plt.close(fig)


class ConvertZX(BasePass):
    def __init__(self, searchdepth, model_path=None) -> None:
        super().__init__()
        self.searchdepth = searchdepth
        self.model_path = model_path

    async def run(self, circuit, data) -> None:
        if circuit.count(CNOTGate()) <= 2:
            return
        qs = bqskit_to_qiskit(circuit)
        pyzx = zx.Circuit.from_qasm(qiskit.qasm2.dumps(qs))
        try:
            out = optimizing(pyzx.copy(), self.searchdepth, model_path=self.model_path)
        except Exception:
            logging.exception("agent optimization failed; leaving block unchanged")
            return
        pyzx, _ = extract_circuit(out)
        circ = OPENQASM2Language().decode(pyzx.to_qasm())
        circuit.become(circ)


class FullReduceZX(BasePass):

    async def run(self, circuit, data) -> None:
        if circuit.count(CNOTGate()) <= 2:
            return
        qs = bqskit_to_qiskit(circuit)
        pyzx = zx.Circuit.from_qasm(qiskit.qasm2.dumps(qs))
        g = pyzx.to_graph()
        zx.full_reduce(g)
        pyzx = zx.extract_circuit(g)
        circ = OPENQASM2Language().decode(pyzx.to_qasm())
        circuit.become(circ)


class KorbinianMethodZX(BasePass):

    async def run(self, circuit, data) -> None:
        if korbinianbench is None:
            raise RuntimeError(
                "pyzx_korbinian is not installed; KorbinianMethodZX is unavailable"
            )
        if circuit.count(CNOTGate()) <= 2:
            return
        qs = bqskit_to_qiskit(circuit)
        pyzx = zx.Circuit.from_qasm(qiskit.qasm2.dumps(qs))
        pyzx, _ = korbinianbench.korbinian(pyzx)
        circ = OPENQASM2Language().decode(pyzx.to_qasm())
        circuit.become(circ)


def CZ_to_CX(circ):
    graph = zx.Circuit(circ.qubits)

    for g in circ.gates:
        if isinstance(g, zx.gates.CZ) and g.name == "CZ":
            tar = g.target
            con = g.control
            graph.add_gate("HAD", con)
            graph.add_gate("CNOT", tar, con)
            graph.add_gate("HAD", con)
        else:
            graph.add_gate(g)
    return graph


def run_zx_workflow(circuit, zx_pass, rounds=3):
    """Compile `circuit` (a zx.Circuit) through the shared bqskit workflow:

    QuickPartitioner(5) -> gate removal -> `zx_pass` -> gate removal -> unfold,
    repeated `rounds` times with a CZ->CX conversion between rounds.

    Returns (zx.Circuit, two_qubit_count).
    """
    bqCirc = OPENQASM2Language().decode(circuit.to_qasm())
    workflow = Workflow(
        [
            QuickPartitioner(5),
            ForEachBlockPass(ScanningGateRemovalPass()),
            ForEachBlockPass(zx_pass),
            ForEachBlockPass(ScanningGateRemovalPass()),
            UnfoldPass(),
        ]
    )
    opt_circuit = bqCirc.copy()
    cnotcount = []
    for _ in range(rounds):
        with Compiler() as compiler:
            opt_circuit = compiler.compile(opt_circuit, workflow=workflow)
        qasm = OPENQASM2Language().encode(opt_circuit)
        opt_circuit = CZ_to_CX(zx.Circuit.from_qasm(qasm))
        opt_circuit = OPENQASM2Language().decode(opt_circuit.to_qasm())
        cnotcount.append(
            opt_circuit.count(CNOTGate()) + opt_circuit.count(CZGate())
        )
    logging.info("POST OPT %s", cnotcount)
    circ = zx.Circuit.from_qasm(OPENQASM2Language().encode(opt_circuit))
    return circ, circ.twoqubitcount()


def our_opt(circuit, searchdepth=128, rounds=3):
    """Agent-guided optimization via the ConvertZX bqskit pass."""
    return run_zx_workflow(circuit, ConvertZX(searchdepth), rounds=rounds)


def korbinian_opt(circuit, rounds=3):
    """Korbinian's PyZX heuristics via the KorbinianMethodZX bqskit pass."""
    return run_zx_workflow(circuit, KorbinianMethodZX(), rounds=rounds)


def full_reduce_opt(circuit, rounds=3):
    """PyZX full_reduce via the FullReduceZX bqskit pass."""
    return run_zx_workflow(circuit, FullReduceZX(), rounds=rounds)


def just_transpile(circuit, *_args):
    """qiskit optimization_level=3 baseline. Extra args are ignored so this can
    share the (circuit, args) calling convention of the other passes."""
    bqCirc = OPENQASM2Language().decode(circuit.to_qasm())
    transpiled = transpile(
        bqskit_to_qiskit(bqCirc),
        optimization_level=3,
        basis_gates=["h", "cx", "rx", "rz"],
    )
    transpiled = qiskit_to_bqskit(transpiled)
    qasm = OPENQASM2Language().encode(transpiled)
    circ = zx.Circuit.from_qasm(qasm)
    return circ, circ.twoqubitcount()


if __name__ == "__main__":
    # Single-circuit demo (formerly harris_bench.py):
    #   python zx_compiler.py --model-path path/to/checkpoint [--searchdepth 128] [--print-result]
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize a single random Clifford+T circuit with the trained ZX agent."
    )
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint (pickled model or state_dict)")
    parser.add_argument("--searchdepth", type=int, default=128,
                        help="Tree-search steps per optimization run")
    parser.add_argument("--print-result", action="store_true",
                        help="Print the optimized state")
    args = parser.parse_args()

    from zx_env import random_circuit

    state = random_circuit(clifford_plus_T=True)
    optimized_state = optimizing(state, searchdepth=args.searchdepth, model_path=args.model_path)
    if args.print_result:
        print(optimized_state)
