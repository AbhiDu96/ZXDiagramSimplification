import bqskit
from bqskit.compiler import Workflow
from bqskit.passes import (
    QuickPartitioner,
    ForEachBlockPass,
    ScanningGateRemovalPass,
    UnfoldPass,
    ClusteringPartitioner,
    ScanPartitioner,
    ExhaustiveGateRemovalPass,
)
import bqskit
from bqskit.ext import qiskit_to_bqskit
from bqskit.compiler import Compiler
from bqskit.compiler.basepass import BasePass
from bqskit.ir.gates import CircuitGate, CNOTGate, CZGate
from bqskit_pass import (
    PrintCNOTsPass,
    ConvertZX,
    cnot_count_scorer,
    OutputGraphImage,
    FullReduceZX,
)
import logging
import pickle
import numpy as np
import torch
from model import BundleNet
from bqskit.ir.lang.qasm2 import OPENQASM2Language
import os
from bqskit.ext import bqskit_to_qiskit
from bqskit.ir.gates import CZGate
import time
from qiskit.compiler import transpile
from zx_env.circuit_utils.circuit_generator import random_circuit
import pyzx as zx
import os
import pickle
import numpy as np
import ray
import random
import importlib

try:
    korbinianbench = importlib.import_module("pyzx_korbinian.pyzx-heuristics.korbinianbench")
    from bqskit_pass import KorbinianMethodZX
except:
    pass

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from qiskit_ibm_transpiler.ai.synthesis import (
    AILinearFunctionSynthesis,
    AIPermutationSynthesis,
    AICliffordSynthesis,
)
from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit_ibm_transpiler.ai.synthesis import AILinearFunctionSynthesis
from qiskit_ibm_transpiler.ai.collection import CollectLinearFunctions
from qiskit.transpiler import PassManager


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


def our_opt(circuit, args):
    bqCirc = OPENQASM2Language().decode(circuit.to_qasm())
    basic_gate_deletion_workflow = Workflow(
        [
            QuickPartitioner(5),
            ForEachBlockPass(
                ScanningGateRemovalPass()
            ),  # Apply gate deletion to each block (in parallel)
            ForEachBlockPass(ConvertZX(args.searchdepth)),
            # ForEachBlockPass(FullReduceZX()),
            ForEachBlockPass(
                ScanningGateRemovalPass()
            ),  # Apply gate deletion to each block (in parallel)
            UnfoldPass(),  # Unfold the blocks back into the original circuit
        ]
    )
    opt_circuit = bqCirc.copy()
    circs = [bqCirc.copy()]
    cnotcount = []
    for _ in range(3):
        with Compiler() as compiler:
            opt_circuit = compiler.compile(
                opt_circuit, workflow=basic_gate_deletion_workflow
            )
            qasm = OPENQASM2Language().encode(opt_circuit)
            opt_circuit = CZ_to_CX(zx.Circuit.from_qasm(qasm))
            qasm = opt_circuit.to_qasm()
            opt_circuit = OPENQASM2Language().decode(qasm)
            circs.append(opt_circuit.copy())
            cnotcount.append(
                opt_circuit.count(CNOTGate()) + opt_circuit.count(CZGate())
            )
    print("POST OPT", cnotcount)
    qasm = OPENQASM2Language().encode(opt_circuit)
    circ = zx.Circuit.from_qasm(qasm)
    return circ, circ.twoqubitcount()


def korbinian_opt(circuit, args):
    bqCirc = OPENQASM2Language().decode(circuit.to_qasm())
    basic_gate_deletion_workflow = Workflow(
        [
            QuickPartitioner(5),
            ForEachBlockPass(
                ScanningGateRemovalPass()
            ),  # Apply gate deletion to each block (in parallel)
            ForEachBlockPass(KorbinianMethodZX()),
            ForEachBlockPass(
                ScanningGateRemovalPass()
            ),  # Apply gate deletion to each block (in parallel)
            UnfoldPass(),  # Unfold the blocks back into the original circuit
        ]
    )
    opt_circuit = bqCirc.copy()
    circs = [bqCirc.copy()]
    cnotcount = []
    for _ in range(3):
        with Compiler() as compiler:
            opt_circuit = compiler.compile(
                opt_circuit, workflow=basic_gate_deletion_workflow
            )
            qasm = OPENQASM2Language().encode(opt_circuit)
            opt_circuit = CZ_to_CX(zx.Circuit.from_qasm(qasm))
            qasm = opt_circuit.to_qasm()
            opt_circuit = OPENQASM2Language().decode(qasm)
            circs.append(opt_circuit.copy())
            cnotcount.append(
                opt_circuit.count(CNOTGate()) + opt_circuit.count(CZGate())
            )
    print("POST OPT", cnotcount)
    qasm = OPENQASM2Language().encode(opt_circuit)
    circ = zx.Circuit.from_qasm(qasm)
    return circ, circ.twoqubitcount()


def full_reduce(circuit, args):
    bqCirc = OPENQASM2Language().decode(circuit.to_qasm())
    basic_gate_deletion_workflow = Workflow(
        [
            QuickPartitioner(5),
            ForEachBlockPass(
                ScanningGateRemovalPass()
            ),  # Apply gate deletion to each block (in parallel)
            # ForEachBlockPass(ConvertZX()),
            ForEachBlockPass(FullReduceZX()),
            ForEachBlockPass(
                ScanningGateRemovalPass()
            ),  # Apply gate deletion to each block (in parallel)
            UnfoldPass(),  # Unfold the blocks back into the original circuit
        ]
    )
    opt_circuit = bqCirc.copy()
    circs = [bqCirc.copy()]
    cnotcount = []
    for _ in range(3):
        with Compiler() as compiler:
            opt_circuit = compiler.compile(
                opt_circuit, workflow=basic_gate_deletion_workflow
            )
            qasm = OPENQASM2Language().encode(opt_circuit)
            opt_circuit = CZ_to_CX(zx.Circuit.from_qasm(qasm))
            qasm = opt_circuit.to_qasm()
            opt_circuit = OPENQASM2Language().decode(qasm)
            circs.append(opt_circuit.copy())
            cnotcount.append(
                opt_circuit.count(CNOTGate()) + opt_circuit.count(CZGate())
            )
    print("POST OPT", cnotcount)
    # opt_circuit.save("compiled.qasm")
    # zx.draw(zx.Circuit.from_qasm_file("compiled.qasm"))
    qasm = OPENQASM2Language().encode(opt_circuit)
    circ = zx.Circuit.from_qasm(qasm)
    return circ, circ.twoqubitcount()


def just_transpile(circuit, args):
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


def ours_transpile(circuit, args):
    circuit, _ = our_opt(circuit, args)
    return just_transpile(circuit, args)


def qiskit_compile(circuit, args):
    bqCirc = OPENQASM2Language().decode(circuit.to_qasm())
    '''IBMProvider.save_account(token=args.API_key)
    provider = IBMProvider(instance="fraunhofer/bayern/iis")
    service = QiskitRuntimeService(
        channel="ibm_quantum", instance="fraunhofer/bayern/iis", token=args.API_key
    )
    print("init done")'''
    circ = bqskit_to_qiskit(bqCirc)
    print("conversion done")
    fake_coupling = []
    for i in range(50):
        for j in range(50):
            fake_coupling.append([i,j])
    """"""
    ai_passmanager = PassManager(
        [
            CollectLinearFunctions(),  # Collect Linear Function blocks
            AILinearFunctionSynthesis(coupling_map=fake_coupling,local_mode=False),  # Re-synthesize Linear Function blocks
        ]
    )
    optim = ai_passmanager.run(circ)
    optim = transpile(
        optim,
        optimization_level=3,
        basis_gates=["h", "cx", "rx", "rz"],
    )

    optim = qiskit_to_bqskit(optim)
    qasm = OPENQASM2Language().encode(optim)
    circ = zx.Circuit.from_qasm(qasm)
    return circ, circ.twoqubitcount()

"""n_qubits = 50
n_depth = 2000
mq_ratio = 1.0
h_ratio = 0.0
t_ratio = 0.0
p_x = 1 - (mq_ratio + h_ratio + (t_ratio / 2))"""

def modified_build_circuit(n_q, n_gates, width, depth, mq_ratio,h_ratio,t_ratio,p_x):
    w_ratio = width // n_q
    d_ratio = depth // n_gates
    print(w_ratio, d_ratio, n_gates)
    circ = zx.Circuit(qubit_amount=width)
    for d in range(d_ratio):
        # tmp = zx.Circuit(n_q)
        # for w in range(w_ratio):
        c = random_circuit(
            n_qubit=n_q,
            num_gates=n_gates,
            p_two_qubit=mq_ratio,
            p_H=h_ratio,
            p_z=t_ratio / 2,
            p_x=p_x,
            clifford_plus_T=True,
        )
        # tmp+=c
        # print(tmp)
        start = np.random.randint(0, width - n_q)
        circ.add_circuit(c, mask=list(range(start, start + n_q)))
    return circ


def bench(seed, args, n_qubits=50, n_depth=2000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    p_x = 1 - (args.mq_ratio + args.h_ratio + (args.t_ratio / 2))
    # first check random circuit
    circuit = random_circuit(
        n_qubit=n_qubits,
        num_gates=n_depth,
        p_two_qubit=args.mq_ratio,
        p_H=args.h_ratio,
        p_z=args.t_ratio / 2,
        p_x=p_x,
        clifford_plus_T=True,
    )
    # run through all compiler passes
    comps = {
        #"qiskit_ai_compile": qiskit_compile,
        #"korbinian": korbinian_opt,
        "ours": our_opt,
        #"full reduce": full_reduce,
        #"transpile": just_transpile,
    }

    results_random = dict()
    results_random["start"] = (circuit, circuit.twoqubitcount())
    print(comps)
    for name, f in comps.items():
        print("iteration", name,args,circuit)
        results_random[name] = f(circuit.copy(), args)
    print(results_random)
    print("RAN RANDOM CIRCUIT")
    # second, check assembled circuit
    circuit = modified_build_circuit(5, 50, n_qubits, n_depth,args.mq_ratio,args.h_ratio,args.t_ratio,p_x)
    results_assembled = dict()
    results_assembled["start"] = (circuit.copy(), circuit.twoqubitcount())
    for name, f in comps.items():
        results_assembled[name] = f(circuit.copy(), args)
        print("iteration", name)
    print(results_assembled)
    print("RAN ASSEMBLED CIRCUIT")
    """with open("results_assembled.pkl", "wb+") as writer:
        pickle.dump({"assembled": results_assembled, "random": results_random}, writer)"""
    return {"assembled": results_assembled, "random": results_random}

def main(args):
    # ray.init(address=os.environ["ip_head"])
    k = list(range(25))
    outs = []
    for i in k:
        print(f"""
        RUNNING ITERATION {i}.
        """)
        b = bench(i,args)
        outs.append(b)
        print("SAVING FILES")
        with open(args.filename, "wb+") as writer:
            pickle.dump(outs, writer)
    pass


def plot(dct):
    keys = list(dct[0]["assembled"].keys())
    # sz=len(keys)
    # width=2
    # height=int(np.ceil(sz/2))
    # fig, axes = plt.subplots(nrows=height, ncols=width,squeeze=False)
    # axes = [k for a in axes for k in a]
    transposed_assembled = [(k, d["assembled"][k][1]) for d in dct for k in keys]
    df = pd.DataFrame(transposed_assembled, columns=["type", "value"])
    print(df)
    ax = sns.barplot(df, x="type", y="value", palette="viridis")
    ax.set_title("Assembled")
    for i in ax.containers:
        print(i)
        ax.bar_label(i, fmt="%.2f")
    plt.savefig("assembled.png")
    plt.close()
    transposed_random = [(k, d["random"][k][1]) for d in dct for k in keys]
    df = pd.DataFrame(transposed_random, columns=["type", "value"])
    ax = sns.barplot(df, x="type", y="value", palette="viridis")
    ax.set_title("Random")
    for i in ax.containers:
        ax.bar_label(i, fmt="%.2f")
    plt.savefig("random.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Benchmark compilers",
        description="Benchmarks compilers against each other",
    )
    parser.add_argument("filename")
    parser.add_argument("searchdepth", type=int)
    parser.add_argument("mq_ratio", type=float)
    parser.add_argument("h_ratio", type=float)
    parser.add_argument("t_ratio", type=float)
    # mq_ratio,h_ratio,t_ratio
    args = parser.parse_args()
    main(args)
