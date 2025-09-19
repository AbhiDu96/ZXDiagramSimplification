import bqskit
from bqskit.compiler import Workflow
from bqskit.passes import QuickPartitioner, ForEachBlockPass, ScanningGateRemovalPass, UnfoldPass
import bqskit
from bqskit.ext import qiskit_to_bqskit
from bqskit.compiler import Compiler
from bqskit.compiler.basepass import BasePass
from bqskit.ir.gates import CircuitGate, CNOTGate
from bqskit.ir.lang.qasm2 import OPENQASM2Language

from bqskit.compiler.basepass import BasePass
from bqskit.ir.gates import CircuitGate, CNOTGate
import logging
import pyzx as zx
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import time
import qiskit.qasm2
import torch
import qiskit

from zx_env import extract_circuit
from zx_env import random_circuit
from zx_env import zx_env
import matplotlib.pyplot as plt
from copy import deepcopy
from model import BundleNet
import os
from brute_force_CX_opt import optimize_CX_circuit
from pyzx.simplify import to_graph_like
import utils
from TreePolicy import Tree
import numpy as np
from zx_env.general_utils.utils import check_equality
from zx_env import extract_circuit
import importlib
korbinianbench = importlib.import_module("pyzx_korbinian.pyzx-heuristics.korbinianbench")

import pickle

#bqskit.enable_logging(True)

class PrintCNOTsPass(BasePass):
    async def run(self, circuit, data) -> None:
        logging.info(f"BQSKit step, current CNOT count:  {circuit.count(CNOTGate())}")
def cnot_count_scorer(ops):
    score = 0.0
    for op in ops:
        if op.gate == CNOTGate():
            score += 10
        else:
            score+=1
    #logging.info(f"score is {score}")
    return score

def make_env(device):
    n_qubits = 20
    n_depth = 500
    mq_ratio= 0.6
    h_ratio= 0.2
    t_ratio = 0.2
    p_x = 1-(mq_ratio+h_ratio+(t_ratio/2))
    env = zx_env(
        mutate_probability = 0,
        mutation_steps = 0,
        reward_fn = "normalized_cnot_count_reward",
        n_qubits=n_qubits,
        depth = n_depth,
        mq_ratio=mq_ratio,
        t_ratio=t_ratio,
        h_ratio=h_ratio,
        max_steps= 4096,
        negative_reward_mean= 0,
        negative_reward_std=0,
        rules_list=["bialgebra","spider_fusion", "euler", "pi_copy"],
        full_fuse_every_step=False,
        reduce_at_reset=False
    )
    
    env = utils.GraphMaskWrapper(env,device)
    return env

def start_tree(obs,zx,info):
    t=Tree()
    t.nodes = [obs]
    t.zx_states=[zx]
    t.depths=[0]
    t.children=[[]]
    r0=info["reward"]
    t.rewards=[r0]
    t.best_reward=r0
    t.infos.append(info)
    t.multi_range=16
    return t

def optimizing(state,searchdepth):
    model = torch.load("runs/QACI_graph_optim__20MIO_32_envs_changed_rules_optimized_5qubit_multirange_4_depth_16__1__1726148732/saves/model-819200.pth")
    agent=BundleNet(16,
            5,
            256,
            4,
            8,
            "cpu",
            "GAT",
    )
    agent.load_state_dict(model)
    logging.info(f"model keys {model.keys()}")
    env = make_env("cpu")
    start = state.copy().to_graph()
    #print(check_equality(zx.Circuit.from_graph(start.copy().to_graph()).to_graph(), start))
    next_obs = None
    for it in range(1):
        next_obs, info = env.reset(initital_circuit_graph=start)
        next_obs = start_tree(next_obs.Graph, next_obs.state_zx_graph, info=info)
        #print("check",check_equality(start, next_obs.get_best_node()))
        max_rew=0
        for step in range(searchdepth):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                # make a temporary batch from our data:
                effect, total_log, _, value = next_obs.select(agent)

            # TRY NOT TO MODIFY: execute the game and log data.
            new_tree, rew, term, trunc, info = next_obs.expand(effect, env)
            if term or trunc:
                print("exiting due to termination signal")
                return next_obs.get_best_node()
            max_rew=max(rew,max_rew)
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


class OutputGraphImage(BasePass):

    async def run(self,circuit,data) -> None:
        qs = bqskit.ext.bqskit_to_qiskit(circuit)
        pyzx = zx.Circuit.from_qasm(qiskit.qasm2.dumps(qs))
        fig = zx.draw(pyzx)
        fig.savefig(f"out-{time.time()}.png")
        fig.close()

class ConvertZX(BasePass):
    def __init__(self,searchdepth) -> None:
        super().__init__()
        self.searchdepth=searchdepth

    async def run(self,circuit,data) -> None:
        t0=time.time()
        nm=t0*100
        cnotcount = circuit.count(CNOTGate())
        if circuit.count(CNOTGate()) <= 2:
            return
        qs = bqskit.ext.bqskit_to_qiskit(circuit)
        pyzx = zx.Circuit.from_qasm(qiskit.qasm2.dumps(qs))
        '''with open(f"original/{int(nm)}-graph.pkl","wb+") as writer:
            pickle.dump(pyzx,writer)'''
        try:
            out = optimizing(pyzx.copy(),self.searchdepth)
        except:
            return
        # DO SOME CURSED OPTIMIZATIONS HERE
        logging.info(f"I'm running whatever you want in pyzx {out}")
        pyzx,_=extract_circuit(out)
        '''with open(f"optimized/{int(nm)}-graph.pkl","wb+") as writer:
            pickle.dump(pyzx,writer)'''
        qasm = pyzx.to_qasm()
        circ = OPENQASM2Language().decode(qasm)
        #print("time taken", time.time()-t0, "advantage", cnotcount-circ.count(CNOTGate()))
        circuit.become(circ)


class FullReduceZX(BasePass):

    async def run(self,circuit,data) -> None:
        t0=time.time()
        cnotcount = circuit.count(CNOTGate())
        if circuit.count(CNOTGate()) <= 2:
            return
        qs = bqskit.ext.bqskit_to_qiskit(circuit)
        pyzx = zx.Circuit.from_qasm(qiskit.qasm2.dumps(qs))
        g=pyzx.to_graph()
        zx.full_reduce(g)
        pyzx = zx.extract_circuit(g)
        qasm = pyzx.to_qasm()
        circ = OPENQASM2Language().decode(qasm)
        #print("time taken", time.time()-t0, "advantage", cnotcount-circ.count(CNOTGate()))
        circuit.become(circ)



class KorbinianMethodZX(BasePass):

    async def run(self,circuit,data) -> None:
        t0=time.time()
        cnotcount = circuit.count(CNOTGate())
        if circuit.count(CNOTGate()) <= 2:
            return
        qs = bqskit.ext.bqskit_to_qiskit(circuit)
        pyzx = zx.Circuit.from_qasm(qiskit.qasm2.dumps(qs))
        pyzx,_=korbinianbench.korbinian(pyzx)
        qasm = pyzx.to_qasm()
        circ = OPENQASM2Language().decode(qasm)
        #print("time taken", time.time()-t0, "advantage", cnotcount-circ.count(CNOTGate()))
        circuit.become(circ)
