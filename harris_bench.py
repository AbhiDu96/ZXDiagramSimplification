import pickle
import torch
import numpy as np
import utils
from zx_env import zx_env
from zx_env import extract_circuit
from TreePolicy import Tree
from zx_env import random_circuit
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit
import pyzx as zx
from qiskit.compiler import transpile



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

def optimizing(state,):
    with open("model.pkl","rb") as reader:
        agent=pickle.load(reader)
    env = make_env("cpu")
    start = state.copy().to_graph()
    #print(check_equality(zx.Circuit.from_graph(start.copy().to_graph()).to_graph(), start))
    next_obs = None
    for it in range(1):
        next_obs, info = env.reset(initital_circuit_graph=start)
        next_obs = start_tree(next_obs.Graph, next_obs.state_zx_graph, info=info)
        #print("check",check_equality(start, next_obs.get_best_node()))
        max_rew=0
        for step in range(128):
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
    # get random state
    state = random_circuit(clifford_plus_T=True)
    optimized_state = optimizing(state)
    print(optimized_state)