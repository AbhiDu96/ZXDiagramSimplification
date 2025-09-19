import torch
import torch_geometric
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
import numpy as np
from collections import defaultdict
from copy import deepcopy
import time


def child_to_parents(child):
    parents = [-1 for _ in range(len(child))]
    for i,cs in enumerate(child):
        for c in cs:
            parents[c] = i
    return parents

#@torch.compile(dynamic=True)
def collect_leaves(w,child,parents):
    #w = w0.clone()
    loop = child[0]
    #parents = child_to_parents(child)
    while len(loop)>0:
        w[loop] = w[loop] + w[parents[loop]]
        loop = [c for lp in loop for c in child[lp]]
    return w


def bottom_up(w,parents,children):
    # get leave nodes
    #print(w.shape,parents.shape)
    parents=torch.tensor(parents)
    leaves = torch.unique(torch.tensor([idx for idx,child in enumerate(children) if len(child)==0]))
    #print("logging",leaves.shape,parents.shape)
    while len(leaves)>0:
        nontrivial_leaves = leaves[torch.where(parents[leaves]>=0)]
        w[parents[nontrivial_leaves]] = w[nontrivial_leaves] + w[parents[nontrivial_leaves]]
        leaves = torch.unique(parents[nontrivial_leaves])
    return w

def top_down(w,parents,children):
    if len(parents) == 1:
        return w
    # get leave nodes
    parents=torch.tensor(parents)
    #leaves = torch.unique(torch.tensor([idx for idx,child in enumerate(children) if len(child)==0]))
    roots = [idx for idx, parent in enumerate(parents) if parent<0]
    children = np.array(children,dtype=object)
    #print(leaves,parents)
    calls=0
    while len(roots)>0:
        chil=children[roots]
        lns = [len(c) for c in chil]
        chil=sum(chil,[])
        roots = np.repeat(roots,lns)
        w[chil] = w[chil] + w[roots]
        roots = chil
        calls+=1
    print("calls",calls)
    return w

def unique_batch(data: torch.Tensor):
    u,loc=torch.unique(data,dim=0,return_inverse =True)
    return u,loc


def synchronous_forward(cfg,trees, agent, selected):
    # collate the tree nodes
    nodes = []
    indices = []
    for i,t in enumerate(trees):
        indices.extend([i for _ in range(len(t.nodes))])
        nodes.append(torch.stack([inf["feats"] for inf in t.infos]).to(cfg.device))
    children, parents = [], []
    indices=torch.LongTensor(indices)
    batch = torch.cat(nodes,0)
    trimmed_batch,loc = unique_batch(batch)
    weights, values = agent.treenet(trimmed_batch)
    total = 0
    for idx, t in enumerate(trees):
        children.extend([[c+total for c in child] for child in t.children])
        ps = [parent + total if parent >=0 else -1 for parent in child_to_parents(t.children)]
        parents.extend(ps)
        total+=len(ps)
    weights = weights[loc]
    weights_aggregated = top_down(weights.clone(),torch.tensor(parents),children)
    values = values[loc]
    # now forward one huge block
    outputs = []
    logits = torch.zeros(len(trees),device=cfg.device)
    entropies = torch.zeros(len(trees),device=cfg.device)
    tree_vals = torch.zeros(len(trees),device=cfg.device)
    for i in range(len(trees)):
        # logits:
        logs = torch.log_softmax(weights_aggregated[indices==i],-1)
        entropies[i] = -(logs*logs.exp()).sum()
        logits[i] = logs[selected[i][0]]
        tree_vals[i] = torch.logsumexp(values[indices==i],-1)
        outputs.append(
            {
                "node_prios": weights[indices==i],
                "node_values":values[indices==i],
                "weights_aggregated": weights_aggregated[indices==i],
                "entropy":entropies[i],
                "logit":logits[i],
                "value":tree_vals[i]
            }
        )
    agent = agent
    return outputs, logits, entropies,tree_vals

class Tree:
    def __init__(self,prev_Tree = None):
        self.nodes = []
        self.zx_states = []
        self.children = []
        self.infos=[]
        self.depths=[]
        self.rewards=[]
        self.best_reward=0.0
        if prev_Tree is not None:
            self.nodes = prev_Tree.nodes.copy()
            self.children= deepcopy(prev_Tree.children)
            self.zx_states = prev_Tree.zx_states.copy()
            self.depths= prev_Tree.depths.copy()
            self.best_reward=prev_Tree.best_reward
            self.rewards=prev_Tree.rewards.copy()
            self.infos=prev_Tree.infos.copy()
        self.parents_map=child_to_parents(self.children)

    def get_best_node(self):
        idx = np.argmax(self.rewards)
        return self.zx_states[idx]

    def get_best_info(self):
        idx = np.argmax(self.rewards)
        return self.infos[idx]

    def select_node(self,treenet,selected=None,device="cpu",cache=None):
        selected = np.argmax([inf["feats"][3] for inf in self.infos])
        logit = 0
        entropy = 0
        return selected, logit, entropy, 0
    
    def select_expansion(self,nodenet, node, action=None,device="cpu",cache=None):
        logits, value_act=None,None
        if cache is not None and cache.get("pos_logits",None) is not None:
            # Apparently this is numerically a little dicier than the select_node one...
            # probably just pytorch doing pytorch things (still 7sigfigs, so presumably just IEEE 32bit)
            #print("delta",(cache["node_logits"]-logits).sum(), (cache["node_value"]-values_act).sum())
            pos_logits, action_logits, values_act = cache["pos_logits"], cache["action_logits"], cache["node_value"]
        else:
            b = Batch.from_data_list([self.nodes[node]]).to(device)
            pos_logits, action_logits, values_act = nodenet(b)

        action_mask = self.nodes[node].action_mask
        logits = pos_logits+action_logits
        value_act = values_act.mean()
        logits = logits-1e12*(1-action_mask)
        cat = torch.distributions.Categorical(logits = logits.reshape(-1))
        if action is None:
            action = cat.sample()
        logit = logits.reshape(-1).log_softmax(-1)[action.detach()]
        entropy = cat.entropy()
        return action, logit, entropy, value_act
    
    def select(self, bundlenet, selected=None,device="cpu",cache=None):
        if selected is None:
            selected=[None,None]
        sel_act, logit_tree, entropy_tree, value_tree = self.select_node(bundlenet.treenet,selected[0],device=device,cache=cache)
        selected[0]=sel_act
        action, logit_act, entropy_act, value_act = self.select_expansion(bundlenet.nodenet,selected[0],selected[1],device=device,cache=cache)
        effect = [torch.tensor(sel_act,device=device),action]
        total_log = logit_tree
        total_entropy = entropy_tree
        value = value_act+value_tree
        return effect, total_log, total_entropy, value
    
    def expand(self,effect, simulator):
        # new state
        n=effect[0]
        nodes, act = self.nodes[n].action_mask.shape
        pos,act = torch.unravel_index(effect[1],(nodes,act))
        obs, rew, term, trunc, info =  simulator.step(position=pos,action=act, pyzx_state=self.zx_states[effect[0]].copy())
        print("observed reward",rew,"current best is", self.best_reward)
        self.best_reward = max(self.best_reward,rew)
        new_tree = Tree(self)
        new_tree.nodes.append(obs.Graph)
        new_tree.zx_states.append(obs.state_zx_graph)
        new_tree.children[effect[0]].append(len(new_tree.nodes)-1)
        new_tree.children.append([])
        new_tree.depths.append(new_tree.depths[effect[0]]+1)
        new_tree.rewards.append(rew)
        new_tree.infos.append(info)
        # we ignore termination signals since we use tree search
        # and can backtrack out of dead ends
        term, trunc = False, False
        return new_tree, self.best_reward, term, trunc, info


def extract_optimal_path(t:Tree):
    # get the optimal index
    idx=np.argmax(t.rewards)
    # get the node associated with it
    transform_steps = [(t.zx_states[idx],t.infos[idx])]
    while idx != 0:
        for idx_tmp, c in enumerate(t.children):
            if idx in c:
                idx = idx_tmp
                break
        transform_steps.append((t.zx_states[idx],t.infos[idx]))
    # flip to turn backtracking into "forward tracking"
    return (transform_steps[::-1],t)

def show_rules(transformed_steps):
    print([k[1]["applied_rule"] for k in transformed_steps])

def start_tree(obs,zx,info) -> Tree:
    t=Tree()
    t.nodes = [obs]
    t.zx_states=[zx]
    t.depths=[0]
    t.children=[[]]
    r0=info["reward"]
    t.rewards=[r0]
    t.best_reward=r0
    t.infos.append(info)
    return t

import pyzx as zx
import numpy as np
from zx_env import extract_circuit
from zx_env import random_circuit
from zx_env import zx_env
import matplotlib.pyplot as plt
from copy import deepcopy
from ppo import make_env
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
    t = cfg.exp_name
    print(f"validation_results/results_{t}")
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
        c=ds[i]
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
        res4.append(deploy_agents.remote(cfg,deepcopy(c),agent))
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
        "greedy CNOT",
        "greedy CNOT - don't count SWAPS",
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
