import logging
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
    startroot = len(roots)
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
    #print("calls",calls,"number of roots",startroot)
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
    def __init__(self,prev_Tree = None, multi_range=1):
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
        self.multi_range=multi_range

    def get_best_node(self):
        idx = np.argmax(self.rewards)
        return self.zx_states[idx]

    def get_best_info(self):
        idx = np.argmax(self.rewards)
        return self.infos[idx]

    def select_node(self,treenet,selected=None,device="cpu",cache=None):
        weights, values=None,None
        if cache is not None:
            weights, values, weights_aggregated = cache["node_prios"], cache["node_values"], cache["weights_aggregated"]
            #weights = collect_leaves(weights.clone(),self.children,child_to_parents(self.children))
            #weights = top_down(weights,child_to_parents(self.children),self.children)
            weights = weights_aggregated
        else:
            nodes = torch.stack([inf["feats"] for inf in self.infos]).to(device)
            weights, values = treenet(nodes)
            #weights = collect_leaves(weights,self.children,child_to_parents(self.children))
            weights = top_down(weights,child_to_parents(self.children),self.children)
        value_tree = torch.logsumexp(values,-1)
        # propagate up
        leaves = list(range(len(self.nodes)))
        """leaves=[]
        for i,child in enumerate(self.children):
            if len(child)< self.nodes[i].action_mask.sum():
                leaves.append(i)
        # Now select the node
        #compute mask
        mask = torch.tensor([n.action_mask.sum()>0 for n in self.nodes],device=device)[leaves].float()
        weights = weights[leaves]-1e26*(1-mask)"""
        logits = torch.log_softmax(weights[leaves],-1)

        cat = torch.distributions.Categorical(logits = logits)
        if selected is None:
            s=cat.sample()
            selected = leaves[s]
        p=torch.tensor(leaves.index(selected),device=device)
        logit = cat.log_prob(p)
        entropy = cat.entropy()
        return selected, logit, entropy, value_tree
    
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
        expansion_rules=[]
        for i in range(self.multi_range):
            action, logit_act, entropy_act, value_act = self.select_expansion(bundlenet.nodenet,selected[0],selected[1],device=device,cache=cache)
            expansion_rules.append(action)
        effect = [torch.tensor(sel_act,device=device)]+expansion_rules
        total_log = logit_tree
        total_entropy = entropy_tree
        value = value_tree#+value_act
        #print("effect",effect)
        return effect, total_log, total_entropy, value
    
    def expand(self,effect, simulator):
        # new state
        n=effect[0]
        new_tree = Tree(self,multi_range=self.multi_range)
        term, trunc = False, False
        for i in range(self.multi_range):
            nodes, act = self.nodes[n].action_mask.shape
            pos,act = torch.unravel_index(effect[1+i],(nodes,act))
            obs, rew, t1, t2, info =  simulator.step(position=pos,action=act, pyzx_state=self.zx_states[n].copy())
            term = term or t1
            trunc = term or t2
            #print("observed reward",rew,"current best is", self.best_reward)
            self.best_reward = max(self.best_reward,rew)
            new_tree.nodes.append(obs.Graph)
            new_tree.zx_states.append(obs.state_zx_graph)
            new_tree.children[effect[0]].append(len(new_tree.nodes)-1)
            new_tree.children.append([])
            new_tree.depths.append(new_tree.depths[effect[0]]+1)
            new_tree.rewards.append(rew)
            new_tree.infos.append(info)
        # we ignore termination signals since we use tree search
        # and can backtrack out of dead ends
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
    logging.info("applied rules: %s", [k[1]["applied_rule"] for k in transformed_steps])

def start_tree(obs, zx, info, multi_range=1) -> Tree:
    t=Tree()
    t.nodes = [obs]
    t.zx_states=[zx]
    t.depths=[0]
    t.children=[[]]
    r0=info["reward"]
    t.rewards=[r0]
    t.best_reward=r0
    t.infos.append(info)
    t.multi_range=multi_range
    return t