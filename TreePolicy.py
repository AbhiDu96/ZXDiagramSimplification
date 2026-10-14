"""Tree-search policy structures for ZX-diagram rewriting.

`Tree` maintains the search tree (nodes = expanded graph states), samples
which node to expand next (`select_node`, using priorities propagated
root-to-leaf by `top_down`), and samples rewrite actions within the chosen
node (`select_expansion`). `synchronous_forward` is the batched forward pass
used during PPO training in main_tree.py.
"""

import logging
from copy import deepcopy
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Batch


class Effect(NamedTuple):
    """Outcome of `Tree.select`: which tree node to expand, and which
    (flattened) action index to apply in each of the `multi_range` expansions."""

    node_idx: int
    actions: list  # list of 0-dim torch.Tensors, one per expansion


def top_down(w: torch.Tensor, parents: Sequence[int], depths: Sequence[int]) -> torch.Tensor:
    """Propagate weights from roots to leaves: w[c] += w[parent(c)].

    Nodes are processed in increasing depth order (a parent's weight is final
    before its children accumulate it), grouped per depth level with vectorized
    tensor indexing. Bitwise-equivalent to the original BFS over child lists.
    """
    if len(parents) <= 1:
        return w
    device = w.device
    parents_t = torch.as_tensor(parents, dtype=torch.long, device=device)
    depths_t = torch.as_tensor(depths, dtype=torch.long, device=device)
    cand = torch.nonzero(parents_t >= 0).squeeze(-1)  # non-root nodes
    if cand.numel() == 0:
        return w
    order = cand[torch.argsort(depths_t[cand])]
    d_sorted = depths_t[order]
    boundaries = (torch.nonzero(d_sorted[1:] != d_sorted[:-1]).squeeze(-1) + 1).tolist()
    for seg in torch.tensor_split(order, boundaries):
        w[seg] = w[seg] + w[parents_t[seg]]
    return w

def unique_batch(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deduplicate feature rows; returns (unique_rows, inverse_indices)."""
    u, loc = torch.unique(data, dim=0, return_inverse=True)
    return u, loc


def synchronous_forward(cfg: Any, trees: Sequence["Tree"], agent: Any, selected: Sequence[Sequence[int]]):
    """Batched treenet forward over multiple trees (PPO training path).

    Node features across all trees are deduplicated (`unique_batch`) before
    the forward pass. Returns (per-tree output dicts, logits, entropies,
    tree_values).
    """
    # collate the tree nodes
    nodes = []
    indices = []
    for i,t in enumerate(trees):
        indices.extend([i for _ in range(len(t.nodes))])
        nodes.append(torch.stack([inf["feats"] for inf in t.infos]).to(cfg.device))
    children, parents, depths = [], [], []
    indices=torch.LongTensor(indices)
    batch = torch.cat(nodes,0)
    trimmed_batch,loc = unique_batch(batch)
    weights, values = agent.treenet(trimmed_batch)
    total = 0
    for idx, t in enumerate(trees):
        ps = [parent + total if parent >=0 else -1 for parent in t.parents]
        parents.extend(ps)
        depths.extend(t.depths)
        total+=len(ps)
    weights = weights[loc]
    weights_aggregated = top_down(weights.clone(), parents, depths)
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
    return outputs, logits, entropies,tree_vals

class Tree:
    """Search tree over ZX-diagram rewrite states.

    Copy semantics (used by expand -> Tree(self)): `children` is DEEP-copied
    because each tree mutates its own adjacency lists, while `nodes`,
    `zx_states`, `infos`, `rewards`, `depths`, `parents` and `legal` are
    SHALLOW list copies whose elements are intentionally shared with the
    parent tree to avoid duplicating graph objects. These lists are only ever
    appended to, never mutated in place, so the sharing is safe. Do not
    "fix" this without measuring memory on deep searches first.
    """

    def __init__(self, prev_Tree: Optional["Tree"] = None, multi_range: int = 1) -> None:
        self.nodes = []
        self.zx_states = []
        self.children = []
        self.infos=[]
        self.depths=[]
        self.rewards=[]
        self.parents=[]
        # legal[i]: whether node i has any legal actions (cached incrementally
        # to avoid a per-node mask reduction on every select)
        self.legal=[]
        self.best_reward=0.0
        if prev_Tree is not None:
            self.nodes = prev_Tree.nodes.copy()
            self.children= deepcopy(prev_Tree.children)
            self.zx_states = prev_Tree.zx_states.copy()
            self.depths= prev_Tree.depths.copy()
            self.best_reward=prev_Tree.best_reward
            self.rewards=prev_Tree.rewards.copy()
            self.infos=prev_Tree.infos.copy()
            self.parents=prev_Tree.parents.copy()
            self.legal=prev_Tree.legal.copy()
        self.multi_range=multi_range

    def get_best_node(self):
        """ZX graph state of the node with the highest reward seen so far."""
        idx = np.argmax(self.rewards)
        return self.zx_states[idx]

    def get_best_info(self):
        """Env info dict of the node with the highest reward seen so far."""
        idx = np.argmax(self.rewards)
        return self.infos[idx]

    def select_node(self, treenet: Any, selected: Optional[int] = None, device: str = "cpu", cache: Optional[dict] = None):
        """Sample which tree node to expand next, from depth-propagated treenet
        priorities. Returns (node_idx, log_prob, entropy, tree_value).
        Nodes with no legal actions are masked out."""
        weights, values = None, None
        if cache is not None:
            weights, values, weights_aggregated = cache["node_prios"], cache["node_values"], cache["weights_aggregated"]
            weights = weights_aggregated
        else:
            nodes = torch.stack([inf["feats"] for inf in self.infos]).to(device)
            weights, values = treenet(nodes)
            weights = top_down(weights, self.parents, self.depths)
        value_tree = torch.logsumexp(values,-1)
        # propagate up
        # mask out nodes with no legal actions: without this, selecting such
        # a node makes select_expansion sample uniformly over INVALID actions
        legal = torch.tensor(self.legal, device=device)
        weights = weights.masked_fill(~legal, torch.finfo(weights.dtype).min)
        # NOTE: fully-expanded nodes (len(children) == action_mask.sum()) are
        # still selectable; only action-less nodes are excluded.
        leaves = list(range(len(self.nodes)))
        logits = torch.log_softmax(weights[leaves],-1)

        cat = torch.distributions.Categorical(logits = logits)
        if selected is None:
            p = cat.sample()
            selected = leaves[int(p)]
        else:
            p = torch.as_tensor(leaves.index(selected))
        p = p.to(device)
        logit = cat.log_prob(p)
        entropy = cat.entropy()
        return selected, logit, entropy, value_tree
    
    def select_expansion(self, nodenet: Any, node: int, action=None, device: str = "cpu", cache: Optional[dict] = None):
        """Sample a (position, rule) action within `node`, masked to legal
        actions. Returns (flat_action_idx, log_prob, entropy, action_value)."""
        logits, value_act = None, None
        if cache is not None and cache.get("pos_logits",None) is not None:
            # Apparently this is numerically a little dicier than the select_node one...
            # probably just pytorch doing pytorch things (still 7sigfigs, so presumably just IEEE 32bit)
            #print("delta",(cache["node_logits"]-logits).sum(), (cache["node_value"]-values_act).sum())
            pos_logits, action_logits, values_act = cache["pos_logits"], cache["action_logits"], cache["node_value"]
        else:
            b = Batch.from_data_list([self.nodes[node]]).to(device)
            pos_logits, action_logits, values_act = nodenet(b)

        action_mask = self.nodes[node].action_mask
        if action_mask.sum() == 0:
            raise RuntimeError(
                "select_expansion called on a node with no legal actions; "
                "select_node should have masked this node out"
            )
        logits = pos_logits+action_logits
        value_act = values_act.mean()
        # mask invalid actions with the dtype's most negative finite value
        # (not -inf: Categorical clamps to finfo.min anyway, and finite values
        # keep entropy/log_prob computations NaN-free)
        logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        cat = torch.distributions.Categorical(logits = logits.reshape(-1))
        if action is None:
            action = cat.sample()
        logit = logits.reshape(-1).log_softmax(-1)[action.detach()]
        entropy = cat.entropy()
        return action, logit, entropy, value_act
    
    def select(self, bundlenet: Any, selected: Optional[list] = None, device: str = "cpu", cache: Optional[dict] = None) -> tuple[Effect, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose a node and `multi_range` expansion actions for it.

        Returns (effect, log_prob, entropy, value). NOTE: log_prob and
        entropy cover only the node-selection step; the per-action log-probs
        from select_expansion are deliberately not accumulated.
        """
        if selected is None:
            selected = [None, None]
        sel_act, logit_tree, entropy_tree, value_tree = self.select_node(bundlenet.treenet, selected[0], device=device, cache=cache)
        selected[0] = sel_act
        expansion_rules = []
        for i in range(self.multi_range):
            action, logit_act, entropy_act, value_act = self.select_expansion(bundlenet.nodenet, selected[0], selected[1], device=device, cache=cache)
            expansion_rules.append(action)
        effect = Effect(node_idx=int(sel_act), actions=expansion_rules)
        total_log = logit_tree
        total_entropy = entropy_tree
        value = value_tree  # +value_act
        return effect, total_log, total_entropy, value

    def expand(self, effect: Effect, simulator: Any) -> tuple["Tree", float, bool, bool, dict]:
        """Apply `effect` in the simulator and return the extended child tree.

        Termination signals are accumulated and returned but deliberately not
        acted on: tree search can backtrack out of dead ends.
        """
        n = effect.node_idx
        new_tree = Tree(self,multi_range=self.multi_range)
        term, trunc = False, False
        for i in range(self.multi_range):
            nodes, act = self.nodes[n].action_mask.shape
            pos, act = torch.unravel_index(effect.actions[i], (nodes, act))
            obs, rew, t1, t2, info =  simulator.step(position=pos,action=act, pyzx_state=self.zx_states[n].copy())
            term = term or t1
            trunc = trunc or t2
            # accumulate on the child tree; new_tree.best_reward was copied
            # from self at construction, so this tracks the same running max
            # without mutating the parent tree
            new_tree.best_reward = max(new_tree.best_reward, rew)
            new_tree.nodes.append(obs.Graph)
            new_tree.zx_states.append(obs.state_zx_graph)
            new_tree.children[n].append(len(new_tree.nodes)-1)
            new_tree.children.append([])
            new_tree.depths.append(new_tree.depths[n]+1)
            new_tree.parents.append(n)
            new_tree.legal.append(bool(obs.Graph.action_mask.sum() > 0))
            new_tree.rewards.append(rew)
            new_tree.infos.append(info)
        # we ignore termination signals since we use tree search
        # and can backtrack out of dead ends
        return new_tree, new_tree.best_reward, term, trunc, info


def extract_optimal_path(t: Tree) -> tuple[list, Tree]:
    """Backtrack from the highest-reward node to the root; returns the path
    in forward (root -> best) order together with the tree."""
    idx = np.argmax(t.rewards)
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

def show_rules(transformed_steps: list) -> None:
    """Log the rewrite rule applied at each step of an extracted path."""
    logging.info("applied rules: %s", [k[1]["applied_rule"] for k in transformed_steps])


def start_tree(obs: Any, zx: Any, info: dict, multi_range: int = 1) -> Tree:
    """Create a single-root-node tree from an env reset observation."""
    t = Tree()
    t.nodes = [obs]
    t.zx_states=[zx]
    t.depths=[0]
    t.children=[[]]
    t.parents=[-1]
    t.legal=[bool(obs.action_mask.sum() > 0)]
    r0=info["reward"]
    t.rewards=[r0]
    t.best_reward=r0
    t.infos.append(info)
    t.multi_range=multi_range
    return t