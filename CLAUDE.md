# ZX Diagram Simplification — Codebase Overview

This repository is the implementation for **"Optimizing Quantum Circuits via ZX Diagrams using Reinforcement Learning and Graph Neural Networks"** (arXiv:2504.03429). The `Circopt-RL-ZXCalc/` submodule is the codebase for its predecessor paper (arXiv:2312.11597, published in *Quantum* 2025). Together, they form a progression: the submodule is the baseline, and the main repo is the direct follow-up that improves upon it.

---

## The Two Papers

### Paper 1 — `Circopt-RL-ZXCalc/` (arXiv:2312.11597)
**"Optimizing Quantum Circuits with the ZX-Calculus in the NISQ era using Reinforcement Learning"**

- **Problem**: Reduce CNOT gate counts in quantum circuits using ZX-calculus rewrite rules, learned via RL instead of hand-crafted heuristics.
- **Approach**: Trains a PPO agent using a GNN (GATv2Conv) as both actor and critic on the ZX graph.
- **Key result**: Trained only on 5-qubit circuits, generalizes to beat PyZX on circuits up to 80 qubits / 2100 gates.

### Paper 2 — main repo (arXiv:2504.03429)
**"Optimizing Quantum Circuits via ZX Diagrams using Reinforcement Learning and Graph Neural Networks"**

- **Problem**: Same core problem, but with an improved action representation, a novel GNN architecture, and tree-search for lookahead.
- **Approach**: Uses an action-at-each-node GNN formulation with GraphCrossAttention + an MCTS-like tree search policy.
- **Key result**: Competitive with state-of-the-art circuit optimizers, better generalization across diverse circuits.

---

## How They Connect: Progression of Changes

### 1. Action Space Representation

**Paper 1** augments the ZX graph by appending extra "action nodes" for each applicable rule (LC, PV, PVB, PVG, ID, GF, STOP), and adds a flat identifier lookup of size 3000. The action space is essentially a giant flat integer (up to ~9M options flattened into a Categorical distribution).

```python
# Circopt-RL-ZXCalc/rl-zx/gym-zx/gym_zx/envs/zx_env.py:397
# For LC: creates a new action node feature[12]=1, connects it to the graph node
node_feature[12] = 1.0
identifier.append(node * self.shape + node)
```

**Paper 2** abandons this augmentation entirely. Instead, each ZX graph node carries an **action mask** tensor of shape `(n_nodes, n_rules)`, and the model directly predicts a `(node, action)` pair. This is the `GraphMask` dataclass in `utils.py`:

```python
# utils.py:14
@dataclass
class GraphMask:
    Graph: data.Data
    action_mask: torch.Tensor  # shape: (n_nodes, n_rules)
    state_zx_graph: Any
```

This is a fundamental design change — actions are **local to nodes** rather than global graph augmentations.

---

### 2. GNN Architecture

**Paper 1** (`rl_agent.py`) uses separate actor and critic GNNs:
- **Actor**: 5-layer `GATv2Conv` with 17 node features + 7 edge features → scalar logit per node
- **Critic**: 5-layer `GATv2Conv` with 12 node features + 2 edge features → `AttentionalAggregation` → scalar value

**Paper 2** (`model.py`) introduces two novel architectures:

**`ActionModel` with `GraphCrossAttention`** — the primary PPO agent:
- Each node embeds an action index → action embedding of shape `(n_nodes, n_actions, action_dim)`
- `GraphCrossAttention` propagates information across edges using a GAT-style weighted aggregation, but cross-attending action embeddings to neighbor node summaries
- Multi-round message passing, then linear heads for policy logits and value per `(node, action)` pair

**`BundleNet = TreeNet + DummyActionModel`** — used in `main_tree.py`:
- **`TreeNet`**: takes 8-dimensional node features → priority and value per node (used to score tree nodes during tree search)
- **`DummyActionModel`**: uniform logits over actions (the tree handles action selection)

---

### 3. Tree Search (Major New Contribution)

Paper 1 has **no tree search** — it's a flat PPO rollout. Paper 2 introduces two tree search implementations:

**`MCTS.py`** — a more classical MCTS-like approach:
- Maintains a `Node` tree rooted at the initial ZX state
- Selects nodes using a softmax over cumulative weight scores (depth-normalized)
- Expands by running the ZX simulator and querying the model for `(W, V)` — policy weights and value estimates
- Extracts data for PPO update using GAE bootstrapping over the tree

**`TreePolicy.py`** — the cleaner production version used by `main_tree.py`:
- `Tree` class maintains lists of `nodes`, `zx_states`, `children`, `depths`, `rewards`, `infos`
- `select()` calls `TreeNet` to score all nodes, runs top-down weight propagation (`top_down()`), samples a node, then calls `DummyActionModel` for action selection
- `expand()` applies `multi_range` actions from the selected node, creating multiple children per expansion step
- `extract_optimal_path()` traces back the highest-reward leaf to get the best rewrite sequence
- `synchronous_forward()` batches all tree nodes across all parallel trees into one GNN call using `unique_batch()` for deduplication

---

### 4. Feature Representation

**Paper 1** uses rich separate features:
- Policy obs: 17-dim node features (one-hot phase ×8, frontier in/out, gadget, LC/ID/PV/GF/STOP node flags), 7-dim edge features
- Value obs: 12-dim node features, 2-dim edge features

**Paper 2** (Appendix C variant) uses a compact 8-dimensional feature vector with a `SlowNorm` normalization layer — a custom running-mean normalizer that keeps consistent behavior between train and eval (unlike BatchNorm):

```python
# model.py:150
class SlowNorm(nn.Module):
    # Exponentially tracks historical mean/std
    # Behaves identically during train AND eval — no distribution shift
```

Graph edges are promoted to **virtual nodes** via `expand_graph()` in `utils.py`, making both nodes and edges first-class in the GNN. The graph is then symmetrized with `ToUndirected()`.

---

### 5. Parallelism and Infrastructure

**Paper 1**: Single-process, sequential environment stepping.

**Paper 2**: Uses **Ray** for distributed rollout collection. `deploy_train` is a `@ray.remote` function; `EnvActor` wraps each environment in a Ray actor. Multiple environments are dispatched simultaneously and joined before the PPO update:

```python
# main_tree.py:252
for i in range(cfg.env.num_envs):
    rollouts.append(deploy_train.remote(cfg, agent))
for i, rollout in enumerate(rollouts):
    (...) = ray.get(rollout)
```

---

### 6. ZX Environment Design

Both papers share the same underlying ZX rules set (from PyZX): **LC** (local complementation), **PV** (pivot parallel), **PVB** (pivot boundary), **PVG** (pivot gadget), **ID** (identity removal), **GF** (gadget fusion), and **STOP**.

Key difference in how the environment is consumed:

| Aspect | Paper 1 | Paper 2 |
|--------|---------|---------|
| State returned | Raw PyZX graph (processed into augmented nx graph) | `GraphMask` (PyG `Data` + action mask + zx clone) |
| Action input | Single flat integer (up to ~9M) | `(node_position, action_index)` pair |
| Reward | CNOT reduction vs PyZX + teleport-reduce at episode end | CNOT reduction, optionally sparsified or noised |
| Env wrappers | None | `RewardTransform`, `GraphMakeDirected`, `GraphMaskWrapper` |

---

### 7. Integration and Benchmarking

Paper 2 adds `bqskit_pass.py` — a **BQSKit pass wrapper** that lets the trained RL agent plug into standard quantum compilation pipelines, enabling "peephole optimization" mode for practical deployment. `bench_compilers.py` provides CLI benchmarking against other compilers (PyZX, etc.).

---

## Summary of the Research Lineage

```
Paper 1 (2312.11597) — Circopt-RL-ZXCalc/
  • PPO + GATv2 actor/critic
  • Augmented-graph action nodes (flat 3000-dim action space)
  • Separate policy/value observations (17+7 / 12+2 features)
  • Single-process training
  • Proved RL+ZX can generalize from 5q to 80q circuits
         ↓  builds on and improves
Paper 2 (2504.03429) — main repo
  • PPO + GraphCrossAttention (action-at-node formulation)
  • Action mask per node (compact, local action space)
  • 8-dim node features + edge-as-virtual-node graph expansion
  • MCTS-like tree search (MCTS.py) + Tree-policy search (TreePolicy.py)
  • Ray-distributed parallel environments
  • BundleNet = TreeNet (score tree nodes) + NodeNet (score actions)
  • BQSKit integration for real-world compiler pipelines
  • SlowNorm for stable normalization across train/eval
```

The core intellectual progression is: Paper 1 proves the RL+GNN+ZX idea works and generalizes. Paper 2 then redesigns the action representation to be fundamentally node-local, adds tree search for lookahead, and improves the GNN architecture to handle this new formulation — resulting in a system that is more practical (BQSKit-compatible), more scalable (Ray), and architecturally cleaner (no augmented action nodes cluttering the graph).

---

## Possible Future Improvements (from Circopt-RL-ZXCalc)

These are ideas for integrating elements of Paper 1's codebase into Paper 2's repo.

### 1. Terminal Flow Optimization at Inference
**Difficulty**: Very easy (2–3 lines). **Impact**: High — free improvement, no retraining needed.

Paper 1's env applies a deterministic cleanup at episode end (`zx_env.py:175`):
```python
zx.teleport_reduce(g)
zx.to_graph_like(g)
zx.flow_2Q_simp(g)
c2 = zx.extract_simple(g, up_to_perm=True).to_basic_gates()
```
Paper 2 goes straight to circuit extraction after `tree.get_best_node()` with no final pass. Applying this post-hoc in `validation()` (`main_tree.py`) and in `bqskit_pass.py` would reduce gate counts further without touching training.

### 2. `CategoricalMasked` with Correct Entropy
**Difficulty**: Easy (drop-in). **Impact**: Medium — fixes a silent NaN risk in the PPO entropy loss.

Paper 1's `CategoricalMasked` (`rl_agent.py:13`) explicitly zeroes masked entries in the entropy computation:
```python
p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
return -p_log_p.sum(-1)
```
Paper 2 adds `-inf` to masked logits and calls PyTorch's standard `.entropy()`, which can produce `0 * log(0) = nan` in edge cases and silently corrupt the entropy bonus term in the PPO loss.

### 3. Attentional Aggregation for the Critic
**Difficulty**: Moderate (new nn.Module). **Impact**: Medium-high — better value estimates.

Paper 2's `TreeNet` aggregates per-node value predictions with `logsumexp`, which is dominated by a single high-scoring node. Paper 1's critic uses a learned `AttentionalAggregation` (`rl_agent.py:52`) that attends over node features to produce a graph-level value. Replacing `logsumexp` in `TreePolicy.synchronous_forward()` with proper attention pooling would give the critic a more informed global view.

### 4. Separate Feature Channels for Actor vs. Critic
**Difficulty**: Moderate (feature split). **Impact**: Medium — cleaner optimization target for the critic.

Paper 1 uses different features for actor (17-dim node + 7-dim edge, includes action-type flags) and critic (12-dim node + 2-dim edge, pure structural ZX info). Paper 2's `TreeNet` uses the same 8-dim features for both `priority_prediction` and `value_prediction`. The critic would benefit from a clean structural representation without action-applicability annotations mixed in.

### 5. `basic_optimise()` in Reward Computation
**Difficulty**: Easy. **Impact**: Medium — removes a bias between training reward and real deployment.

Paper 1 calls `zx.basic_optimization(c.copy(), do_swaps=True)` before counting gates for the reward at each step. Paper 2 counts gates directly from the extracted circuit. Since baseline compilers apply basic optimization, the training reward in Paper 2 is systematically inflated relative to inference — fixing this makes the reward signal more honest.

### 6. Per-Rule Logging and Win-Rate Tracking
**Difficulty**: Easy. **Impact**: Low on performance, high on interpretability.

Paper 1 tracks per-rule usage counts (LC, PV, PVB, PVG, ID, GF) per episode and logs win/draw/loss vs. PyZX. Paper 2 only logs aggregate reward and CNOT count. Adding rule-level breakdown to the tensorboard logging in `main_tree.py` would make it much easier to diagnose whether the policy is over-specializing on a subset of rules.

---

## Key Files

| File | Purpose |
|------|---------|
| `main_tree.py` | Main training entry point (tree search + PPO) |
| `ppo.py` | Alternative flat-PPO training loop |
| `model.py` | All GNN model definitions (ActionModel, BundleNet, TreeNet, GraphCrossAttention) |
| `TreePolicy.py` | Tree data structure + synchronous batch forward pass |
| `MCTS.py` | MCTS-style tree expansion with PPO update |
| `utils.py` | GraphMask, env wrappers, graph expansion utilities |
| `multiEnv.py` | Ray-based parallel environment actor |
| `bqskit_pass.py` | BQSKit compiler pass wrapper |
| `bench_compilers.py` | CLI benchmarking against other compilers |
| `conf/config.yaml` | Hydra config root |
| `Circopt-RL-ZXCalc/rl-zx/rl_agent.py` | Paper 1 AgentGNN (GATv2Conv actor/critic) |
| `Circopt-RL-ZXCalc/rl-zx/gym-zx/gym_zx/envs/zx_env.py` | Paper 1 ZX environment |

## Training

```bash
# Main entry point (tree search variant, Appendix C features)
python -u main_tree.py +algorithm=PPO exp_name="my_run" +model=GATActionModel \
  +env=more_complex_more_rules_ranges env.num_envs=32 \
  model.model_type="ActionAtt" model.n_message_passing=4 \
  algorithm.total_timesteps=20_000_000 algorithm.num_steps=129 \
  device="cpu" max_tree_size=128 multi_range=4 \
  algorithm.learning_rate=3e-3 env.n_qubits=5

# Flat PPO variant
python -u ppo.py ...

# Benchmark pretrained model
python bench_compilers.py ...
```

Requires a Ray cluster for large-scale training. See the [Ray docs](https://docs.ray.io/en/latest/ray-overview/getting-started.html) for cluster setup.
