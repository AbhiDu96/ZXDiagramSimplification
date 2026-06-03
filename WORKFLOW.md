# ZX Diagram Simplification — Complete Workflow

## 1. The Problem

A quantum circuit (e.g., CNOT + T + H gates) is converted into a **ZX-diagram** — a graph where:
- **Nodes** = Z-spiders or X-spiders (colored vertices with phase angles)
- **Edges** = wires connecting spiders

ZX-calculus has provably correct **rewrite rules** that simplify the diagram while preserving the circuit's semantics. The goal: apply rules in a smart order to minimize the two-qubit (CNOT) gate count after converting back to a circuit.

---

## 2. Input: Circuit → ZX Graph

**Entry point**: `pyzx_environment/zx_env/env.py` — the `zx_env` class

```
Random circuit (n_qubits, depth, gate ratios)
       ↓  zx.Circuit.to_graph()
ZX graph (PyZX internal format)
       ↓  pyzy_to_homogeneous_torchData()
PyTorch Geometric Data object
  - x: node features (8-dim float tensor)
  - edge_index: connectivity
  - edge_attr: edge type (Hadamard vs regular)
```

**The 8 node features** (`mk_features` in `env.py:142`):
```
[gates, tcount, clifford, twoqubit, had, depth, depth_cz, edges]
```
All normalized by expected circuit size. This is a graph-level summary vector attached to every node (not per-node features from the ZX graph structure).

**Mutation at reset** (`env.py:210`): Before training begins, the env randomly applies ZX rules to the initial circuit to create variation in starting states — this forces the agent to learn from diverse positions, not just "clean" graphs.

**Baseline computed at reset**:
- `baseline_cnot_count` = CNOTs before any RL simplification
- `pyzx_cnot_count` = CNOTs after PyZX's greedy `full_reduce` (the target to beat)

---

## 3. Observation: `GraphMask` + Wrappers

The raw environment observation goes through a wrapper stack defined in `ppo.py:make_env`:

```
zx_env (raw)
   ↓ [optional] RewardTransform  — sparsify rewards (only reward every N steps)
   ↓ [optional] GraphMakeDirected — strip symmetrized edges, keep directed only
   ↓ GraphMaskWrapper            — converts everything to a GraphMask
```

**`GraphMaskWrapper`** (`utils.py:67`) does three things:
1. Transposes action masks from `(n_rules, n_nodes)` → `(n_nodes, n_rules)`
2. Calls `expand_graph()` — promotes every **edge to a virtual node**, so both nodes and edges are first-class graph citizens in the GNN
3. Calls `ToUndirected()` — symmetrizes the graph

**Output is a `GraphMask` dataclass** (`utils.py:14`):
```python
@dataclass
class GraphMask:
    Graph: data.Data       # PyG Data with expanded virtual-edge nodes
    action_mask: Tensor    # shape (n_nodes, n_rules) — 1 where rule is applicable
    state_zx_graph: Any    # raw PyZX graph clone (for rule application)
```

The `action_mask` is critical: it tells the model which `(node, rule)` pairs are actually applicable at the current state.

---

## 4. The Model Architecture: `BundleNet`

`BundleNet` (`models/bundle_net.py`) has two sub-networks:

```
BundleNet
├── TreeNet          — scores nodes in the search tree (priority + value)
└── DummyActionModel — uniform action scores (tree handles selection)
```

### 4a. `TreeNet` (`models/tree_net.py`)

Takes a **batch of tree node feature vectors** (8-dim, one per tree node):

```
Input: [n_tree_nodes, 8]
  ↓ SlowNorm(8)            — running mean/std normalization
  ↓ Linear(8 → hidden_dim) + LeakyReLU
  ↓ 2× [LayerNorm → Linear(hidden → 2*hidden) → GLU] + residual
  ↓ priority_prediction head → [n_tree_nodes]   (scalar per node)
  ↓ value_prediction head   → [n_tree_nodes]   (scalar per node)
```

`SlowNorm` (`models/slow_norm.py`): exponential moving average of mean/std during training. During eval it uses the stored statistics — no distribution shift between train and inference (unlike BatchNorm).

### 4b. `DummyActionModel` (`models/dummy_action_model.py`)

Returns **all-ones logits** for positions and actions — uniform distribution. The tree search itself (not the action model) drives which action to pick. This is intentional: `TreeNet` does the heavy lifting of node selection; action selection within a node is uniform.

### 4c. `ActionModel` + `GraphCrossAttention` (used in `ppo.py`, flat PPO variant)

This is the full GNN agent for the flat-PPO variant:

```
Input: PyG batch of expanded ZX graphs
  ↓ action_embedder: Embedding(n_rules+1, action_dim)
      — each node gets an action-indexed embedding: shape (n_nodes, n_rules, action_dim)
  ↓ message_passing_loop (n_message_passing rounds):
      each round:
        action_masking — zero out inapplicable action embeddings
        GraphCrossAttention pass — propagate info across edges
        mlp (Linear + LeakyReLU + LayerNorm) + residual
  ↓ policy_projection: Linear(action_dim, 1) per (node, action) → logits [n_nodes, n_rules]
  ↓ value_projection:  Linear(action_dim, 1) per (node, action) → values [n_nodes, n_rules]
```

**`GraphCrossAttention`** (`models/graph_cross_attention.py`):
- Keys = neighbor node mean-pooled embeddings projected through `nn.Linear`
- Queries = per-action embeddings at each node projected through `nn.Linear`
- Attention weight α = exp(LeakyReLU(key + query)) — one scalar per edge per action
- Values = neighbor embeddings; aggregated with `SimpleConv` weighted by α
- Uses `torch.vmap` to efficiently vectorize over the action dimension

---

## 5. Tree Search: `Tree` + `TreePolicy`

This is the core algorithmic contribution of Paper 2. The tree maintains **multiple candidate rewrite sequences** simultaneously.

### Tree Structure (`TreePolicy.py:Tree`)

```python
Tree:
  nodes      = [PyG graph for each tree node]    # graph state at that node
  zx_states  = [PyZX graph clone at each node]   # for rule application
  children   = [[child_idx, ...] per node]       # tree adjacency
  depths     = [int per node]                    # depth from root
  rewards    = [float per node]                  # reward at that state
  infos      = [dict per node]                   # metadata (features, applied rule)
```

Root = initial circuit. Each expansion creates child nodes by applying rewrite rules.

### `Tree.select()` — choosing what to expand

```
1. select_node(TreeNet):
   - Forward all nodes through TreeNet → priorities [n_nodes], values [n_nodes]
   - top_down(): propagate priority scores down the tree
     (parent score accumulates into children → rewards exploring subtrees near good parents)
   - Sample from Categorical(logits = top_down_priorities)
   → selected_node_index, log_prob, entropy, logsumexp(values)

2. select_expansion(DummyActionModel):
   - Returns uniform logits over (node_positions × n_rules)
   - Sample action from Categorical masked by action_mask
   → action_index (encodes which node + which rule)
```

### `Tree.expand()` — applying rules

```
For each rule in multi_range:
  1. Decode action_index → (position, rule_index) via torch.unravel_index
  2. simulator.step(position, action, pyzx_state=zx_states[selected_node])
     → new ZX graph, reward, done
  3. Append new node to tree as child of selected_node
```

**`multi_range`** = number of children created per expansion step. With `multi_range=4`, each expand call creates 4 children from the selected node, each applying a different action.

### `synchronous_forward()` (`TreePolicy.py:70`) — the batched training pass

During PPO update, instead of running each tree node through the GNN separately, all tree nodes across all parallel trees are batched into one forward pass. It uses `unique_batch()` to deduplicate identical tree node states (saves computation since repeated states appear in many trees).

---

## 6. Training Loop: `main_tree.py`

```
Initialization:
  - Create num_envs environments
  - Create validation_circuits (100 fixed circuits for evaluation)
  - Create BundleNet agent

For each iteration:
  ┌─────────────── Rollout Collection (Ray-parallel) ───────────────┐
  │  For each env i in parallel (@ray.remote deploy_train):         │
  │    env.reset() → initial GraphMask                              │
  │    start_tree() → Tree with root = initial state                │
  │    For num_steps steps:                                         │
  │      tree.select(agent) → action (node_idx, rule_indices)       │
  │      tree.expand(action, env) → new_tree, reward, done          │
  │      if step % max_tree_size == 0: restart_tree()               │
  │        (prune tree, restart from best node found so far)        │
  │      store: obs[t], actions[t], logprobs[t], rewards[t], ...    │
  └─────────────────────────────────────────────────────────────────┘

  Bootstrap last value (GAE):
    next_value via tree.select(agent)
    Compute advantages with GAE-λ (γ=0.99, λ=0.95)
    returns = advantages + values

  ┌─────────────── PPO Update ──────────────────────────────────────┐
  │  For update_epochs epochs:                                      │
  │    Shuffle batch indices                                        │
  │    For each minibatch:                                          │
  │      synchronous_forward(trees_in_minibatch, agent, actions)    │
  │        → cache, newlogprob, entropy, newvalue                   │
  │      ratio = exp(newlogprob - oldlogprob)                       │
  │      pg_loss = clipped surrogate loss (PPO-clip)                │
  │      v_loss = clipped value loss                                │
  │      entropy_loss = policy entropy (encourages exploration)     │
  │      loss = pg_loss - ent_coef*entropy + vf_coef*v_loss        │
  │      optimizer.step() with grad clipping                        │
  └─────────────────────────────────────────────────────────────────┘

  Every 50 iterations: validation()
```

**`restart_tree`**: When the tree grows past `max_tree_size` nodes, it is pruned — a new tree starts rooted at the best state found so far. This keeps memory bounded while preserving the best solution.

---

## 7. Reward Signal

Defined in `pyzx_environment/zx_env/general_utils/reward_functions.py`, selected by config.

Default (`normalized_cnot_count_reward`):
```
reward = (baseline_cnot_count - current_cnot_count) / baseline_cnot_count
```

The agent receives positive reward proportional to how many CNOTs it removes compared to the original circuit. The tree stores the **cumulative best reward** and uses it to decide which branch to continue exploring.

---

## 8. Validation & Output

During validation (`main_tree.py:validation`):
1. Run `deploy_agents` for each validation circuit
2. Call `tree.get_best_node()` → the ZX state with the highest cumulative reward
3. Call `extract_circuit(best_zx_state)` → convert back to a quantum circuit
4. Measure `.stats_dict()["twoqubit"]` = final CNOT count
5. Log to TensorBoard: mean/max/min CNOT count, validation return

**`extract_optimal_path(tree)`** traces back from the best leaf to the root, recovering the exact sequence of rewrite rules applied.

Checkpoints saved as:
- `runs/{run_name}/saves/model-{step}.pth` — model weights
- `runs/{run_name}/saves/data-{step}.pkl` — optimal rewrite paths

---

## 9. Inference / Benchmarking

The trained model plugs into BQSKit compiler pipelines via `bqskit_pass.py`. BQSKit partitions a circuit into subcircuits, and the RL agent acts as a **peephole optimizer** — it optimizes each subcircuit independently via the tree search, then the results are reassembled.

`bench_compilers.py` provides CLI benchmarking against other compilers (PyZX, Qiskit, etc.).

---

## Summary Diagram

```
Quantum Circuit (QASM / random generation)
         │
         ▼
   PyZX ZX-Graph ──── baseline_cnot_count computed
         │                (PyZX full_reduce for comparison)
         ▼
  [Mutation: random ZX rules applied for curriculum diversity]
         │
         ▼
  GraphMask (PyG Data + action_mask + zx_clone)
    - expand_graph(): edges promoted to virtual nodes
    - ToUndirected(): symmetrize
         │
         ▼
  Tree (search structure)
  ┌─────────────────────────────┐
  │  Root = initial ZX graph    │
  │  Children = rule applications│
  │  Branching factor = multi_range│
  └─────────────────────────────┘
         │
         ▼
  BundleNet (GNN Agent)
  ├── TreeNet: score tree nodes (priority + value) via MLP on 8-dim features
  └── DummyActionModel: uniform action scores (tree handles selection)
         │
         ▼
  PPO Training (Ray-parallel rollouts)
  - GAE advantage estimation
  - Clipped surrogate objective
  - Entropy bonus for exploration
         │
         ▼
  Best Tree Node → extract_circuit() → Optimized Quantum Circuit
  (fewer CNOT gates than baseline / PyZX full_reduce)
```

---

## Key Insight: The Split Architecture

`TreeNet` scores *which tree node* to expand next — a graph-level decision based on the circuit's current 8-dim summary features. `DummyActionModel` then picks *which rule to apply*, but since it returns uniform logits, randomness in action selection is what creates diverse children.

All GNN learning is concentrated in `TreeNet`: it learns which intermediate states are worth exploring further, effectively learning a value function over the rewrite search space.
