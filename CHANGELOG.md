# Changelog

## Refactoring (2026-05-18)

### High Priority

#### 1. Consolidated `start_tree()` factory function
- Changed `TreePolicy.start_tree()` signature from `(cfg, obs, zx, info)` to `(obs, zx, info, multi_range=1)`, removing the dependency on Hydra config objects in the factory.
- Removed duplicated `start_tree()` definitions from `bqskit_pass.py` and `harris_bench.py` — both now import the canonical version from `TreePolicy`.
- Updated all 6 callers to pass `multi_range` explicitly.

#### 2. Refactored `greedy_random_picking.py`
- Replaced ~250 lines of duplicated `Tree`, `child_to_parents`, `top_down`, `unique_batch`, `synchronous_forward`, `extract_optimal_path`, `show_rules`, `start_tree` code with a 20-line `GreedyTree(Tree)` subclass.
- `GreedyTree` inherits from `TreePolicy.Tree` and overrides only `select_node()` (greedy argmax) and `select()` (no multi-range).
- All shared benchmark utilities imported from `benchmark_utils`.

#### 3. Consolidated benchmark scripts
- Created `benchmark_utils.py` with shared code: `Dataset`, `extract_level`, `deploy_agents`, `generate_N_colors`, `random_CX_circuit`, `get_benchmark_circuits`.
- `graph.py` and `bench_circuits.py` now import from `benchmark_utils` instead of defining their own copies.
- `greedy_random_picking.py` also imports shared utilities from `benchmark_utils`.

#### 4. Archived dead code
- Moved `actionAttention.py`, `Datastore.py`, `test.py`, and `MCTS.py` to `archive/`.
- All four files were unimported by any live code. `MCTS.py` was only used by `test.py` (now archived).

### Medium Priority

#### 5. Fixed typographical errors
- **Filenames**: `grpah_format_converter.py` → `graph_format_converter.py`, `grpah_format_converter_indexAdjusted.py` → `graph_format_converter_index_adjusted.py`. Updated import in `env.py`.
- **Class name**: `DummyActionModle` → `DummyActionModel` in `model.py` and all references.
- **Variable/parameter names**: `initital_circuit_graph` → `initial_circuit_graph` (26 occurrences across 10 files). `state_zx_graph_initital` → `state_zx_graph_initial` (3 occurrences).
- **File `model.py`**: Kept as a deprecation shim with `DeprecationWarning`.

#### 6. Split `model.py` into `models/` package
- Created `models/` package with 9 modules:
  - `slow_norm.py` — `SlowNorm`, `graphnorm`
  - `graph_cross_attention.py` — `GraphCrossAttention`
  - `action_model.py` — `ActionModel` (main PPO agent GNN)
  - `mcts_model.py` — `MCTS_like_model`
  - `action_net.py` — `ActionNet`
  - `tree_net.py` — `TreeNet`
  - `dummy_action_model.py` — `DummyActionModel`
  - `action_embedding_model.py` — `ActionEmbeddingModel`
  - `bundle_net.py` — `BundleNet` (combines `TreeNet` + `DummyActionModel`)
  - `categorical_masked.py` — `CategoricalMasked`
- `models/__init__.py` re-exports all classes for backwards compatibility.
- Updated 8 files to import from `models` instead of `model`.

### Low Priority

#### 7. Ported `CategoricalMasked` entropy fix from Paper 1
- Added `models/categorical_masked.py` with `CategoricalMasked(Categorical)` that correctly zeroes masked entries in entropy computation, avoiding `0 * log(0) = NaN`.
- `ActionModel.get_action_and_value()` now uses `CategoricalMasked` instead of raw `Categorical`.
- Based on Paper 1's fix in `Circopt-RL-ZXCalc/rl-zx/rl_agent.py:13-28`.

#### 8. Replaced `print()` with proper logging
- Added `logging.basicConfig(...)` setup to `main_tree.py`, `ppo.py`, `benchmark_utils.py`, `graph.py`, `bench_circuits.py`, `greedy_random_picking.py`, `bqskit_pass.py`, `harris_bench.py`, `TreePolicy.py`, `env.py`, `multiEnv.py`.
- Converted all `print()` calls in the above files to `logging.info()`, `logging.debug()`, or `logging.warning()` as appropriate.
- Training loops: status updates (`SPS`, `iteration`, `episodic_return`) use `info`. Timing/per-step diagnostics use `debug`.
- Benchmarking: dataset loading, iteration results, and histogram data use `info`.

#### 9. Removed hardcoded model paths
- `bqskit_pass.py:optimizing()` — added `model_path` parameter (defaults to original path). Also respects `ZX_MODEL_PATH` environment variable.
- `harris_bench.py:optimizing()` — added `model_path` parameter (defaults to `model.pkl`). Also respects `ZX_MODEL_PATH` environment variable. Added `argparse` CLI (`--model-path`, `--print-result`).
- Both files now allow the model path to be overridden without editing the source code.
