
# ZXDiagramSimplification

Code for the paper:
"Optimizing Quantum Circuits via ZX Diagrams using Reinforcement Learning and Graph Neural Networks"

This repository implements the Appendix C setting (compact 8-feature representation), including:
- Tree-search training with PPO (`main_tree.py`)
- Flat PPO baseline (`ppo.py`)
- BQSKit integration (`zx_compiler.py`)
- Compiler benchmarking (`bench_compilers.py`)

## 1. How the pipeline works

High-level workflow (see `WORKFLOW.md` for the full walkthrough):

1. Circuit -> ZX graph (`pyzx_environment/zx_env/env.py`)
2. Observation wrapping -> `GraphMask` (`utils.py`):
	 - Expanded graph representation
	 - Action mask of applicable ZX rewrite rules per node
3. Tree search over rewrite trajectories (`TreePolicy.py`)
4. Agent scoring via `BundleNet` (`models/bundle_net.py`)
5. PPO optimization (`main_tree.py` or `ppo.py`)
6. Best ZX state -> extracted optimized circuit

The main optimization target is reducing two-qubit gate count (CNOT/CZ).

## 2. Setup (with `uv`)

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). `uv` reads
`pyproject.toml` + `uv.lock` and creates a reproducible virtual environment for you — you
do **not** need to create a venv or `pip install` anything by hand.

### 2.1 Prerequisite: clone both repos side by side

This project depends on the RL environment package [`pyzx_environment`](https://github.com/AbhiDu96/pyzx_environment)
as an **editable local dependency at the relative path `../pyzx_environment`** (see
`[tool.uv.sources]` in `pyproject.toml`). The two repositories must therefore be cloned as
**siblings, keeping their default directory names**:

```bash
# pick any parent directory, then clone both into it
git clone https://github.com/AbhiDu96/ZXDiagramSimplification.git
git clone https://github.com/AbhiDu96/pyzx_environment.git

# resulting layout (names matter — `uv` looks for ../pyzx_environment):
# <parent>/
# ├── ZXDiagramSimplification/
# └── pyzx_environment/
```

If the two directories are not siblings with these exact names, `uv sync` fails to find
`zx_env`.

### 2.2 Install `uv` (once per machine)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # macOS / Linux
# then restart the shell, or: source $HOME/.local/bin/env
```

### 2.3 Create the environment

```bash
cd ZXDiagramSimplification
uv sync
```

This creates `./.venv` (Python 3.11, pinned by `.python-version`) and installs the core
training dependencies (torch, torch-geometric, hydra-core, gymnasium, ray, numpy<2, ...)
plus the local `zx_env` package in editable mode.

> **Note on numpy:** `zx_env` pins `numpy<2` (PyZX 0.8.0 relies on numpy 1.x semantics).
> This means the stack supports **Python 3.11 / 3.12** — numpy 1.x has no wheels for 3.13+.

### 2.4 Running commands

For **non-Ray** commands (tests, TensorBoard) the simplest launcher is `uv run`, which
auto-selects the project `.venv`:

```bash
uv run pytest
uv run tensorboard --logdir runs
```

> ⚠️ **Do NOT launch `main_tree.py` (or `bench_compilers.py`) with `uv run`.** They use
> [Ray](https://docs.ray.io/), and Ray detects the `uv run` wrapper and tries to recreate
> the environment inside every worker process from a temporary working directory — where the
> editable `../pyzx_environment` path no longer resolves, so the workers die with
> *"Failed to startup worker after retrying 5 times"*. (`ppo.py` does not use Ray, but for
> consistency the examples launch it the same way.)

Launch training with the venv's Python **directly** instead (both forms work and Ray workers
inherit the environment cleanly):

```bash
source .venv/bin/activate            # then use `python ...`
# — or call the interpreter by path without activating:
.venv/bin/python -u main_tree.py ...
```

The examples below use `.venv/bin/python`; if you created the env at a custom location
(`UV_PROJECT_ENVIRONMENT`), substitute that path.

### 2.5 Optional dependency groups

```bash
uv sync --extra benchmark    # bqskit, qiskit, qiskit-ibm-transpiler, pandas, seaborn, matplotlib
uv sync --extra analysis     # pytorch-metric-learning, shap, matplotlib (SHAP/graph-similarity scripts)
```

`GraphSimilarity.py` / `data_generation.py` additionally need `torch-scatter`, which must be
built against your exact torch version:

```bash
uv pip install torch-scatter --no-build-isolation
```

## 3. Running training

All training entry points are [Hydra](https://hydra.cc/)-based. Config groups live in `conf/`
(`env/`, `model/`, `algorithm/`, `validation/`) and are selected with `group=value`.

> **Important — do not prefix group selections with `+`.** `conf/config.yaml` already
> declares defaults for `env`, `model`, `algorithm`, and `validation`. Use `env=simple`
> (override), **not** `+env=simple` (append) — the `+` form raises
> `hydra.errors.ConfigCompositionException: Multiple values for <group>`.
> Use `+` only to add a brand-new key that is not already in the config.

### 3.1 Quick smoke test (tree-search training)

```bash
.venv/bin/python -u main_tree.py \
	algorithm=PPO \
	model=GATActionModel \
	env=simple \
	exp_name="smoke_test" \
	device="cpu" \
	env.num_envs=2 \
	algorithm.total_timesteps=50000 \
	algorithm.num_steps=32 \
	max_tree_size=64 \
	multi_range=2
```

### 3.2 Main training command (paper-style tree-search run)

```bash
.venv/bin/python -u main_tree.py \
	algorithm=PPO \
	model=GATActionModel \
	env=more_complex_more_rules_ranges \
	exp_name="20MIO_32_envs_5qubit_128treesize" \
	device="cpu" \
	env.num_envs=32 \
	env.n_qubits=5 \
	model.model_type="ActionAtt" \
	model.n_message_passing=4 \
	algorithm.total_timesteps=20_000_000 \
	algorithm.num_steps=129 \
	algorithm.learning_rate=3e-3 \
	max_tree_size=128 \
	multi_range=4
```

### 3.3 Flat PPO variant

```bash
.venv/bin/python -u ppo.py \
	algorithm=PPO \
	model=GATActionModel \
	env=more_complex_more_rules_ranges \
	exp_name="flat_ppo_run" \
	device="cpu"
```

### 3.4 Structured training on MQTBench circuits

`run_structured_training.sh` trains the tree-search agent on small structured circuits
sampled from [MQTBench](https://www.cda.cit.tum.de/mqtbench/) (`env=mqtbench_small`,
≤5 qubits, ≤100 gates). The script is self-contained: it runs `uv sync`, installs the
`mqt-bench` circuit generator into the project venv (so Ray workers can import it too; the
MQTBench pool is generated and cached on first use), and then launches training with the
venv's Python directly. You do **not** need to pre-install `mqt-bench` or activate anything.
`mqt-bench` requires Python 3.11+, which `.python-version` already pins; `numpy<2` is kept.

```bash
# full run on GPU (defaults)
./run_structured_training.sh

# CPU, custom name/seed/length — settings go BEFORE the script name as env vars;
# anything AFTER the script name is forwarded verbatim to Hydra:
DEVICE=cpu EXP_NAME=mqt_test SEED=44 TOTAL_TIMESTEPS=10000 NUM_ENVS=8 \
	./run_structured_training.sh algorithm.learning_rate=1e-3
```

Overridable environment variables (with their defaults): `DEVICE=cuda`,
`EXP_NAME=mqt_structured`, `SEED=1`, `TOTAL_TIMESTEPS=2000000`, `NUM_ENVS=8`,
`NUM_STEPS=128`, `MAX_TREE_SIZE=128`, and `MQT_SPEC=mqt-bench>=2.3,<3`.

## 4. Tests

```bash
uv run pytest
```

The suite (`tests/`) checks Hydra config composition, env/model construction, and the public
import surface (`zx_env`, `models`). All tests should pass on a fresh `uv sync`.

## 5. Ray parallelism notes

- Large-scale runs use [Ray](https://docs.ray.io/) for distributed rollouts.
- For multi-node deployment, configure your Ray cluster before launching training.

## 6. Checkpoints and logs

Training outputs are written under `runs/`.

- TensorBoard logs: `runs/<run_name>/`
- Saved models: `runs/<run_name>/saves/model-<step>.pth`
- Saved optimal paths: `runs/<run_name>/saves/data-<step>.pkl`

Launch TensorBoard with:

```bash
uv run tensorboard --logdir runs
```

## 7. Benchmarking and inference

### 7.1 Benchmark compilers

`bench_compilers.py` CLI (needs `uv sync --extra benchmark`). It uses Ray, so launch it with
the venv Python directly (see the §2.4 warning), not `uv run`:

```bash
.venv/bin/python bench_compilers.py <output_pickle> <searchdepth> <mq_ratio> <h_ratio> <t_ratio>
```

Example:

```bash
.venv/bin/python bench_compilers.py results.pkl 4 1.0 0.0 0.0
```

Notes:
- Some benchmark options depend on external optimizer code that is not distributed in this repository.
- If you do not use those external methods, disable/comment the corresponding benchmark paths.

### 7.2 Use a trained model in BQSKit flows

`zx_compiler.py` loads a model path from `ZX_MODEL_PATH` (or falls back to a default path).

```bash
export ZX_MODEL_PATH="runs/<run_name>/saves/model-<step>.pth"
```

Then run your BQSKit-based workflow (for examples, see `bench_compilers.py` and `zx_compiler.py`).

## 8. Useful files

- `main_tree.py`: Tree-search PPO training loop
- `ppo.py`: Flat PPO baseline
- `TreePolicy.py`: Tree data structure and batched policy/value forward pass
- `models/`: Model definitions (`BundleNet`, `TreeNet`, `ActionModel`, etc.)
- `utils.py`: Observation wrappers and `GraphMask` handling
- `run_structured_training.sh`: One-line launcher for MQTBench structured training
- `pyzx_environment/`: ZX RL environment package (cloned as a sibling, see §2.1)
- `WORKFLOW.md`: Full end-to-end algorithmic walkthrough
