
# ZXDiagramSimplification

Code for the paper:
"Optimizing Quantum Circuits via ZX Diagrams using Reinforcement Learning and Graph Neural Networks"

This repository implements the Appendix C setting (compact 8-feature representation), including:
- Tree-search training with PPO (`main_tree.py`)
- Flat PPO baseline (`ppo.py`)
- BQSKit integration (`bqskit_pass.py`)
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

## 2. Setup

### 2.1 Python environment

Use Python 3.10+ (recommended) and create a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2.2 Install dependencies

Install core dependencies (adjust CUDA-specific Torch wheels as needed):

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install hydra-core omegaconf gymnasium numpy tqdm tensorboard ray pyzx
```

Install the local ZX environment package:

```bash
pip install -e ./pyzx_environment
```

Optional dependencies for benchmarking / compiler integration:

```bash
pip install bqskit qiskit qiskit-ibm-transpiler pandas seaborn matplotlib
```

## 3. Running training

All training entry points are Hydra-based and accept config group overrides from `conf/`.

### 3.1 Quick smoke test (tree-search training)

```bash
python -u main_tree.py \
	+algorithm=PPO \
	+model=GATActionModel \
	+env=simple \
	exp_name="smoke_test" \
	env.num_envs=2 \
	algorithm.total_timesteps=50000 \
	algorithm.num_steps=32 \
	max_tree_size=64 \
	multi_range=2 \
	device="cpu"
```

### 3.2 Main training command (paper-style tree-search run)

```bash
python -u main_tree.py \
	+algorithm=PPO \
	exp_name="20MIO_32_envs_5qubit_128treesize" \
	+model=GATActionModel \
	+env=more_complex_more_rules_ranges \
	env.num_envs=32 \
	model.model_type="ActionAtt" \
	model.n_message_passing=4 \
	algorithm.total_timesteps=20_000_000 \
	algorithm.num_steps=129 \
	algorithm.learning_rate=3e-3 \
	max_tree_size=128 \
	multi_range=4 \
	env.n_qubits=5 \
	device="cpu"
```

### 3.3 Flat PPO variant

```bash
python -u ppo.py \
	+algorithm=PPO \
	+model=GATActionModel \
	+env=more_complex_more_rules_ranges \
	exp_name="flat_ppo_run"
```

## 4. Ray parallelism notes

- Large-scale runs are intended to use Ray for distributed rollouts.
- For multi-node deployment, configure your Ray cluster before launching training.
- Reference: https://docs.ray.io/en/latest/ray-overview/getting-started.html

## 5. Checkpoints and logs

Training outputs are written under `runs/`.

- TensorBoard logs: `runs/<run_name>/`
- Saved models: `runs/<run_name>/saves/model-<step>.pth`
- Saved optimal paths: `runs/<run_name>/saves/data-<step>.pkl`

Launch TensorBoard with:

```bash
tensorboard --logdir runs
```

## 6. Benchmarking and inference

### 6.1 Benchmark compilers

`bench_compilers.py` CLI:

```bash
python bench_compilers.py <output_pickle> <searchdepth> <mq_ratio> <h_ratio> <t_ratio>
```

Example:

```bash
python bench_compilers.py results.pkl 4 1.0 0.0 0.0
```

Notes:
- Some benchmark options depend on external optimizer code that is not distributed in this repository.
- If you do not use those external methods, disable/comment the corresponding benchmark paths.

### 6.2 Use a trained model in BQSKit flows

`bqskit_pass.py` loads a model path from `ZX_MODEL_PATH` (or falls back to a default path).

```bash
export ZX_MODEL_PATH="runs/<run_name>/saves/model-<step>.pth"
```

Then run your BQSKit-based workflow (for examples, see `bench_compilers.py` and `bqskit_pass.py`).

## 7. Useful files

- `main_tree.py`: Tree-search PPO training loop
- `ppo.py`: Flat PPO baseline
- `TreePolicy.py`: Tree data structure and batched policy/value forward pass
- `models/`: Model definitions (`BundleNet`, `TreeNet`, `ActionModel`, etc.)
- `utils.py`: Observation wrappers and `GraphMask` handling
- `pyzx_environment/`: ZX RL environment package
- `WORKFLOW.md`: Full end-to-end algorithmic walkthrough
