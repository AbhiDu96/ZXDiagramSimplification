#!/usr/bin/env bash
# Train the tree-search PPO agent on structured MQTBench circuits (<=5 qubits, <=100 gates).
#
# Usage (extra args are passed through as Hydra overrides):
#   ./run_structured_training.sh                                  # full run on cuda
#   DEVICE=cpu ./run_structured_training.sh                       # CPU only
#   EXP_NAME=run2 SEED=3 ./run_structured_training.sh algorithm.learning_rate=1e-3
#   ./run_structured_training.sh algorithm.total_timesteps=5_000_000 env.num_envs=16
#
# Settings are passed as ENVIRONMENT VARIABLES *before* the script name; anything
# *after* the script name is forwarded verbatim to Hydra. For example:
#   DEVICE=cpu EXP_NAME=run2 ./run_structured_training.sh algorithm.learning_rate=1e-3
# (writing `./run_structured_training.sh DEVICE=cpu` instead passes DEVICE=cpu to
#  Hydra, which is not what you want.)
#
# The values below are STARTING POINTS, not tuned; override via env vars or Hydra args.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Resolve the project environment ourselves rather than relying on a pre-activated
# venv (which may be named/located differently, e.g. a custom UV_PROJECT_ENVIRONMENT,
# and was the source of "ModuleNotFoundError: No module named 'hydra'").
VENV_PATH="${UV_PROJECT_ENVIRONMENT:-.venv}"
MQT_SPEC="${MQT_SPEC:-mqt-bench>=2.3,<3}"

# 1) Make sure the environment exists with all core deps (idempotent).
uv sync
# 2) MQTBench circuit source. Install it INTO the project venv (not a `uv run --with`
#    overlay): Ray spawns worker processes that run this same venv python, so every
#    worker -- not just the driver -- must be able to `import mqt.bench`.
#    The explicit "numpy<2" is CRITICAL: `uv pip install` is not bound by the project's
#    numpy<2 pin and will otherwise pull numpy>=2, which silently breaks zx_env (PyZX
#    0.8.0 / the `int(np.where(...))` indexing relies on numpy 1.x scalar semantics).
#    mqt-bench 2.3.0 is compatible with numpy 1.x on Python 3.11 (pinned by .python-version).
uv pip install --python "$VENV_PATH/bin/python" "$MQT_SPEC" "numpy<2"

DEVICE="${DEVICE:-cuda}"                 # cuda | cpu
EXP_NAME="${EXP_NAME:-mqt_structured}"
SEED="${SEED:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"  # starting point
NUM_ENVS="${NUM_ENVS:-8}"                      # starting point (yaml default is 4)
NUM_STEPS="${NUM_STEPS:-128}"                  # steps per env per iteration
MAX_TREE_SIZE="${MAX_TREE_SIZE:-128}"

# 3) Launch with the venv python DIRECTLY -- NOT `uv run`. Ray detects `uv run` and
#    tries to recreate the environment inside each worker from a temporary working
#    dir, where the editable `../pyzx_environment` path no longer resolves, so the
#    workers die ("Failed to startup worker after retrying 5 times").
exec "$VENV_PATH/bin/python" -u main_tree.py env=mqtbench_small \
  exp_name="$EXP_NAME" seed="$SEED" device="$DEVICE" \
  algorithm.total_timesteps="$TOTAL_TIMESTEPS" algorithm.num_steps="$NUM_STEPS" \
  env.num_envs="$NUM_ENVS" max_tree_size="$MAX_TREE_SIZE" \
  "$@"
