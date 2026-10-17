#!/usr/bin/env bash
# Train the tree-search PPO agent on structured MQTBench circuits (<=5 qubits, <=100 gates).
#
# Usage (extra args are passed through as Hydra overrides):
#   ./run_structured_training.sh                                  # full run on cuda
#   DEVICE=cpu ./run_structured_training.sh                       # CPU only
#   EXP_NAME=run2 SEED=3 ./run_structured_training.sh algorithm.learning_rate=1e-3
#   ./run_structured_training.sh algorithm.total_timesteps=5_000_000 env.num_envs=16
#
# The values below are STARTING POINTS, not tuned; override via env vars or Hydra args.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate

DEVICE="${DEVICE:-cuda}"                 # cuda | cpu
EXP_NAME="${EXP_NAME:-mqt_structured}"
SEED="${SEED:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"  # starting point
NUM_ENVS="${NUM_ENVS:-8}"                      # starting point (yaml default is 4)
NUM_STEPS="${NUM_STEPS:-128}"                  # steps per env per iteration
MAX_TREE_SIZE="${MAX_TREE_SIZE:-128}"

exec python -u main_tree.py env=mqtbench_small \
  exp_name="$EXP_NAME" seed="$SEED" device="$DEVICE" \
  algorithm.total_timesteps="$TOTAL_TIMESTEPS" algorithm.num_steps="$NUM_STEPS" \
  env.num_envs="$NUM_ENVS" max_tree_size="$MAX_TREE_SIZE" \
  "$@"
