#!/bin/bash
# One solver:sweep:exponent point per array task, picked from the args by task id. jax shards the
# batch across the allocated cores, one serial `lax.map` each; scipy's Fortran is serial whatever
# the width, so it only appears in the single-core job's configs. Args: <config>...
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR:-/tmp}/vlse-venv-$SLURM_JOB_ID"
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT"' EXIT
# the job holds the whole node, so `taskset` rather than the allocation is what fixes the width;
# the low cpu ids are one socket's cores, so every width here stays on one socket's memory
cores="${SLURM_CPUS_PER_TASK:-1}"

configs=("$@")
config="${configs[$((SLURM_ARRAY_TASK_ID - 1))]}"
solver="${config%%:*}"
rest="${config#*:}"
sweep="${rest%%:*}"
exponent="${rest##*:}"
echo "config: $config"

common="--sweep $sweep --exponent $exponent --repeats ${REPEATS:-12} --reps ${REPS:-5} \
  --dim ${DIM:-64} --batch ${BATCH:-1024} --max-seconds ${MAX_SECONDS:-10}"

if [ "$cores" = 1 ]; then
  taskset -c 0 uv run python -m bench optim run \
    --solver "$solver" --label "$solver-cpu-1core" --threads 1 $common
else
  taskset -c "0-$((cores - 1))" uv run python -m bench optim run \
    --solver jax --label "jax-cpu-${cores}core" --threads 1 --shards "$cores" $common
fi
