#!/bin/bash
# One dtype:sweep:exponent point per array task, picked from the args by task id. The job holds the
# whole node, so `taskset` rather than the allocation is what fixes the width -- XLA sizes its
# thread pool from the affinity mask. Args: <config>...
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR:-/tmp}/vlse-venv-$SLURM_JOB_ID"
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT"' EXIT
# the low cpu ids are one socket's cores, so every width here stays on one socket's memory
cores="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$cores" OPENBLAS_NUM_THREADS="$cores" MKL_NUM_THREADS="$cores"

configs=("$@")
config="${configs[$((SLURM_ARRAY_TASK_ID - 1))]}"
dtype="${config%%:*}"
rest="${config#*:}"
sweep="${rest%%:*}"
exponent="${rest##*:}"
echo "config: $config"

common="--sweep $sweep --dtype $dtype --exponent $exponent --repeats ${REPEATS:-12} \
  --reps ${REPS:-5} --dim ${DIM:-64} --batch ${BATCH:-1024} --max-seconds ${MAX_SECONDS:-1}"

taskset -c "0-$((cores - 1))" \
  uv run python -m bench functions run --label "cpu-${cores}core" $common
