#!/bin/bash
# One CPU width, one task: the latency has no axis. The job holds the whole node, so `taskset`
# rather than the allocation is what fixes the width. Args: <config>...
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR:-/tmp}/vlse-venv-$SLURM_JOB_ID"
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT"' EXIT
# the low cpu ids are one socket's cores, so every width here stays on one socket's memory
cores="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$cores" OPENBLAS_NUM_THREADS="$cores" MKL_NUM_THREADS="$cores"

# one point that every other curve is corrected by, run once: worth a far larger sample than a
# sweep point, and a call this cheap means the whole job is still minutes
taskset -c "0-$((cores - 1))" \
  uv run python -m bench dispatch run \
  --label "cpu-${cores}core" --repeats "${REPEATS:-500}" --reps "${REPS:-11}"
