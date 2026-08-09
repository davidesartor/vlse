#!/bin/bash
# One GPU chip type; the array task id picks one solver:dtype:sweep:exponent point from the args.
# Args: <label> <config>... Submitted by `python -m bench optim submit`.
# scipy is absent here on purpose: its L-BFGS-B is a CPU library, and the CPU job runs it.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# node-local env: jobs run concurrently, and a shared venv on NFS gets resynced under them
export UV_PROJECT_ENVIRONMENT="${SLURM_TMPDIR:-/tmp}/vlse-venv-$SLURM_JOB_ID"
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT"' EXIT

# the uv on PATH is the login node's, wrong ELF class on the aarch64 nodes
export PATH="$HOME/.local/bin-$(uname -m):$PATH"
if ! uv --version >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$HOME/.local/bin-$(uname -m)" sh
  hash -r
fi

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# CUDA graphs for the solver's while_loop bodies: ~10% off launch overhead, measured on V100/A100
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,WHILE --xla_gpu_graph_min_graph_size=2"

label="$1"
shift
configs=("$@")
config="${configs[$((SLURM_ARRAY_TASK_ID - 1))]}"
IFS=: read -r solver dtype sweep exponent <<<"$config"
echo "config: $config"

uv run --with "jax[cuda12]" python -m bench optim run \
  --dtype "$dtype" --sweep "$sweep" --solver "$solver" --exponent "$exponent" --label "$label" \
  --repeats "${REPEATS:-12}" --reps "${REPS:-5}" --dim "${DIM:-64}" --batch "${BATCH:-1024}"
