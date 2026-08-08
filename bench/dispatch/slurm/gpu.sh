#!/bin/bash
# One GPU chip type, one task: the latency has no axis. Args: <label> <config>...
# Submitted by `python -m bench dispatch submit`.
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

label="$1"

# one point that every other curve is corrected by, run once: worth a far larger sample than a
# sweep point, and a call this cheap means the whole job is still minutes
uv run --with "jax[cuda12]" python -m bench dispatch run \
  --label "$label" --repeats "${REPEATS:-500}" --reps "${REPS:-11}"
