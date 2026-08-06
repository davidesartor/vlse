#!/bin/bash
# The correctness suite on one GPU chip. Args: <chip>. Submitted by tools/slurm/submit_tests.py.
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

# the suite solves d<=25 from 32 starts, so it has no use for the 75% of the card XLA reserves by
# default; growing on demand leaves the rest of the chip to whatever else the node is running
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# tests/optim by default rather than the whole suite: it is the only half with a GPU mode, and
# the function tests run on the CPU wherever they are. The pyproject default's xdist workers
# share the one card, which a suite this small fits inside of, and that is what puts the
# per-process XLA compiles -- most of the wall time here -- in parallel
uv run --with "jax[cuda12]" --group dev pytest -q ${PYTEST_ARGS:--rf tests/optim}
