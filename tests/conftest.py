import os

# pin BLAS to one thread before numpy/jax import: small-d BLAS calls lose 10x to thread barriers
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
