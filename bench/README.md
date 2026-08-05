# Benchmarks

Two separate things are measured, so they live in two separate folders with their own results and
README:

- [`functions/`](functions/) — how fast the ported test functions evaluate. Throughput of one
  function against batch size and against dimension, swept on every GPU and CPU of one cluster.
- [`optim/`](optim/) — how fast `vlse.optim.lbfgsb` solves, against scipy's Fortran L-BFGS-B on the
  same objective, over dimension and over a multistart batch.

The results stay separate — `functions/` writes `functions/results/`, `optim/` writes
`optim/results/` — but the machinery around them is one implementation, so a point on one page means
what a point on the other does:

- `sweep.py` — the power-of-two climb, the block sizing, the cache clearing between points, and the
  rules that end a sweep. A bench supplies a size and gets back a call to time.
- `runinfo.py` — the constants table and the locked row append every parallel job writes into.
- `submit.py` — one Slurm job array per GPU chip type the cluster exposes, plus one per CPU core
  count. Takes the bench as its first argument: `uv run python bench/submit.py functions`.
- `plot.py`, `style.py` — the median line with its interval shaded, and the hue per hardware class.
  Every page is written twice: the interactive `.html` and an `.svg` still of the same panel.

Everything is run from the repo root.

Both measure speed. Correctness lives in `tests/`, split the same way — `tests/functions/` against
the R sources, `tests/optim/` against scipy — and `tools/slurm/submit_tests.py` runs that suite on
the GPU chips that have broken a bench run.
