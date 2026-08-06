# Benchmarks

Two things are measured, so there are two benches with their own results and README:

- [`functions/`](functions/) — how fast the ported test functions evaluate.
- [`optim/`](optim/) — how fast `vlse.optim.lbfgsb` solves, against scipy's Fortran L-BFGS-B on the
  same objective.

Both sweep the same two axes — grow the batch at a fixed dimension, grow the dimension at a fixed
batch — on every GPU and CPU of one cluster, and share one entry point, run from the repo root:

```bash
uv run python -m bench <functions|optim> run     [--sweep batch|dim] ...   # one sweep, one row
uv run python -m bench <functions|optim> plot                              # every row, one page
uv run python -m bench <functions|optim> submit  [--dry-run]               # one Slurm array per chip
```

`plot` needs plotly, which is not a project dep:
`uv run --with plotly --with "kaleido==0.2.1" python -m bench functions plot`.

The results stay separate — each bench writes its own `results/` — but the machinery is one
implementation, so a point on one page means what a point on the other does:

- [`sweep.py`](sweep.py) — the power-of-two climb, the block sizing, the cache clearing between
  points and the rules that end a sweep. A bench hands back a call to time at each size.
- [`results.py`](results.py) — the constants table and the locked row append every parallel job
  writes into.
- [`submit.py`](submit.py) — one Slurm job array per GPU chip type the cluster exposes, plus one
  per CPU core count.
- [`plot.py`](plot.py), [`style.py`](style.py) — the median line with its interval shaded, and the
  hue per hardware class. Every page is written twice, an interactive `.html` and an `.svg` still.

A bench itself is one module — [`functions/__init__.py`](functions/__init__.py),
[`optim/__init__.py`](optim/__init__.py) — holding only what is its own: its arguments, the call to
time at each point, and the pages it draws.

Both measure speed. Correctness lives in `tests/`, split the same way.
