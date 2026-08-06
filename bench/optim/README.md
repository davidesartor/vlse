# L-BFGS-B benchmarks

How fast `vlse.optim.lbfgsb` solves, across the GPUs and CPUs of one cluster, against scipy's Fortran
L-BFGS-B on the same objective. The same two axes as the function bench:

- **batch sweep** — grow a multistart batch at a fixed dimension: [fp64](scaling-batch-fp64.html)
- **dim sweep** — grow the dimension at a fixed batch: [fp64](scaling-dim-fp64.html)

Speed only. Whether the two solvers land in the same place is `tests/optim/`, which runs all 48
functions in both variants against the same scipy; nothing here is a correctness claim.

## Running it

```bash
uv run python -m bench optim run --sweep batch --solver jax
uv run python -m bench optim run --sweep dim --solver scipy --threads 1
uv run --group bench python -m bench optim plot
uv run python -m bench optim submit --dry-run
```

One CPU job per width, `--cores` defaulting to `1,2,4,8,16`: each jax job shards its batch across
its cores, and scipy — serial whatever the width — runs only in the single-core job. Every width
lands on the same node (`--cpu-node`, defaulting to an idle node carrying `--cpu-feature`) and
takes it exclusively, so the widths differ by width and nothing else; the script `taskset`s to
cpus `0..cores-1`, since the allocation no longer bounds them. Each array task is one point — one solver, sweep and power-of-two size, passed as a
`solver:sweep:exponent` script argument the task id picks from — and runs `--repeats` sequential
repeats of it. The axis runs 2**0 through 2**24 regardless of device (a batch smaller than the
shard count is skipped rather than submitted, since it cannot split); most exponents OOM or TIMEOUT
long before the top, which costs that one array task and nothing else. `REPEATS`, `REPS`, `DIM`,
`BATCH` and `MAX_SECONDS` override the job scripts' defaults.

## What is measured

A point is a solve, not an evaluation. The timed call runs L-BFGS-B from a random start in the box
until the projected gradient is under `--tol` (1e-9), so what the axis buys is solves per second.
Everything around that call is the function bench's — same block sizing, same axis climbed one array
task per size — so the two sets of curves can be read against each other. The CPU ceiling is
`--max-seconds 10` rather than the function bench's 1, because at a second a call the axis would be
three points long.

`--solver jax` puts the whole batch under one dispatch. `--solver scipy` runs the starts one after
another, which is the only way scipy has, against the same jax objective through a host round trip.
scipy is CPU only and is drawn dashed on every page: it is the baseline, not another device.

The single-core sweeps are pinned, `taskset` to one core plus `--threads 1` — XLA sizes its pool
from the affinity mask, so `taskset` narrows it and `--threads` caps scipy's BLAS beside it, leaving
the two compared at the same width. The wider CPU jobs keep `--threads 1` and instead shard:
`--shards N` forces N XLA host devices (`--xla_force_host_platform_device_count`) and `shard_map`s
the batch over them, one serial `lax.map` per core.

### Why a solve is serial on CPU, and `lax.map` rather than `vmap`

`vmap` is a rewrite, not a scheduler: it puts a batch axis on every op, and whether the batched op
then threads is the backend's choice. On GPU each op becomes one kernel over the whole batch, which
is why the GPU curves climb with the batch. On CPU the batched `jnp.linalg.inv` and
`jnp.linalg.solve` in `_limited_memory_matrices` and `_subspace_minimum` lower to LAPACK custom
calls whose handlers walk the batch serially on the calling thread, on matrices that are 2m×2m =
20×20 whatever the dimension — too small for LAPACK to thread internally. Those calls are most of
the solve, they sit inside one `lax.while_loop`, and XLA:CPU splits work within a fusion rather than
across loop iterations. Measured at batch 256, `d=20`: 1703 solves/s on 1 core against 1232 on 8,
with only 1.4 cores busy.

Since it runs serially either way, a batch axis on CPU buys nothing and costs the straggler: a
vmapped batch runs every start to the slowest one's trip count, while a mapped one lets each start
stop when it converges. One core, `d=20`: 1291 solves/s vmapped at batch 16 against 957 at 1024,
where `lax.map` holds 1248 and 1277. At `d=64` it is 519 against 750 at batch 1024. `lax.scan`
measures the same as `lax.map`, which is what it lowers to.

What does scale is sharding across solves: independent chunks on separate host devices run on
separate cores, `check_vma` off because the check rejects the solver's `while_loop` carries and no
shard communicates. Four cores, `d=64`: 3002 solves/s at batch 1024 against 910 on one core, 83%
scaling efficiency. The small end pays the straggler inside each chunk — a batch of one solve per
core costs the slowest of them — which is why the sharded curves only pull ahead as the batch
grows, and why each starts at its own width.

The GPU job sets `XLA_FLAGS=--xla_gpu_enable_command_buffer=...,WHILE
--xla_gpu_graph_min_graph_size=2`, worth 10-14% on V100/A100: the gap between wall and device time
on small batches is the while-loop condition sync, and CUDA graphs are what shave it.

## The result files

`results/<label>.csv` — one file per device, both sweeps in it, one row per timing
(`run,repeat,sweep,dtype,size,seconds`). The label carries the solver and, on CPU, the width:
`jax-a100`, `jax-cpu-1core`, `jax-cpu-16core`, `scipy-cpu-1core`.

`results/configs.csv` — the constants of each run, keyed by `label,sweep,dtype`, and beyond the
shared columns: `solver`, `tol`, `max_iterations`, `threads`, `shards`, `devices`.

## Caveats

- The small end of either axis is per-call overhead, not solve cost: scipy pays a Python call and a
  host round trip per evaluation, ours pays one dispatch for the whole `while_loop`. Read a win
  there as a harness result.
- Iteration counts are not held equal. Both solvers run to the same projected-gradient tolerance
  from the same starts; solves per second at a fixed tolerance is the whole measurement.
- Nodes are shared and clocks vary with what else is on the machine. Order-of-magnitude picture,
  not a certified ranking.
