# Dispatch latency

What one blocking jitted call costs before any work happens: `jax.jit(lambda v: v + 1)` on a single
element, `block_until_ready`, on every device the other two benches run on. One number per device,
no axis and no plot — [`../plot.py`](../plot.py) reads it and subtracts it off every point of the
function and optim curves.

```bash
uv run python -m bench dispatch run --label a100
uv run python -m bench dispatch submit --dry-run
```

Run once and left alone: the latency is a property of the host, the backend and the driver, not of
what is being benchmarked, so it does not need re-measuring when a sweep is re-run. `--repeats`
defaults to 500 here against the other benches' 12, and `--reps` to 11 — a call this cheap makes the
whole job minutes, and every other curve is corrected by this one number.

## Why it is measured rather than inferred

Every timing in the other benches is blocked on, so the round trip to the device adds to the kernel
instead of overlapping it. At the small end of either axis that round trip is the whole measurement,
and it is host-side: an aarch64 Grace host costs ~0.5 ms a call against ~0.12 ms on an x86 one, so
gh200 read as *slower* than a V100 for everything under a million evaluations, on the host and not
the GPU.

Estimating the floor from the curves themselves does not work. The flat region wobbles by as much as
the floor is worth — gh200's runs 0.38–0.65 ms — so every order statistic of it (min of medians,
pooled median, max-across-runs-min-across-configs) lands somewhere inside that spread rather than
under it. Subtracting a floor that some timings sit *below* divides by near-zero: a100 spiked to
122 G evals/s, 25× its real peak. Capping the correction instead put a false peak in gh200's curve
at batch 4.19M and a decline after it, because the cap bound the small end and not the large. A
separately measured floor has none of these problems: it is a constant with its own sample, so the
subtraction is well-founded and a point whose timings fall under it is honestly dispatch-bound and
dropped.

## The result files

`results/<label>.csv` — same shape as the other benches (`run,repeat,sweep,dtype,size,seconds`),
with `sweep` always `dispatch`, `dtype` `f32` and `size` 1. The label must match the one the other
benches write under, minus any solver prefix: `a100.csv` covers both `a100` and `jax-a100`.
`scipy-*` has no row and gets no correction — it pays a Python call per evaluation, which is the
measurement rather than an overhead on it.
