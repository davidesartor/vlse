"""The host's per-call dispatch latency: the cheapest jitted call there is, timed on its own.

Every timing in the other benches is blocked on, so the round trip to the device adds to the
kernel. It is host-side -- an arm host costs four times an x86 one -- so it is measured once per
device here and subtracted there. One point, no axis: `plot` draws nothing, the number is read
by `bench/plot.py`.
"""

import argparse

from ..results import device_label
from ..sweep import Run

UNIT = "calls/s"
CORES = "1,2,4,8,16"
GPU_CONFIGS = ("latency",)
# every other curve is corrected by this one number, and the call is microseconds: buy the sample
REPEATS = 500
WALLTIME = "00:20:00"


def cpu_configs(cores: int) -> tuple[str, ...]:
    return GPU_CONFIGS


def task_key(device: str, config: str) -> tuple[str, str, str, int]:
    return device, "dispatch", "f32", 1


def add_run_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--label", default=None, help="row label, default <platform>-<device kind>"
    )
    p.add_argument("--device", type=int, default=0, help="index into jax.devices()")


def add_plot_arguments(p: argparse.ArgumentParser) -> None:
    pass


def prepare(args) -> Run:
    """One element through one jitted add, blocked on: all round trip, no kernel worth timing."""
    import jax
    import jax.numpy as jnp

    device = jax.devices()[args.device]
    step = jax.jit(lambda v: v + 1)

    def setup(size: int, seed: int):
        x = jax.device_put(jnp.full((1,), seed, jnp.float32), device)
        return lambda: jax.block_until_ready(step(x)), x.delete

    return Run(
        label=args.label or device_label(device),
        device=device,
        sweep="dispatch",
        setup=setup,
        work_at=lambda _: 1,
        constants=dict(dtype="f32", fn="", d=1, batch=1, version=jax.__version__),
    )


def pages(args, results: str):
    return ()
