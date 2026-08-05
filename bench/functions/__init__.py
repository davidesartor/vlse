"""Throughput of one test function, against batch size at a fixed dimension and the other way round.

`--sweep batch` grows the batch at a fixed dimension, `--sweep dim` grows the dimension at a fixed
batch -- the same climb with the axes swapped.
"""

import argparse
import dataclasses

from ..plot import Page, load
from ..results import device_label
from ..sweep import Run

DTYPES = ("f16", "f32", "f64")
UNIT = "evals/s"
SWEEPS = ("batch", "dim")
PAGES = {"batch": "scaling-batch", "dim": "scaling-dim"}
# one CPU job per width, all on one node; 16 keeps every width on one socket
CORES = "1,2,4,8,16"
EXPONENTS = range(0, 25)
# what one array task runs: one dtype at one sweep at one size
GPU_CONFIGS = tuple(
    f"{dtype}:{sweep}:{e}" for dtype in DTYPES for sweep in SWEEPS for e in EXPONENTS
)
NOTE = (
    "median of per-call throughputs, 95% interval on the median shaded; "
    "jit warmed up, batch resident on the device"
    "<br>the host's per-call dispatch latency is subtracted, and a point within 2x of it dropped"
)
# what the hardware does with that dtype
CAVEATS = {
    "f16": "pascal runs fp16 at 1/64 the f32 rate; turing and newer at twice it",
    "f64": "only the hpc parts run f64 at half the f32 rate; the rest are 1/32 or 1/64",
}


def cpu_configs(cores: int) -> tuple[str, ...]:
    return GPU_CONFIGS


def task_key(device: str, config: str) -> tuple[str, str, str, int]:
    """The (label, sweep, dtype, point) one array task writes, so the board can look it up."""
    dtype, sweep, exponent = config.split(":")
    return device, sweep, dtype, 2 ** int(exponent)


def add_run_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--label", default=None, help="row label, default <platform>-<device kind>"
    )
    p.add_argument("--dtype", choices=DTYPES, default="f32")
    p.add_argument("--fn", default="Ackley", help="vlse class name")
    p.add_argument("--sweep", choices=SWEEPS, default="batch", help="the axis grown")
    p.add_argument(
        "--dim", type=int, default=64, help="dimension the batch sweep runs at"
    )
    p.add_argument(
        "--batch", type=int, default=1024, help="batch the dim sweep runs at"
    )
    p.add_argument("--device", type=int, default=0, help="index into jax.devices()")


def add_plot_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--devices",
        default=None,
        help="comma separated subset, default every device with rows",
    )
    p.add_argument(
        "--dim", type=int, default=64, help="dimension the batch sweep ran at"
    )
    p.add_argument("--batch", type=int, default=1024, help="batch the dim sweep ran at")


def prepare(args) -> Run:
    """The batch to evaluate at each point of the axis, and the call that evaluates it."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    jax.config.update("jax_enable_x64", args.dtype == "f64")

    import vlse

    cls = getattr(vlse, args.fn)
    variable_dim = any(f.name == "d" for f in dataclasses.fields(cls))
    dtype = dict(f16=jnp.float16, f32=jnp.float32, f64=jnp.float64)[args.dtype]

    def instance(dim: int):
        fn = cls(d=dim) if variable_dim else cls()
        return eqx.filter_jit(fn), fn.d, jnp.asarray(fn.domain)

    def batch_of(n: int, dim: int, domain, seed: int):
        # built on the device: at the top of the sweep the batch exceeds host memory too
        lo, hi = domain
        key = jr.fold_in(jr.key(n * dim), seed)
        return jr.uniform(key, (n, dim), dtype, minval=lo, maxval=hi)

    def timed_call(fn, x):
        return lambda: jax.block_until_ready(fn(x)), x.delete

    if args.sweep == "batch":
        # one instance and one jit for the whole sweep, since only the batch axis moves
        fixed_fn, fixed_dim, fixed_domain = instance(args.dim)
        setup = lambda n, seed: timed_call(
            fixed_fn, batch_of(n, fixed_dim, fixed_domain, seed)
        )
        work_at, axes = (lambda n: n), dict(d=fixed_dim, batch="")
    else:
        if not variable_dim:
            raise SystemExit(
                f"{args.fn} is fixed-dimension, so there is no dim to sweep"
            )

        # every point is its own instance and so its own jit: `d` is held static
        def setup(d: int, seed: int):
            fn, dim, domain = instance(d)
            return timed_call(fn, batch_of(args.batch, dim, domain, seed))

        work_at, axes = (lambda _: args.batch), dict(d="", batch=args.batch)

    device = jax.devices()[args.device]
    label = args.label or device_label(device)
    return Run(
        label=label,
        device=device,
        sweep=args.sweep,
        setup=setup,
        work_at=work_at,
        constants=dict(dtype=args.dtype, fn=args.fn, **axes, version=jax.__version__),
    )


def pages(args, results: str):
    """One page per sweep and dtype; a dtype with no rows yet is skipped by `figure`."""
    axes = {
        "batch": ("batch size", f"Ackley, d = {args.dim}", lambda point: point),
        "dim": (
            "dimension",
            f"Ackley, batch = {args.batch}",
            lambda _: args.batch,
        ),
    }
    wanted = set(args.devices.split(",")) if args.devices else None
    for sweep, (axis_name, title, evals_at) in axes.items():
        curves = load(results, sweep, evals_at)
        for dtype in DTYPES:
            labels = {label for label, seen in curves if seen == dtype}
            if wanted is not None:
                labels &= wanted
            precision = f"fp{dtype.removeprefix('f')}"
            yield Page(
                name=f"{PAGES[sweep]}-{precision}",
                title=f"{title}, {precision}",
                axis_name=axis_name,
                y_name="evaluations / second",
                note="<br>".join(filter(None, (NOTE, CAVEATS.get(dtype)))),
                series={label: curves[label, dtype] for label in sorted(labels)},
            )
