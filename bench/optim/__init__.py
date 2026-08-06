"""Solves per second of `vlse.optim.lbfgsb`, against scipy's Fortran L-BFGS-B on the same objective.

The two axes the function bench sweeps: `--sweep batch` grows a multistart batch at a fixed
dimension, `--sweep dim` grows the dimension at a fixed batch. A point here is a solve rather than
an evaluation -- the timed call runs L-BFGS-B from a random start in the box until the projected
gradient is under `--tol`.
"""

import argparse

import numpy as np

from ..plot import Page, load
from ..results import device_label
from ..sweep import Run

UNIT = "solves/s"
SWEEPS = ("batch", "dim")
PAGES = {"batch": "scaling-batch", "dim": "scaling-dim"}
# one CPU job per width, all on one node; 16 keeps every width on one socket
CORES = "1,2,4,8,16"
EXPONENTS = range(0, 25)
# scipy is CPU only, so GPU jobs run jax alone
GPU_CONFIGS = tuple(f"jax:{sweep}:{e}" for sweep in SWEEPS for e in EXPONENTS)
NOTE = (
    "median of per-call throughputs, 95% interval on the median shaded; "
    "L-BFGS-B to a projected gradient of 1e-9, jit warmed up"
    "<br>the host's per-call dispatch latency is subtracted, and a point within 2x of it dropped"
    "<br>only the hpc parts run f64 at full rate; the rest are 1/32 or 1/64 in hardware"
)


def cpu_configs(cores: int) -> tuple[str, ...]:
    """A batch smaller than the shard count cannot split, so its exponents start at the width."""
    solvers = ("jax", "scipy") if cores == 1 else ("jax",)
    min_exponent = (cores - 1).bit_length()
    exponents = {"batch": range(min_exponent, 31), "dim": EXPONENTS}
    return tuple(
        f"{solver}:{sweep}:{e}"
        for solver in solvers
        for sweep in SWEEPS
        for e in exponents[sweep]
    )


def task_key(device: str, config: str) -> tuple[str, str, str, int]:
    """The (label, sweep, dtype, point) one array task writes, so the board can look it up."""
    solver, sweep, exponent = config.split(":")
    return f"{solver}-{device}", sweep, "f64", 2 ** int(exponent)


def add_run_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--label",
        default=None,
        help="row label, default <solver>-<platform>-<device kind>",
    )
    p.add_argument("--fn", default="Ackley", help="vlse class name")
    p.add_argument("--solver", choices=("jax", "scipy"), default="jax")
    p.add_argument("--sweep", choices=SWEEPS, default="batch", help="the axis grown")
    # the axes the function bench fixes, so a point means the same problem in either
    p.add_argument(
        "--dim", type=int, default=64, help="dimension the batch sweep runs at"
    )
    p.add_argument(
        "--batch", type=int, default=1024, help="batch the dim sweep runs at"
    )
    p.add_argument("--tol", type=float, default=1e-9, help="projected gradient at stop")
    p.add_argument("--max-iterations", type=int, default=1000)
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="cap BLAS, 0 leaves it alone; XLA's own pool follows the affinity mask instead",
    )
    p.add_argument(
        "--shards",
        type=int,
        default=1,
        help="CPU host devices the batch is split across, each mapping its chunk on its own core",
    )
    p.add_argument("--device", type=int, default=0, help="index into jax.devices()")


def add_plot_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--devices",
        default=None,
        help="comma separated subset, default every chip with rows",
    )
    p.add_argument(
        "--dim", type=int, default=64, help="dimension the batch sweep ran at"
    )
    p.add_argument("--batch", type=int, default=1024, help="batch the dim sweep ran at")


def jax_call(
    fun, starts, bounds, tol: float, max_iterations: int, on_cpu: bool, shards: int = 1
):
    """The whole batch in one dispatch: `vmap` on GPU, serial `lax.map` per shard on CPU.
    Why CPU is serial and mapped rather than vmapped: see the README.
    """
    import jax

    from vlse.optim.lbfgsb import minimise

    one = lambda x0: minimise(fun, x0, bounds, tol=tol, max_iterations=max_iterations)
    mapped = lambda xs: jax.lax.map(one, xs)
    if on_cpu and shards > 1:
        mesh = jax.make_mesh((shards,), ("starts",))
        spec = jax.sharding.PartitionSpec("starts")
        starts = jax.device_put(starts, jax.sharding.NamedSharding(mesh, spec))
        # check_vma trips on the solver's while_loop carries, and no shard communicates anyway
        batched = jax.shard_map(
            mapped, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
        )
    else:
        batched = mapped if on_cpu else jax.vmap(one)
    solve = jax.jit(batched)
    return lambda: jax.block_until_ready(solve(starts))


def scipy_call(fun, starts, bounds, tol: float, max_iterations: int):
    """Every start through scipy, one after another, on the jax objective evaluated on the host."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import Bounds, minimize

    host = jax.local_devices(backend="cpu")[0]
    value_and_grad = jax.jit(jax.value_and_grad(fun))

    def objective(x):
        value, grad = value_and_grad(jax.device_put(jnp.asarray(x), host))
        return float(value), np.asarray(grad, dtype=np.float64)

    box = Bounds(*(np.asarray(bound) for bound in bounds))
    options = dict(maxiter=max_iterations, maxcor=10, ftol=0.0, gtol=tol)
    starts = np.asarray(starts, dtype=np.float64)

    def call() -> None:
        for x0 in starts:
            minimize(
                objective, x0, jac=True, method="L-BFGS-B", bounds=box, options=options
            )

    return call


def prepare(args) -> Run:
    """The multistart problem at each point of the axis, and the call that solves it."""
    import os

    # BLAS only: XLA sizes its pool from the affinity mask, so `taskset` is what narrows an XLA run
    if args.threads:
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[variable] = str(args.threads)

    # the host devices the shards map onto; must be set before jax initialises its backend
    if getattr(args, "shards", 1) > 1:
        flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = (
            f"{flags} --xla_force_host_platform_device_count={args.shards}".strip()
        )

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    import vlse

    cls = getattr(vlse, args.fn)
    # scipy never sees the accelerator, so it is labelled by the CPU it actually ran on
    device = jax.devices()[0 if args.solver == "scipy" else args.device]

    def timed_call(d: int, n_starts: int, seed: int):
        fun = cls(d=d)
        lo, hi = (np.asarray(bound, dtype=np.float64) for bound in fun.domain)
        bounds = tuple(jnp.broadcast_to(jnp.asarray(bound), (d,)) for bound in (lo, hi))
        rng = np.random.default_rng((d, n_starts, seed))
        starts = jnp.asarray(rng.uniform(lo, hi, size=(n_starts, d)), dtype=jnp.float64)

        if args.solver == "scipy":
            return (
                scipy_call(fun, starts, bounds, args.tol, args.max_iterations),
                lambda: None,
            )

        on_cpu = device.platform == "cpu"
        return (
            jax_call(
                fun, starts, bounds, args.tol, args.max_iterations, on_cpu, args.shards
            ),
            starts.delete,
        )

    if args.sweep == "batch":
        setup, work_at, axes = (
            (lambda n, seed: timed_call(args.dim, n, seed)),
            (lambda n: n),
            dict(d=args.dim, batch=""),
        )
    else:
        setup, work_at, axes = (
            lambda d, seed: timed_call(d, args.batch, seed),
            lambda _: args.batch,
            dict(d="", batch=args.batch),
        )

    label = args.label or f"{args.solver}-{device_label(device)}"
    return Run(
        label=label,
        device=device,
        sweep=args.sweep,
        setup=setup,
        work_at=work_at,
        constants=dict(
            dtype="f64",
            fn=args.fn,
            **axes,
            version=jax.__version__,
            solver=args.solver,
            tol=args.tol,
            max_iterations=args.max_iterations,
            threads=args.threads or "all",
            shards=args.shards,
            devices=len(jax.local_devices()),
        ),
    )


def pages(args, results: str):
    """One page per sweep, f64 only. scipy is drawn dashed: it is the baseline, not another device."""
    axes = {
        "batch": ("multistart batch", f"Ackley, d = {args.dim}", lambda point: point),
        "dim": (
            "dimension",
            f"Ackley, batch = {args.batch}",
            lambda _: args.batch,
        ),
    }
    wanted = set(args.devices.split(",")) if args.devices else None
    for sweep, (axis_name, title, solves_at) in axes.items():
        curves = load(results, sweep, solves_at)
        reference = {
            "scipy": curve
            for (label, _), curve in curves.items()
            if label.startswith("scipy")
        }
        # the backend is the same for every device here, so the label is just the chip
        chips = {
            label.removeprefix("jax-")
            for label, _ in curves
            if label.startswith("jax-")
        }
        if wanted is not None:
            chips &= wanted
        series = {chip: curves[f"jax-{chip}", "f64"] for chip in sorted(chips)}
        yield Page(
            name=f"{PAGES[sweep]}-fp64",
            title=f"{title}, fp64",
            axis_name=axis_name,
            y_name="solves / second",
            note=NOTE,
            series={**reference, **series},
            dashed=tuple(reference),
        )
