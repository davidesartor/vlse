"""Instances the L-BFGS-B tests solve, plus their cached scipy reference."""

import hashlib
import os
import pathlib
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from scipy.optimize import Bounds, minimize as scipy_minimize

import vlse

TOL = 1e-9
MAX_ITERATIONS = 1000
N_STARTS = 64

# one instance per vlse function, built either published or normalized
BATTERY: dict[str, Callable[..., eqx.Module]] = {
    "Ackley": lambda **kw: vlse.Ackley(d=5, **kw),
    "Beale": vlse.Beale,
    "Bohachevsky1": lambda **kw: vlse.Bohachevsky(variant=1, **kw),
    "Bohachevsky2": lambda **kw: vlse.Bohachevsky(variant=2, **kw),
    "Bohachevsky3": lambda **kw: vlse.Bohachevsky(variant=3, **kw),
    "Booth": vlse.Booth,
    "Branin": vlse.Branin,
    "Bukin6": vlse.Bukin6,
    "Camel3": vlse.Camel3,
    "Camel6": vlse.Camel6,
    "Colville": vlse.Colville,
    "CrossInTray": vlse.CrossInTray,
    "DeJong5": vlse.DeJong5,
    "DixonPrice": lambda **kw: vlse.DixonPrice(d=5, **kw),
    "DropWave": vlse.DropWave,
    "Easom": vlse.Easom,
    "EggHolder": vlse.EggHolder,
    "Forrester": vlse.Forrester,
    "ForresterLowFidelity": vlse.ForresterLowFidelity,
    "GoldsteinPrice": vlse.GoldsteinPrice,
    "GramacyLee": vlse.GramacyLee,
    "Griewank": lambda **kw: vlse.Griewank(d=4, **kw),
    "Hartmann3": vlse.Hartmann3,
    "Hartmann4": vlse.Hartmann4,
    "Hartmann6": vlse.Hartmann6,
    "HolderTable": vlse.HolderTable,
    "Langermann": vlse.Langermann,
    "Levy": lambda **kw: vlse.Levy(d=6, **kw),
    "Levy13": vlse.Levy13,
    "Matyas": vlse.Matyas,
    "McCormick": vlse.McCormick,
    "Michalewicz": lambda **kw: vlse.Michalewicz(d=5, **kw),
    "Perm": lambda **kw: vlse.Perm(d=4, **kw),
    "Perm0": lambda **kw: vlse.Perm0(d=4, **kw),
    "Powell": lambda **kw: vlse.Powell(d=8, **kw),
    "PowerSum": lambda **kw: vlse.PowerSum(d=4, **kw),
    "Rastrigin": lambda **kw: vlse.Rastrigin(d=4, **kw),
    "Rosenbrock": lambda **kw: vlse.Rosenbrock(d=6, **kw),
    "RotatedHyperEllipsoid": lambda **kw: vlse.RotatedHyperEllipsoid(d=5, **kw),
    "Schaffer2": vlse.Schaffer2,
    "Schaffer4": vlse.Schaffer4,
    "Schwefel": lambda **kw: vlse.Schwefel(d=3, **kw),
    "Shekel": vlse.Shekel,
    "Shubert": vlse.Shubert,
    "Sphere": lambda **kw: vlse.Sphere(d=8, **kw),
    "StyblinskiTang": lambda **kw: vlse.StyblinskiTang(d=5, **kw),
    "SumPowers": lambda **kw: vlse.SumPowers(d=4, **kw),
    "SumSquares": lambda **kw: vlse.SumSquares(d=6, **kw),
    "Trid": lambda **kw: vlse.Trid(d=6, **kw),
    "Zakharov": lambda **kw: vlse.Zakharov(d=5, **kw),
}

VARIANTS = {"published": False, "normalized": True}


def objective_of(name: str, normalized: bool) -> tuple[Callable, tuple, int]:
    """The objective, its box and its dimension; a normalized instance takes the unit cube."""
    f = BATTERY[name](normalized=normalized)
    box = (0.0, 1.0) if normalized else f.domain
    lower, upper = (
        jnp.broadcast_to(jnp.asarray(bound, dtype=jnp.float64), (f.d,)) for bound in box
    )
    return f, (lower, upper), f.d


def starts_in(bounds: tuple, n: int, seed: int) -> Float[Array, "n d"]:
    """Uniform starts over the whole box."""
    rng = np.random.default_rng(seed)
    lower, upper = (np.asarray(bound, dtype=np.float64) for bound in bounds)
    return jnp.asarray(rng.uniform(lower, upper, size=(n, lower.size)))


@eqx.filter_jit
def value_and_grad(
    fun: Callable, x: Float[Array, "d"], *args
) -> tuple[Float[Array, ""], Float[Array, "d"]]:
    """Jitted once at module level so cases share the compile cache."""
    return jax.value_and_grad(fun)(x, *args)


def scipy_reference(
    fun: Callable,
    starts: Float[Array, "n d"],
    bounds: tuple,
    args: tuple = (),
    tol: float = TOL,
    history_length: int = 10,
) -> tuple[Float[Array, "n d"], Float[Array, "n"]]:
    """The same problems through scipy's L-BFGS-B, one start at a time."""

    def numpy_value_and_grad(x):
        f, grad = value_and_grad(fun, jnp.asarray(x), *args)
        return float(f), np.asarray(grad, dtype=np.float64)

    lower, upper = (np.asarray(bound, dtype=np.float64) for bound in bounds)
    results = [
        scipy_minimize(
            numpy_value_and_grad,
            np.asarray(x0, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=Bounds(lower, upper),
            options=dict(
                maxiter=MAX_ITERATIONS, maxcor=history_length, ftol=0.0, gtol=tol
            ),
        )
        for x0 in starts
    ]
    return (
        jnp.asarray(np.stack([r.x for r in results])),
        jnp.asarray(np.array([float(r.fun) for r in results])),
    )


REFERENCE_CACHE = pathlib.Path(__file__).parent / ".reference_cache"


def _fingerprint(
    f_starts: Float[Array, "n"], grads: Float[Array, "n d"], bounds: tuple
) -> str:
    """Everything the scipy solve reads, as one digest.
    The objective enters through its values and gradients at the starts, not its name.
    """
    payload = np.concatenate(
        [
            np.asarray(f_starts, dtype=np.float64).ravel(),
            np.asarray(grads, dtype=np.float64).ravel(),
            np.asarray(bounds[0], dtype=np.float64).ravel(),
            np.asarray(bounds[1], dtype=np.float64).ravel(),
            np.array([TOL, MAX_ITERATIONS, N_STARTS, 10, 0], dtype=np.float64),
        ]
    )
    return hashlib.blake2b(payload.tobytes(), digest_size=16).hexdigest()


def reference_of(
    name: str, normalized: bool
) -> tuple[Float[Array, "n"], Float[Array, "n d"], Float[Array, "n"]]:
    """f at the starts and scipy's solutions for one case, off disk when the case is unchanged.
    Keyed by fingerprint, so an edited function or tolerance misses rather than returns stale answers.
    """
    fun, bounds, _ = objective_of(name, normalized)
    starts = starts_in(bounds, N_STARTS, seed=0)
    f_starts, grads = jax.vmap(jax.value_and_grad(fun))(starts)

    variant = "normalized" if normalized else "published"
    entry = (
        REFERENCE_CACHE
        / f"{name}-{variant}-{_fingerprint(f_starts, grads, bounds)}.npz"
    )
    if entry.exists():
        cached = np.load(entry)
        return f_starts, jnp.asarray(cached["x"]), jnp.asarray(cached["f"])

    x_scipy, f_scipy = scipy_reference(fun, starts, bounds)

    REFERENCE_CACHE.mkdir(exist_ok=True)
    (REFERENCE_CACHE / ".gitignore").write_text("*\n")
    # write-and-rename: concurrent jobs on a shared filesystem must never read a half-written npz
    staged = entry.with_name(f"{entry.name}.{os.getpid()}.tmp.npz")
    np.savez(staged, x=np.asarray(x_scipy), f=np.asarray(f_scipy))
    staged.replace(entry)
    return f_starts, x_scipy, f_scipy
