"""`vlse.optim.lbfgsb` against scipy's L-BFGS-B, from random starts over every function's box.
Each case is one function in one variant, solved in every (program, device) mode and
measured against scipy.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import wilcoxon

from lbfgsb_cases import (
    BATTERY,
    MAX_ITERATIONS,
    N_STARTS,
    TOL,
    VARIANTS,
    objective_of,
    reference_of,
    scipy_reference,
    starts_in,
)

import vlse
from vlse.optim.lbfgsb import minimise, projected_gradient_norm

CASES = [(name, variant) for name in BATTERY for variant in VARIANTS]
IDS = [f"{name}-{variant}" for name, variant in CASES]

GPU = [device for device in jax.devices() if device.platform != "cpu"]
# full (program, device) grid, so a disagreement is attributed to one factor
PROGRAMS = ["sequential", "vmap"]
MODES = PROGRAMS + ([f"{program}-gpu" for program in PROGRAMS] if GPU else [])

# cases multimodal enough that the solvers legitimately land in different basins;
# a case earns its place only by failing the exact-match assertions below in both variants
MULTIPLE_MINIMA: set[tuple[str, str]] = {
    (name, variant)
    for name in [
        "Ackley",
        "Bohachevsky1",
        "Bohachevsky2",
        "Bohachevsky3",
        "Branin",
        "Bukin6",
        "Camel6",
        "CrossInTray",
        "DeJong5",
        "DixonPrice",
        "DropWave",
        "EggHolder",
        "GramacyLee",
        "Griewank",
        "Hartmann3",
        "Hartmann4",
        "Hartmann6",
        "HolderTable",
        "Langermann",
        "Levy",
        "Levy13",
        "Michalewicz",
        "Perm",
        "Perm0",
        "PowerSum",
        "Rastrigin",
        "Rosenbrock",
        "Schaffer2",
        "Schaffer4",
        "Schwefel",
        "Shekel",
        "Shubert",
        "StyblinskiTang",
    ]
    for variant in VARIANTS
}

# Sidak-corrected per-instance level, counted over classes: the other factors are not
# independent tries at the question
CLASSES_UNDER_TEST = {type(instance()) for instance in BATTERY.values()}
ALPHA = 1 - (1 - 0.05) ** (1 / len(CLASSES_UNDER_TEST))


def device_of(mode: str):
    return GPU[0] if mode.endswith("-gpu") else jax.devices("cpu")[0]


def solved(fun, bounds, mode: str):
    """One mode's solutions, as (x, f, iteration, error, failed_linesearch) stacked on the host."""
    starts = jax.device_put(starts_in(bounds, N_STARTS, seed=0), device_of(mode))

    def solve(x0):
        return minimise(fun, x0, bounds, tol=TOL, max_iterations=MAX_ITERATIONS)

    if mode.startswith("sequential"):
        jitted = jax.jit(solve)
        states = [jitted(x0) for x0 in starts]
        state = jax.tree.map(lambda *leaves: jnp.stack(leaves), *states)
    else:
        state = jax.jit(jax.vmap(solve))(starts)
    # back to the host: mixing a GPU array with a CPU one in a comparison is an error
    return jax.device_get(jax.block_until_ready(state))


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_case_in_every_mode(case):
    """Scipy once, every mode solved once, everything asserted in one place."""
    name, variant = case
    fun, bounds, _ = objective_of(name, VARIANTS[variant])
    f_starts, x_scipy, f_scipy = reference_of(name, VARIANTS[variant])

    states = {mode: solved(fun, bounds, mode) for mode in MODES}

    # every mode stops for a reason (not the iteration cap), inside the box, below its start
    for mode, state in states.items():
        assert jnp.all(state.iteration < MAX_ITERATIONS), mode
        assert jnp.all(state.f <= f_starts + 1e-12), mode
        assert jnp.all((state.error <= TOL) | state.failed_linesearch), mode
        assert jnp.all(state.x >= bounds[0]) and jnp.all(state.x <= bounds[1]), mode

    # sequential and vmap must agree per device, to a tolerance since XLA reassociates the
    # vmapped matmuls; only where one basin exists
    devices = (["", "-gpu"] if GPU else [""]) if case not in MULTIPLE_MINIMA else []
    for suffix in devices:
        reference_program, *others = [f"{program}{suffix}" for program in PROGRAMS]
        expected = states[reference_program]
        for program in others:
            state = states[program]
            against = f"{program} against {reference_program}"
            assert np.allclose(state.f, expected.f, rtol=1e-9, atol=1e-9), against
            assert np.allclose(state.x, expected.x, atol=5e-3, rtol=1e-4), against

    # a different basin than scipy on one start is allowed, systematically worse ones are not:
    # paired Wilcoxon on the shared starts
    for mode, state in states.items():
        gaps = (np.asarray(state.f) - np.asarray(f_scipy)) / (1.0 + np.abs(f_scipy))
        # deadband so the test doesn't rank f64 round-off
        gaps = np.where(np.abs(gaps) <= 1e-8, 0.0, gaps)
        p = (
            wilcoxon(gaps, alternative="greater", zero_method="zsplit").pvalue
            if np.any(gaps != 0.0)
            else 1.0
        )
        assert p > ALPHA, (
            f"{mode}: median {float(jnp.median(state.f)):.12g} against scipy's "
            f"{float(jnp.median(f_scipy)):.12g}, worse on {int(np.sum(gaps > 0))} of "
            f"{N_STARTS} starts, p={p:.3g}"
        )

    # unimodal cases must agree with scipy on the point, not just the value
    if case not in MULTIPLE_MINIMA:
        for mode, state in states.items():
            assert np.allclose(state.f, f_scipy, rtol=1e-9, atol=1e-9), mode
            assert np.allclose(state.x, x_scipy, atol=5e-3, rtol=1e-4), mode


def test_the_battery_covers_every_vlse_function():
    shipped = {
        cls.__name__
        for cls in vars(vlse).values()
        if isinstance(cls, type)
        and issubclass(cls, eqx.Module)
        and cls is not eqx.Module
    }
    covered = {cls.__name__ for cls in CLASSES_UNDER_TEST}
    assert shipped == covered


def test_bounds_none_is_the_unbounded_problem():
    rosenbrock = vlse.Rosenbrock(d=3)
    state = minimise(rosenbrock, jnp.array([-1.2, 1.0, 0.5]), None, tol=TOL)
    assert np.allclose(state.x, 1.0, atol=1e-4)


def test_active_upper_bound_matches_scipy():
    fun = vlse.Rosenbrock(d=10)
    bounds = (jnp.full(10, -2.0), jnp.full(10, 0.5))
    starts = jnp.full((1, 10), -0.5)

    state = minimise(fun, starts[0], bounds, tol=TOL, max_iterations=MAX_ITERATIONS)
    x_scipy, f_scipy = scipy_reference(fun, starts, bounds)

    assert state.f == pytest.approx(float(f_scipy[0]), abs=1e-9)
    assert np.allclose(state.x, x_scipy[0], atol=1e-5)
    # the upper bound is what stops it, so the solution must sit on it
    assert jnp.max(state.x) == pytest.approx(0.5)


def test_run_longer_than_the_history_matches_scipy():
    """A short memory forces the history buffer to wrap."""
    fun = vlse.SumSquares(d=25)
    bounds = tuple(jnp.broadcast_to(jnp.asarray(bound), (25,)) for bound in fun.domain)
    starts = starts_in(bounds, 1, seed=0)

    state = minimise(
        fun, starts[0], bounds, tol=TOL, max_iterations=MAX_ITERATIONS, history_length=3
    )
    x_scipy, _ = scipy_reference(fun, starts, bounds, history_length=3)

    assert np.allclose(state.x, x_scipy[0], atol=1e-5)
    assert state.n_updates == 3
    assert state.iteration > 3


def test_reports_a_projected_gradient_below_its_tolerance():
    """A box whose whole interior is uphill, so the answer is the lower corner."""
    fun = vlse.SumSquares(d=10)
    bounds = (jnp.ones(10), jnp.full(10, 2.0))
    state = minimise(fun, jnp.full(10, 1.5), bounds, tol=1e-6, max_iterations=500)

    _, grad = jax.value_and_grad(fun)(state.x)
    assert state.error <= 1e-6
    assert projected_gradient_norm(state.x, grad, *bounds) <= 1e-6
    assert np.allclose(state.x, 1.0)


def test_the_objective_takes_x_alone_or_extra_args():
    """`args` is scipy's: splatted onto `fun` whatever its length."""
    shift, scale = jnp.full(3, 0.25), 3.0
    bounds = (jnp.zeros(3), jnp.ones(3))
    x0 = jnp.full(3, 0.9)

    closed_over = minimise(
        lambda x: scale * jnp.sum((x - shift) ** 2), x0, bounds, tol=TOL
    )
    one_arg = minimise(
        lambda x, s: scale * jnp.sum((x - s) ** 2), x0, bounds, args=(shift,), tol=TOL
    )
    two_args = minimise(
        lambda x, s, c: c * jnp.sum((x - s) ** 2),
        x0,
        bounds,
        args=(shift, scale),
        tol=TOL,
    )

    assert np.allclose(closed_over.x, shift, atol=1e-6)
    assert np.allclose(one_arg.x, closed_over.x)
    assert np.allclose(two_args.x, closed_over.x)


def test_a_wrapped_objective_solves_the_same_as_the_bare_module():
    """Wrapping the module must change nothing but the call."""
    f = vlse.Sphere(d=4)
    bounds = tuple(jnp.broadcast_to(jnp.asarray(bound), (4,)) for bound in f.domain)
    x0 = jnp.full(4, 2.0)

    bare = minimise(f, x0, bounds, tol=TOL)
    closed_over = minimise(lambda x: f(x), x0, bounds, tol=TOL)
    shifted = minimise(lambda x, s: f(x - s), x0, bounds, args=(jnp.zeros(4),), tol=TOL)

    assert np.allclose(bare.x, 0.0, atol=1e-6)
    assert np.allclose(closed_over.x, bare.x)
    assert np.allclose(shifted.x, bare.x)


def test_takes_args_and_vmaps_over_them():
    """A batch costs its slowest member; each element runs its own while_loop."""
    shifts = jnp.linspace(-1.0, 1.0, 5)

    def fun(x, shift):
        return jnp.sum((x - shift) ** 2)

    bounds = (jnp.zeros(3), jnp.ones(3))
    states = jax.vmap(
        lambda shift: minimise(fun, jnp.full(3, 0.5), bounds, args=(shift,), tol=TOL)
    )(shifts)

    assert np.allclose(states.x, jnp.clip(shifts, 0.0, 1.0)[:, None], atol=1e-6)
