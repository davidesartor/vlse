"""`ymin` must be the global minimum: attained at the published minimizer, and never undercut."""

import dataclasses

import jax.numpy as jnp
import jax.random as jr
import pytest

import vlse
from test_api import CLASSES, IDS, instantiate

# (instance, published minimizer in native coordinates, published minimum)
KNOWN_MINIMA = [
    (vlse.Ackley(d=5), [0.0] * 5, 0.0),
    (vlse.Beale(), [3.0, 0.5], 0.0),
    (vlse.Bohachevsky(variant=1), [0.0, 0.0], 0.0),
    (vlse.Bohachevsky(variant=2), [0.0, 0.0], 0.0),
    (vlse.Bohachevsky(variant=3), [0.0, 0.0], 0.0),
    (vlse.Booth(), [1.0, 3.0], 0.0),
    (vlse.Branin(), [-jnp.pi, 12.275], 0.397887),
    (vlse.Camel6(), [0.0898, -0.7126], -1.0316),
    (
        vlse.DixonPrice(d=5),
        [2.0 ** -((2.0 ** jnp.arange(1, 6) - 2) / 2.0 ** jnp.arange(1, 6))],
        0.0,
    ),
    (vlse.DropWave(), [0.0, 0.0], -1.0),
    (vlse.Easom(), [jnp.pi, jnp.pi], -1.0),
    (vlse.GoldsteinPrice(), [0.0, -1.0], 3.0),
    (vlse.Griewank(d=5), [0.0] * 5, 0.0),
    (vlse.Hartmann3(), [0.114614, 0.555649, 0.852547], -3.86278),
    (
        vlse.Hartmann6(),
        [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
        -3.32237,
    ),
    (vlse.Levy(d=5), [1.0] * 5, 0.0),
    (vlse.Levy13(), [1.0, 1.0], 0.0),
    (vlse.Matyas(), [0.0, 0.0], 0.0),
    (vlse.McCormick(), [-0.54719, -1.54719], -1.9133),
    (vlse.Powell(d=8), [0.0] * 8, 0.0),
    (vlse.Rastrigin(d=5), [0.0] * 5, 0.0),
    (vlse.Rosenbrock(d=5), [1.0] * 5, 0.0),
    (vlse.RotatedHyperEllipsoid(d=5), [0.0] * 5, 0.0),
    (vlse.Schwefel(d=5), [420.9687] * 5, 0.0),
    (vlse.Shekel(m=10), [4.0, 4.0, 4.0, 4.0], -10.5364),
    (vlse.Sphere(d=5), [0.0] * 5, 0.0),
    (vlse.StyblinskiTang(d=5), [-2.903534] * 5, -39.16599 * 5),
    (vlse.SumPowers(d=5), [0.0] * 5, 0.0),
    (vlse.SumSquares(d=5), [0.0] * 5, 0.0),
    (vlse.Trid(d=5), [i * (6 - i) for i in range(1, 6)], -30.0),
    (vlse.Zakharov(d=5), [0.0] * 5, 0.0),
]
MIN_IDS = [type(f).__name__ for f, _, _ in KNOWN_MINIMA]


@pytest.mark.parametrize("f, argmin, ymin", KNOWN_MINIMA, ids=MIN_IDS)
def test_ymin_is_attained_at_the_published_minimizer(f, argmin, ymin):
    x = jnp.ravel(jnp.asarray(argmin))
    assert float(f(x)) == pytest.approx(ymin, rel=1e-4, abs=1e-4)
    assert f.ymin == pytest.approx(ymin, rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("f, argmin, ymin", KNOWN_MINIMA, ids=MIN_IDS)
def test_normalized_form_is_zero_at_the_published_minimizer(f, argmin, ymin):
    lo, hi = jnp.asarray(f.domain)
    u = (jnp.ravel(jnp.asarray(argmin)) - lo) / (hi - lo)
    g = dataclasses.replace(f, normalized=True)
    # the published minimizers are rounded, so this is only as tight as their last digit
    assert float(g(u)) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("cls", CLASSES, ids=IDS)
def test_random_search_never_undercuts_ymin(cls):
    f = instantiate(cls, normalized=True)
    u = jr.uniform(jr.key(0), (20_000, f.d))
    assert float(jnp.min(f(u))) > -1e-9
