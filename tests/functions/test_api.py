"""Contract every test function must satisfy: shapes, the normalized form, jit, grad."""

import dataclasses
import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import vlse

CLASSES = [
    c
    for c in vars(vlse).values()
    if isinstance(c, type) and issubclass(c, eqx.Module) and c is not eqx.Module
]
IDS = [c.__name__ for c in CLASSES]


def instantiate(cls, any_d=8, **kwargs):
    # fill missing required fields: `d` gets any dimension, a Literal its first option
    for field in dataclasses.fields(cls):
        if (
            not field.init
            or field.name in kwargs
            or field.default is not dataclasses.MISSING
        ):
            continue
        kwargs[field.name] = (
            any_d if field.name == "d" else typing.get_args(field.type)[0]
        )
    return cls(**kwargs)


@pytest.fixture(params=CLASSES, ids=IDS)
def cls(request):
    return request.param


def sample(f, n=7, seed=0):
    return jr.uniform(jr.key(seed), (n, f.d))


def test_domain_bounds_broadcast_and_are_ordered(cls):
    f = instantiate(cls)
    lo, hi = jnp.asarray(f.domain)
    assert bool(jnp.all(lo < hi))
    shapes = (jnp.shape(lo), jnp.shape(hi), (f.d,))
    assert jnp.broadcast_shapes(*shapes) == (f.d,)


def test_maps_a_batch_of_inputs_to_a_batch_of_scalars(cls):
    f = instantiate(cls)
    assert f(sample(f)).shape == (7,)
    assert f(sample(f)[0]).shape == ()


def test_normalized_is_the_published_form_on_the_rescaled_box_minus_ymin(cls):
    f, g = instantiate(cls), instantiate(cls, normalized=True)
    lo, hi = jnp.asarray(f.domain)
    u = sample(f)
    assert g(u).tolist() == pytest.approx((f(lo + (hi - lo) * u) - f.ymin).tolist())


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
@pytest.mark.parametrize("normalized", [False, True])
def test_output_keeps_the_dtype_of_the_input(cls, dtype, normalized):
    # constants and tables must not promote even with x64 on
    f = instantiate(cls, normalized=normalized)
    u = sample(f).astype(dtype)
    assert f(u).dtype == dtype


def test_rejects_inputs_of_the_wrong_dimension(cls):
    f = instantiate(cls)
    with pytest.raises(AssertionError):
        f(jnp.zeros((3, f.d + 1)))


def test_is_jittable_and_differentiable(cls):
    f = instantiate(cls, normalized=True)
    u = sample(f)
    # eqx.filter_jit: a module carrying array tables is not hashable as a static arg
    assert eqx.filter_jit(f)(u).tolist() == pytest.approx(f(u).tolist())
    grad = jax.grad(lambda z: f(z).sum())(u)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_normalized_branches_the_same_when_the_module_is_a_jitted_argument(cls):
    # `normalized` is traced; remaining python-level branches are on static leaves
    u = sample(instantiate(cls))
    for f in (instantiate(cls), instantiate(cls, normalized=True)):
        traced = eqx.filter_jit(lambda module, z: module(z))(f, u)
        assert traced.tolist() == pytest.approx(f(u).tolist())


def test_powell_needs_a_multiple_of_four_dimensions():
    with pytest.raises(AssertionError):
        vlse.Powell(d=5)


def test_shekel_needs_at_most_ten_terms():
    with pytest.raises(AssertionError):
        vlse.Shekel(m=11)
