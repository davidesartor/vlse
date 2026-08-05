"""Functions with steep ridges or drops, per https://www.sfu.ca/~ssurjano/optimization.html."""

from functools import cached_property
from typing import ClassVar
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp

from vlse.functions.base import TestFunction


class DeJong5(TestFunction):
    """
    The fifth function of De Jong is multimodal, with very sharp drops on a mainly flat surface.
    See https://www.sfu.ca/~ssurjano/dejong5.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -65.536)
        hi = jnp.full(self.d, 65.536)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.998)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/dejong5r.html
        x1, x2 = x[..., 0], x[..., 1]
        A = jnp.array([[-32.0, -16.0, 0.0, 16.0, 32.0]] * 5, x.dtype)
        ii = jnp.arange(1, 25 + 1)
        term1 = (x1[..., None] - A.flatten()) ** 6
        term2 = (x2[..., None] - A.T.flatten()) ** 6
        inner = jnp.sum(1 / (ii + term1 + term2), axis=-1)
        y = 1 / (0.002 + inner)
        return y


class Easom(TestFunction):
    """
    The Easom function has several local minima. It is unimodal, and the global minimum has a small area relative to the search space.
    See https://www.sfu.ca/~ssurjano/easom.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -100.0)
        hi = jnp.full(self.d, 100.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-1.0)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/easomr.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = -jnp.cos(x1) * jnp.cos(x2)
        term2 = jnp.exp(-((x1 - jnp.pi) ** 2) - (x2 - jnp.pi) ** 2)
        y = term1 * term2
        return y


class Michalewicz(TestFunction):
    """
    The Michalewicz function has d! local minima, and it is multimodal.
    The parameter m defines the steepness of they valleys and ridges, a larger m leads to a more difficult search.
    See https://www.sfu.ca/~ssurjano/michal.html for original implementation and more details.
    """

    d: int
    m: float = 10.0

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, 0.0)
        hi = jnp.full(self.d, jnp.pi)
        return lo, hi

    @cached_property
    def ymin(self) -> Float[Array, ""]:
        # no closed form: the terms are separable, so a grid is exact up to its resolution.
        with jax.ensure_compile_time_eval():
            x = jnp.linspace(0.0, jnp.pi, 200_001)
            ii = jnp.arange(1, self.d + 1)[:, None]
            terms = jnp.sin(x) * jnp.sin(ii * x**2 / jnp.pi) ** (2 * self.m)
            return -terms.max(axis=-1).sum()

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/michalr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        inner = jnp.sin(x) * jnp.sin(ii / jnp.pi * x**2) ** (2 * self.m)
        y = -jnp.sum(inner, axis=-1)
        return y


__all__ = [
    "DeJong5",
    "Easom",
    "Michalewicz",
]
