"""Functions with steep ridges or drops, per https://www.sfu.ca/~ssurjano/optimization.html."""

from functools import lru_cache
from typing import ClassVar
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
import equinox as eqx


class DeJong5(eqx.Module):
    """
    The fifth function of De Jong is multimodal, with very sharp drops on a mainly flat surface.
    See https://www.sfu.ca/~ssurjano/dejong5.html for original implementation and more details.
    """

    normalized: bool = False

    d: ClassVar[int] = 2
    domain: ClassVar[tuple[float, float]] = (-65.536, 65.536)
    ymin: ClassVar[float] = 0.998

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"DeJong5 needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/dejong5r.html
        x1, x2 = x[..., 0], x[..., 1]
        A = jnp.array([[-32.0, -16.0, 0.0, 16.0, 32.0]] * 5)
        ii = jnp.arange(1, 25 + 1)
        term1 = (x1[..., None] - A.flatten()) ** 6
        term2 = (x2[..., None] - A.T.flatten()) ** 6
        inner = jnp.sum(1 / (ii + term1 + term2), axis=-1)
        y = 1 / (0.002 + inner)

        if self.normalized:
            y = y - self.ymin
        return y


class Easom(eqx.Module):
    """
    The Easom function has several local minima. It is unimodal, and the global minimum has a small area relative to the search space.
    See https://www.sfu.ca/~ssurjano/easom.html for original implementation and more details.
    """

    normalized: bool = False

    d: ClassVar[int] = 2
    domain: ClassVar[tuple[float, float]] = (-100.0, 100.0)
    ymin: ClassVar[float] = -1.0

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"Easom needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/easomr.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = -jnp.cos(x1) * jnp.cos(x2)
        term2 = jnp.exp(-((x1 - jnp.pi) ** 2) - (x2 - jnp.pi) ** 2)
        y = term1 * term2

        if self.normalized:
            y = y - self.ymin
        return y


class Michalewicz(eqx.Module):
    """
    The Michalewicz function has d! local minima, and it is multimodal.
    The parameter m defines the steepness of they valleys and ridges, a larger m leads to a more difficult search.
    See https://www.sfu.ca/~ssurjano/michal.html for original implementation and more details.
    """

    d: int
    m: float = 10.0
    normalized: bool = False

    @property
    def domain(self) -> tuple[float, float]:
        return (0.0, jnp.pi)

    @property
    def ymin(self) -> float:
        return self._grid_ymin(self.d, self.m)

    @staticmethod
    @lru_cache
    def _grid_ymin(d: int, m: float, grid: int = 200_001) -> float:
        # no closed form: the terms are separable, so a grid is exact up to its resolution.
        with jax.ensure_compile_time_eval():
            x = jnp.linspace(0.0, jnp.pi, grid)
            ii = jnp.arange(1, d + 1)[:, None]
            terms = jnp.sin(x) * jnp.sin(ii * x**2 / jnp.pi) ** (2 * m)
            return float(-terms.max(axis=-1).sum())

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"Michalewicz needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/michalr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        inner = jnp.sin(x) * jnp.sin(ii / jnp.pi * x**2) ** (2 * self.m)
        y = -jnp.sum(inner, axis=-1)

        if self.normalized:
            y = y - self.ymin
        return y


__all__ = [
    "DeJong5",
    "Easom",
    "Michalewicz",
]
