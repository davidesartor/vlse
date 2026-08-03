"""Bowl-shaped functions, per https://www.sfu.ca/~ssurjano/optimization.html."""

from typing import ClassVar, Literal
from jaxtyping import Array, Float
import jax.numpy as jnp
import equinox as eqx


class Bohachevsky(eqx.Module):
    """
    The three Bohachevsky functions all have the same similar bowl shape.
    See https://www.sfu.ca/~ssurjano/boha.html for original implementation and more details.
    """

    variant: Literal[1, 2, 3]
    normalized: bool = False

    d: ClassVar[int] = 2
    domain: ClassVar[tuple[float, float]] = (-100.0, 100.0)
    ymin: ClassVar[float] = 0.0

    def __check_init__(self):
        assert self.variant in (1, 2, 3), "Bohachevsky has three variants, 1 to 3"

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"Bohachevsky needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        if self.variant == 1:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/boha1r.html
            x1, x2 = x[..., 0], x[..., 1]
            term1 = x1**2
            term2 = 2 * x2**2
            term3 = -0.3 * jnp.cos(3 * jnp.pi * x1)
            term4 = -0.4 * jnp.cos(4 * jnp.pi * x2)
            y = term1 + term2 + term3 + term4 + 0.7
        elif self.variant == 2:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/boha2r.html
            x1, x2 = x[..., 0], x[..., 1]
            term1 = x1**2
            term2 = 2 * x2**2
            term3 = -0.3 * jnp.cos(3 * jnp.pi * x1) * jnp.cos(4 * jnp.pi * x2)
            y = term1 + term2 + term3 + 0.3
        else:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/boha3r.html
            x1, x2 = x[..., 0], x[..., 1]
            term1 = x1**2
            term2 = 2 * x2**2
            term3 = -0.3 * jnp.cos(3 * jnp.pi * x1 + 4 * jnp.pi * x2)
            y = term1 + term2 + term3 + 0.3

        if self.normalized:
            y = y - self.ymin
        return y


class Perm0(eqx.Module):
    """
    The Perm 0 function has a single global minimum.
    See https://www.sfu.ca/~ssurjano/perm0db.html for original implementation and more details.
    """

    d: int
    beta: float = 10.0
    normalized: bool = False

    ymin: ClassVar[float] = 0.0

    @property
    def domain(self) -> tuple[float, float]:
        return -float(self.d), float(self.d)

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"Perm0 needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/perm0dbr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        xi = x[..., None, :] ** ii[..., None]
        inner = (ii + self.beta) * (xi - (1 / ii) ** ii[..., None])
        y = jnp.sum(jnp.sum(inner, axis=-1) ** 2, axis=-1)

        if self.normalized:
            y = y - self.ymin
        return y


class RotatedHyperEllipsoid(eqx.Module):
    """
    The Rotated Hyper-Ellipsoid function is continuous, convex and unimodal.
    It is an extension of the Axis Parallel Hyper-Ellipsoid function, also referred to as the Sum Squares function.
    See https://www.sfu.ca/~ssurjano/rothyp.html for original implementation and more details.
    """

    d: int
    normalized: bool = False

    domain: ClassVar[tuple[float, float]] = (-65.536, 65.536)
    ymin: ClassVar[float] = 0.0

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"RotatedHyperEllipsoid needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/rothypr.html
        inner = jnp.cumsum(x**2, axis=-1)
        y = jnp.sum(inner, axis=-1)

        if self.normalized:
            y = y - self.ymin
        return y


class Sphere(eqx.Module):
    """
    The Sphere function has d local minima except for the global one. It is continuous, convex and unimodal.
    `rescaled=True` is the form of Picheny et al. (2012), on [0, 1]^d, standardized by the mean and
    sd of its weighted sum, both of which grow with the dimension.
    See https://www.sfu.ca/~ssurjano/spheref.html for original implementation and more details.
    """

    d: int
    rescaled: bool = False
    normalized: bool = False

    @property
    def domain(self) -> tuple[float, float]:
        return (0.0, 1.0) if self.rescaled else (-5.12, 5.12)

    @property
    def ymin(self) -> float:
        return -self.mean / self.sd if self.rescaled else 0.0

    @property
    def mean(self) -> float:
        # mean of the raw weighted sum, pinned to the published d=6 constant 1745.
        weight_sum = 2 ** (self.d + 1) - 2
        return 1745 * weight_sum / 126

    @property
    def sd(self) -> float:
        # sd of the raw weighted sum, pinned to the published d=6 constant 899.
        squared_weight_sum = (4 ** (self.d + 1) - 4) / 3
        return 899 * (squared_weight_sum / 5460) ** 0.5

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"Sphere needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        if self.rescaled:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/spherefmodr.html
            ii = jnp.arange(1, x.shape[-1] + 1, dtype=x.dtype)
            y = (jnp.sum(x**2 * 2.0**ii, axis=-1) - self.mean) / self.sd
        else:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/spherefr.html
            y = jnp.sum(x**2, axis=-1)

        if self.normalized:
            y = y - self.ymin
        return y


class SumPowers(eqx.Module):
    """
    The Sum of Different Powers function is unimodal.
    See https://www.sfu.ca/~ssurjano/sumpow.html for original implementation and more details.
    """

    d: int
    normalized: bool = False

    domain: ClassVar[tuple[float, float]] = (-1.0, 1.0)
    ymin: ClassVar[float] = 0.0

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"SumPowers needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/sumpowr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        y = jnp.sum(jnp.abs(x) ** (ii + 1), axis=-1)

        if self.normalized:
            y = y - self.ymin
        return y


class SumSquares(eqx.Module):
    """
    The Sum Squares function, also referred to as the Axis Parallel Hyper-Ellipsoid function, has no local minimum except the global one.
    It is continuous, convex and unimodal.
    See https://www.sfu.ca/~ssurjano/sumsqu.html for original implementation and more details.
    """

    d: int
    normalized: bool = False

    domain: ClassVar[tuple[float, float]] = (-5.12, 5.12)
    ymin: ClassVar[float] = 0.0

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"SumSquares needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/sumsqur.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        y = jnp.sum(ii * x**2, axis=-1)

        if self.normalized:
            y = y - self.ymin
        return y


class Trid(eqx.Module):
    """
    The Trid function has no local minimum except the global one.
    See https://www.sfu.ca/~ssurjano/trid.html for original implementation and more details.
    """

    d: int
    normalized: bool = False

    @property
    def domain(self) -> tuple[float, float]:
        return -float(self.d**2), float(self.d**2)

    @property
    def ymin(self) -> float:
        return -self.d * (self.d + 4) * (self.d - 1) / 6

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"Trid needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/tridr.html
        sum1 = jnp.sum((x - 1) ** 2, axis=-1)
        sum2 = jnp.sum(x[..., 1:] * x[..., :-1], axis=-1)
        y = sum1 - sum2

        if self.normalized:
            y = y - self.ymin
        return y


__all__ = [
    "Bohachevsky",
    "Perm0",
    "RotatedHyperEllipsoid",
    "Sphere",
    "SumPowers",
    "SumSquares",
    "Trid",
]
