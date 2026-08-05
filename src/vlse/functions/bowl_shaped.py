"""Bowl-shaped functions, per https://www.sfu.ca/~ssurjano/optimization.html."""

from typing import ClassVar, Literal
from jaxtyping import Array, Float
import jax.numpy as jnp
import equinox as eqx

from vlse.functions.base import TestFunction


class Bohachevsky(TestFunction):
    """
    The three Bohachevsky functions all have the same similar bowl shape.
    See https://www.sfu.ca/~ssurjano/boha.html for original implementation and more details.
    """

    d: ClassVar[int] = 2
    variant: Literal[1, 2, 3]

    def __check_init__(self):
        assert self.variant in (1, 2, 3), "Bohachevsky has three variants, 1 to 3"

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -100.0)
        hi = jnp.full(self.d, 100.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
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
        return y


class Perm0(TestFunction):
    """
    The Perm 0 function has a single global minimum.
    See https://www.sfu.ca/~ssurjano/perm0db.html for original implementation and more details.
    """

    d: int
    beta: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=10.0)

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, -float(self.d))
        hi = jnp.full(self.d, float(self.d))
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/perm0dbr.html
        ii = jnp.arange(1, x.shape[-1] + 1, dtype=x.dtype)
        xi = x[..., None, :] ** ii[..., None]
        inner = (ii + self.beta) * (xi - (1 / ii) ** ii[..., None])
        y = jnp.sum(jnp.sum(inner, axis=-1) ** 2, axis=-1)
        return y


class RotatedHyperEllipsoid(TestFunction):
    """
    The Rotated Hyper-Ellipsoid function is continuous, convex and unimodal.
    It is an extension of the Axis Parallel Hyper-Ellipsoid function, also referred to as the Sum Squares function.
    See https://www.sfu.ca/~ssurjano/rothyp.html for original implementation and more details.
    """

    d: int

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, -65.536)
        hi = jnp.full(self.d, 65.536)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/rothypr.html
        inner = jnp.cumsum(x**2, axis=-1)
        y = jnp.sum(inner, axis=-1)
        return y


class Sphere(TestFunction):
    """
    The Sphere function has d local minima except for the global one. It is continuous, convex and unimodal.
    `rescaled=True` is the form of Picheny et al. (2012), on [0, 1]^d, standardized by the mean and
    sd of its weighted sum, both of which grow with the dimension.
    See https://www.sfu.ca/~ssurjano/spheref.html for original implementation and more details.
    """

    d: int
    rescaled: bool = False

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, 0.0 if self.rescaled else -5.12)
        hi = jnp.full(self.d, 1.0 if self.rescaled else 5.12)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        ymin = -self.mean / self.sd if self.rescaled else 0.0
        return jnp.asarray(ymin)

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

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        if self.rescaled:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/spherefmodr.html
            ii = jnp.arange(1, x.shape[-1] + 1, dtype=x.dtype)
            y = (jnp.sum(x**2 * 2.0**ii, axis=-1) - self.mean) / self.sd
        else:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/spherefr.html
            y = jnp.sum(x**2, axis=-1)
        return y


class SumPowers(TestFunction):
    """
    The Sum of Different Powers function is unimodal.
    See https://www.sfu.ca/~ssurjano/sumpow.html for original implementation and more details.
    """

    d: int

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, -1.0)
        hi = jnp.full(self.d, 1.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/sumpowr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        y = jnp.sum(jnp.abs(x) ** (ii + 1), axis=-1)
        return y


class SumSquares(TestFunction):
    """
    The Sum Squares function, also referred to as the Axis Parallel Hyper-Ellipsoid function, has no local minimum except the global one.
    It is continuous, convex and unimodal.
    See https://www.sfu.ca/~ssurjano/sumsqu.html for original implementation and more details.
    """

    d: int

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, -5.12)
        hi = jnp.full(self.d, 5.12)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/sumsqur.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        y = jnp.sum(ii * x**2, axis=-1)
        return y


class Trid(TestFunction):
    """
    The Trid function has no local minimum except the global one.
    See https://www.sfu.ca/~ssurjano/trid.html for original implementation and more details.
    """

    d: int

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, -float(self.d**2))
        hi = jnp.full(self.d, float(self.d**2))
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        ymin = -self.d * (self.d + 4) * (self.d - 1) / 6
        return jnp.asarray(ymin)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/tridr.html
        sum1 = jnp.sum((x - 1) ** 2, axis=-1)
        sum2 = jnp.sum(x[..., 1:] * x[..., :-1], axis=-1)
        y = sum1 - sum2
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
