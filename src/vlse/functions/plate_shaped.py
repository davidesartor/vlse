"""Plate-shaped functions, per https://www.sfu.ca/~ssurjano/optimization.html."""

from typing import ClassVar
from jaxtyping import Array, Float
import jax.numpy as jnp

from vlse.functions.base import TestFunction


class Booth(TestFunction):
    """
    The Booth function has a single global minimum, and is relatively flat around it.
    See https://www.sfu.ca/~ssurjano/booth.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -10.0)
        hi = jnp.full(self.d, 10.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/boothr.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = (x1 + 2 * x2 - 7) ** 2
        term2 = (2 * x1 + x2 - 5) ** 2
        y = term1 + term2
        return y


class Matyas(TestFunction):
    """
    The Matyas function has a single global minimum, and is relatively flat around it.
    See https://www.sfu.ca/~ssurjano/matya.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -10.0)
        hi = jnp.full(self.d, 10.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/matyar.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = 0.26 * (x1**2 + x2**2)
        term2 = -0.48 * x1 * x2
        y = term1 + term2
        return y


class McCormick(TestFunction):
    """
    The McCormick function has a single global minimum, and is relatively flat around it.
    See https://www.sfu.ca/~ssurjano/mccorm.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.array([-1.5, -3.0])
        hi = jnp.array([4.0, 4.0])
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-1.9133)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/mccormr.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = jnp.sin(x1 + x2)
        term2 = (x1 - x2) ** 2
        term3 = -1.5 * x1
        term4 = 2.5 * x2
        y = term1 + term2 + term3 + term4 + 1
        return y


class PowerSum(TestFunction):
    """
    The Power Sum function. The recommended value of the b-vector, for d = 4, is: b = (8, 18, 44, 114).
    See https://www.sfu.ca/~ssurjano/powersum.html for original implementation and more details.
    """

    d: int

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, 0.0)
        hi = jnp.full(self.d, float(self.d))
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # this b ensures x* = (1, 2, ..., d) and f(x*) = 0.0
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/powersumr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        b = jnp.sum(ii[:, None] ** ii, axis=-2)
        inner = jnp.sum(x[..., None] ** ii, axis=-2)
        y = jnp.sum((inner - b) ** 2, axis=-1)
        return y


class Zakharov(TestFunction):
    """
    The Zakharov function has no local minima except the global one.
    See https://www.sfu.ca/~ssurjano/zakharov.html for original implementation and more details.
    """

    d: int

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, -5.0)
        hi = jnp.full(self.d, 10.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/zakharovr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        sum1 = jnp.sum(x**2, axis=-1)
        sum2 = jnp.sum(0.5 * ii * x, axis=-1)
        y = sum1 + sum2**2 + sum2**4
        return y


__all__ = [
    "Booth",
    "Matyas",
    "McCormick",
    "PowerSum",
    "Zakharov",
]
