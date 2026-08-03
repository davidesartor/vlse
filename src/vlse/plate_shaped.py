"""Plate-shaped functions, per https://www.sfu.ca/~ssurjano/optimization.html."""

from typing import ClassVar
from jaxtyping import Array, Float
import jax.numpy as jnp
import equinox as eqx


class Booth(eqx.Module):
    """
    The Booth function has a single global minimum, and is relatively flat around it.
    See https://www.sfu.ca/~ssurjano/booth.html for original implementation and more details.
    """

    normalized: bool = False

    d: ClassVar[int] = 2
    domain: ClassVar[tuple[float, float]] = (-10.0, 10.0)
    ymin: ClassVar[float] = 0.0

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"Booth needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/boothr.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = (x1 + 2 * x2 - 7) ** 2
        term2 = (2 * x1 + x2 - 5) ** 2
        y = term1 + term2

        if self.normalized:
            y = y - self.ymin
        return y


class Matyas(eqx.Module):
    """
    The Matyas function has a single global minimum, and is relatively flat around it.
    See https://www.sfu.ca/~ssurjano/matya.html for original implementation and more details.
    """

    normalized: bool = False

    d: ClassVar[int] = 2
    domain: ClassVar[tuple[float, float]] = (-10.0, 10.0)
    ymin: ClassVar[float] = 0.0

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"Matyas needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/matyar.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = 0.26 * (x1**2 + x2**2)
        term2 = -0.48 * x1 * x2
        y = term1 + term2

        if self.normalized:
            y = y - self.ymin
        return y


class McCormick(eqx.Module):
    """
    The McCormick function has a single global minimum, and is relatively flat around it.
    See https://www.sfu.ca/~ssurjano/mccorm.html for original implementation and more details.
    """

    normalized: bool = False

    d: ClassVar[int] = 2
    domain: ClassVar[tuple[Float[Array, "2"], Float[Array, "2"]]] = (
        jnp.array([-1.5, -3.0]),
        jnp.array([4.0, 4.0]),
    )
    ymin: ClassVar[float] = -1.9133

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"McCormick needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/mccormr.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = jnp.sin(x1 + x2)
        term2 = (x1 - x2) ** 2
        term3 = -1.5 * x1
        term4 = 2.5 * x2
        y = term1 + term2 + term3 + term4 + 1

        if self.normalized:
            y = y - self.ymin
        return y


class PowerSum(eqx.Module):
    """
    The Power Sum function. The recommended value of the b-vector, for d = 4, is: b = (8, 18, 44, 114).
    See https://www.sfu.ca/~ssurjano/powersum.html for original implementation and more details.
    """

    d: int
    normalized: bool = False

    ymin: ClassVar[float] = 0.0

    @property
    def domain(self) -> tuple[float, float]:
        return 0.0, float(self.d)

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"PowerSum needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # this b ensures x* = (1, 2, ..., d) and f(x*) = 0.0
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/powersumr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        b = jnp.sum(ii[:, None] ** ii, axis=-2)
        inner = jnp.sum(x[..., None] ** ii, axis=-2)
        y = jnp.sum((inner - b) ** 2, axis=-1)

        if self.normalized:
            y = y - self.ymin
        return y


class Zakharov(eqx.Module):
    """
    The Zakharov function has no local minima except the global one.
    See https://www.sfu.ca/~ssurjano/zakharov.html for original implementation and more details.
    """

    d: int
    normalized: bool = False

    domain: ClassVar[tuple[float, float]] = (-5.0, 10.0)
    ymin: ClassVar[float] = 0.0

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"Zakharov needs d={self.d} inputs"

        if self.normalized:
            lo, hi = self.domain
            x = lo + (hi - lo) * x

        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/zakharovr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        sum1 = jnp.sum(x**2, axis=-1)
        sum2 = jnp.sum(0.5 * ii * x, axis=-1)
        y = sum1 + sum2**2 + sum2**4

        if self.normalized:
            y = y - self.ymin
        return y


__all__ = [
    "Booth",
    "Matyas",
    "McCormick",
    "PowerSum",
    "Zakharov",
]
