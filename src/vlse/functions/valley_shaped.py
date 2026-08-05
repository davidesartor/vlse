"""Valley-shaped functions, per https://www.sfu.ca/~ssurjano/optimization.html."""

from typing import ClassVar
from jaxtyping import Array, Float
import jax.numpy as jnp

from vlse.functions.base import TestFunction


class Camel3(TestFunction):
    """
    The Three-Hump Camel function has three local minima.
    See https://www.sfu.ca/~ssurjano/camel3.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -5.0)
        hi = jnp.full(self.d, 5.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/camel3r.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = 2 * x1**2
        term2 = -1.05 * x1**4
        term3 = x1**6 / 6
        term4 = x1 * x2
        term5 = x2**2
        y = term1 + term2 + term3 + term4 + term5
        return y


class Camel6(TestFunction):
    """
    The Six-Hump Camel function has six local minima, two of which are global.
    See https://www.sfu.ca/~ssurjano/camel6.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.array([-3.0, -2.0])
        hi = jnp.array([3.0, 2.0])
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-1.0316285)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/camel6r.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2
        y = term1 + term2 + term3
        return y


class DixonPrice(TestFunction):
    """
    The Dixon-Price function is continuous and unimodal.
    See https://www.sfu.ca/~ssurjano/dixonpr.html for original implementation and more details.
    """

    d: int

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, -10.0)
        hi = jnp.full(self.d, 10.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/dixonprr.html
        ii = jnp.arange(2, x.shape[-1] + 1)
        term1 = (x[..., 0] - 1) ** 2
        term2 = ii * (2 * x[..., 1:] ** 2 - x[..., :-1]) ** 2
        y = term1 + jnp.sum(term2, axis=-1)
        return y


class Rosenbrock(TestFunction):
    """
    The Rosenbrock function, also referred to as the Valley or Banana function, has a narrow, curved valley containing the global minimum.
    `rescaled=True` is the 4D form of Picheny et al. (2012), on [0, 1]^4.
    See https://www.sfu.ca/~ssurjano/rosen.html for original implementation and more details.
    """

    d: int
    rescaled: bool = False

    def __check_init__(self):
        assert (
            not self.rescaled or self.d == 4
        ), "the rescaled Rosenbrock is only defined for d=4"

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, 0.0 if self.rescaled else -5.0)
        hi = jnp.full(self.d, 1.0 if self.rescaled else 10.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        ymin = -3.827e5 / 3.755e5 if self.rescaled else 0.0
        return jnp.asarray(ymin)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        if self.rescaled:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/rosenscr.html
            x = 15 * x - 5
            term1 = 100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2
            term2 = (x[..., :-1] - 1) ** 2
            y = (jnp.sum(term1 + term2, axis=-1) - 3.827e5) / 3.755e5
        else:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/rosenr.html
            term1 = 100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2
            term2 = (x[..., :-1] - 1) ** 2
            y = jnp.sum(term1 + term2, axis=-1)
        return y


__all__ = [
    "Camel3",
    "Camel6",
    "DixonPrice",
    "Rosenbrock",
]
