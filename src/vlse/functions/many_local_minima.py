"""Functions with many local minima, per https://www.sfu.ca/~ssurjano/optimization.html."""

from typing import ClassVar
from jaxtyping import Array, Float
import jax.numpy as jnp
import equinox as eqx

from vlse.functions.base import TestFunction


class Ackley(TestFunction):
    """
    The Ackley function is characterized by a nearly flat outer region, and a large hole at the centre.
    See https://www.sfu.ca/~ssurjano/ackley.html for original implementation and more details.
    """

    d: int
    a: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=20.0)
    b: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=0.2)
    c: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=2 * jnp.pi)

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, -32.768)
        hi = jnp.full(self.d, 32.768)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/ackleyr.html
        mean1 = jnp.mean(x**2, axis=-1)
        mean2 = jnp.mean(jnp.cos(self.c * x), axis=-1)
        term1 = -self.a * jnp.exp(-self.b * jnp.sqrt(mean1))
        term2 = -jnp.exp(mean2)
        y = term1 + term2 + self.a + jnp.exp(1.0)
        return y


class Bukin6(TestFunction):
    """
    The sixth Bukin function has many local minima, all of which lie in a ridge.
    See https://www.sfu.ca/~ssurjano/bukin6.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.array([-15.0, -3.0])
        hi = jnp.array([-5.0, 3.0])
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/bukin6r.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = 100 * jnp.sqrt(jnp.abs(x2 - 0.01 * x1**2))
        term2 = 0.01 * jnp.abs(x1 + 10)
        y = term1 + term2
        return y


class CrossInTray(TestFunction):
    """
    The Cross-in-Tray function has multiple global minima, in the characteristic "cross" pattern.
    See https://www.sfu.ca/~ssurjano/crossit.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -10.0)
        hi = jnp.full(self.d, 10.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-2.06261)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/crossitr.html
        x1, x2 = x[..., 0], x[..., 1]
        fact1 = jnp.sin(x1) * jnp.sin(x2)
        fact2 = jnp.exp(jnp.abs(100 - jnp.sqrt(x1**2 + x2**2) / jnp.pi))
        y = -0.0001 * (jnp.abs(fact1 * fact2) + 1) ** 0.1
        return y


class DropWave(TestFunction):
    """
    The Drop-Wave function is multimodal and highly complex.
    See https://www.sfu.ca/~ssurjano/drop.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -5.12)
        hi = jnp.full(self.d, 5.12)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-1.0)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/dropr.html
        x1, x2 = x[..., 0], x[..., 1]
        frac1 = 1 + jnp.cos(12 * jnp.sqrt(x1**2 + x2**2))
        frac2 = 0.5 * (x1**2 + x2**2) + 2
        y = -frac1 / frac2
        return y


class EggHolder(TestFunction):
    """
    The Eggholder function is a difficult function to optimize, because of the large number of local minima.
    See https://www.sfu.ca/~ssurjano/egg.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -512.0)
        hi = jnp.full(self.d, 512.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-959.6407)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/eggr.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = -(x2 + 47) * jnp.sin(jnp.sqrt(jnp.abs(x2 + x1 / 2 + 47)))
        term2 = -x1 * jnp.sin(jnp.sqrt(jnp.abs(x1 - (x2 + 47))))
        y = term1 + term2
        return y


class GramacyLee(TestFunction):
    """
    The Gramacy-Lee function is a 1D function with many local minima.
    See https://www.sfu.ca/~ssurjano/grlee12.html for original implementation and more details.
    """

    d: ClassVar[int] = 1

    @property
    def domain(self) -> tuple[Float[Array, "1"], Float[Array, "1"]]:
        lo = jnp.full(self.d, 0.5)
        hi = jnp.full(self.d, 2.5)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-0.869011135)

    def f(self, x: Float[Array, "... 1"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/grlee12r.html
        x = x[..., 0]
        term1 = jnp.sin(10 * jnp.pi * x) / (2 * x)
        term2 = (x - 1) ** 4
        y = term1 + term2
        return y


class Griewank(TestFunction):
    """
    The Griewank function has many widespread local minima, which are regularly distributed.
    See https://www.sfu.ca/~ssurjano/griewank.html for original implementation and more details.
    """

    d: int

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, -600.0)
        hi = jnp.full(self.d, 600.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/griewankr.html
        ii = jnp.arange(1, x.shape[-1] + 1, dtype=x.dtype)
        sum_term = jnp.sum(x**2 / 4000, axis=-1)
        prod_term = jnp.prod(jnp.cos(x / jnp.sqrt(ii)), axis=-1)
        y = sum_term - prod_term + 1
        return y


class HolderTable(TestFunction):
    """
    The Holder Table function has many local minima, with four global minima.
    See https://www.sfu.ca/~ssurjano/holder.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -10.0)
        hi = jnp.full(self.d, 10.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-19.2085)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/holderr.html
        x1, x2 = x[..., 0], x[..., 1]
        fact1 = jnp.sin(x1) * jnp.cos(x2)
        fact2 = jnp.exp(jnp.abs(1 - jnp.sqrt(x1**2 + x2**2) / jnp.pi))
        y = -jnp.abs(fact1 * fact2)
        return y


class Langermann(TestFunction):
    """
    The Langermann function is multimodal, with many unevenly distributed local minima.
    See https://www.sfu.ca/~ssurjano/langer.html for original implementation and more details.
    """

    c: Float[Array, "m"] = eqx.field(
        converter=jnp.asarray, default=(1.0, 2.0, 5.0, 2.0, 3.0)
    )
    A: Float[Array, "m d"] = eqx.field(
        converter=jnp.asarray,
        default=((3.0, 5.0), (5.0, 2.0), (2.0, 1.0), (1.0, 4.0), (7.0, 9.0)),
    )

    def __check_init__(self):
        assert self.A.ndim == 2, "A must be a 2D array"
        assert self.c.ndim == 1, "c must be a 1D array"
        assert self.A.shape[0] == self.c.shape[0], (
            "A and c must have the same number of rows"
        )

    @property
    def d(self) -> int:
        return self.A.shape[-1]

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, 0.0)
        hi = jnp.full(self.d, 10.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-4.1558)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/langerr.html
        A, c = self.A.astype(x.dtype), self.c.astype(x.dtype)
        inner = jnp.sum((x[..., None] - A.T) ** 2, axis=-2)
        term1 = jnp.exp(-inner / jnp.pi)
        term2 = jnp.cos(jnp.pi * inner)
        y = jnp.sum(c * term1 * term2, axis=-1)
        return y


class Levy(TestFunction):
    """
    The Levy function has many local minima, with a regular distribution.
    See https://www.sfu.ca/~ssurjano/levy.html for original implementation and more details.
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
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/levyr.html
        w = 1 + (x - 1) / 4
        w1, wi, wd = w[..., 0], w[..., :-1], w[..., -1]
        term1 = jnp.sin(jnp.pi * w1) ** 2
        term2 = (wi - 1) ** 2 * (1 + 10 * jnp.sin(jnp.pi * wi + 1.0) ** 2)
        term3 = (wd - 1) ** 2 * (1 + jnp.sin(2 * jnp.pi * wd) ** 2)
        y = term1 + term2.sum(axis=-1) + term3
        return y


class Levy13(TestFunction):
    """
    The Levy N.13 function is a 2D function with many local minima.
    See https://www.sfu.ca/~ssurjano/levy13.html for original implementation and more details.
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
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/levy13r.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = jnp.sin(3 * jnp.pi * x1) ** 2
        term2 = (x1 - 1) ** 2 * (1 + jnp.sin(3 * jnp.pi * x2) ** 2)
        term3 = (x2 - 1) ** 2 * (1 + jnp.sin(2 * jnp.pi * x2) ** 2)
        y = term1 + term2 + term3
        return y


class Rastrigin(TestFunction):
    """
    The Rastrigin function has several local minima. It is highly multimodal, but locations of the minima are regularly distributed.
    See https://www.sfu.ca/~ssurjano/rastr.html for original implementation and more details.
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
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/rastrr.html
        terms = x**2 - 10 * jnp.cos(2 * jnp.pi * x)
        y = 10 * x.shape[-1] + jnp.sum(terms, axis=-1)
        return y


class Schaffer2(TestFunction):
    """
    The Schaffer N.2 function is a 2D function with many local minima.
    See https://www.sfu.ca/~ssurjano/schaffer2.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -100.0)
        hi = jnp.full(self.d, 100.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/schaffer2r.html
        x1, x2 = x[..., 0], x[..., 1]
        fact1 = jnp.sin(x1**2 - x2**2) ** 2 - 0.5
        fact2 = (1 + 0.001 * (x1**2 + x2**2)) ** 2
        y = 0.5 + fact1 / fact2
        return y


class Schaffer4(TestFunction):
    """
    The Schaffer N.4 function is a 2D function with many local minima.
    See https://www.sfu.ca/~ssurjano/schaffer4.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -100.0)
        hi = jnp.full(self.d, 100.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.29257873)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/schaffer4r.html
        x1, x2 = x[..., 0], x[..., 1]
        fact1 = jnp.cos(jnp.sin(jnp.abs(x1**2 - x2**2))) ** 2 - 0.5
        fact2 = (1 + 0.001 * (x1**2 + x2**2)) ** 2
        y = 0.5 + fact1 / fact2
        return y


class Schwefel(TestFunction):
    """
    The Schwefel function is complex, with many local minima.
    See https://www.sfu.ca/~ssurjano/schwef.html for original implementation and more details.
    """

    d: int

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, -500.0)
        hi = jnp.full(self.d, 500.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/schwefr.html
        terms = x * jnp.sin(jnp.sqrt(jnp.abs(x)))
        y = 418.9829 * x.shape[-1] - jnp.sum(terms, axis=-1)
        return y


class Shubert(TestFunction):
    """
    The Shubert function has several local minima and many global minima.
    See https://www.sfu.ca/~ssurjano/shubert.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -5.12)
        hi = jnp.full(self.d, 5.12)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-186.7309)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/shubertr.html
        x1, x2 = x[..., 0], x[..., 1]
        ii = jnp.arange(1, 6)
        term1 = ii * jnp.cos((ii + 1) * x1[..., None] + ii)
        term2 = ii * jnp.cos((ii + 1) * x2[..., None] + ii)
        y = jnp.sum(term1, axis=-1) * jnp.sum(term2, axis=-1)
        return y


__all__ = [
    "Ackley",
    "Bukin6",
    "CrossInTray",
    "DropWave",
    "EggHolder",
    "GramacyLee",
    "Griewank",
    "HolderTable",
    "Langermann",
    "Levy",
    "Levy13",
    "Rastrigin",
    "Schaffer2",
    "Schaffer4",
    "Schwefel",
    "Shubert",
]
