"""Functions in no other shape category, per https://www.sfu.ca/~ssurjano/optimization.html."""

from typing import ClassVar
from jaxtyping import Array, Float
import jax.numpy as jnp
import equinox as eqx

from vlse.functions.base import TestFunction


class Beale(TestFunction):
    """
    The Beale function is multimodal, with sharp peaks at the corners of the input domain.
    See https://www.sfu.ca/~ssurjano/beale.html for original implementation and more details.
    """

    d: ClassVar[int] = 2

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, -4.5)
        hi = jnp.full(self.d, 4.5)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/bealer.html
        x1, x2 = x[..., 0], x[..., 1]
        term1 = (1.5 - x1 + x1 * x2) ** 2
        term2 = (2.25 - x1 + x1 * x2**2) ** 2
        term3 = (2.625 - x1 + x1 * x2**3) ** 2
        y = term1 + term2 + term3
        return y


class Branin(TestFunction):
    """
    The Branin, or Branin-Hoo, function has three global minima.
    `rescaled=True` is the form of Picheny et al. (2012) on [0, 1]^2, which hardcodes its coefficients.
    `modified=True` adds the 5*x1 term of Forrester et al. (2008), which leaves a single global
    minimum among two local ones.
    See https://www.sfu.ca/~ssurjano/branin.html for original implementation and more details.
    """

    d: ClassVar[int] = 2
    a: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=1.0)
    b: Float[Array, ""] = eqx.field(
        converter=jnp.asarray, default=5.1 / (4 * jnp.pi**2)
    )
    c: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=5.0 / jnp.pi)
    r: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=6.0)
    s: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=10.0)
    t: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=1.0 / (8 * jnp.pi))
    rescaled: bool = False
    modified: bool = False

    def __check_init__(self):
        # the site publishes no rescaling of the modified form
        assert not (
            self.rescaled and self.modified
        ), "Branin is either rescaled or modified, not both"

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        if self.rescaled:
            lo, hi = jnp.full(self.d, 0.0), jnp.full(self.d, 1.0)
        else:
            lo, hi = jnp.array([-5.0, 0.0]), jnp.array([10.0, 15.0])
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        if self.rescaled:
            return jnp.asarray(-1.047393)
        return jnp.asarray(-16.644022 if self.modified else 0.397887)

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        if self.rescaled:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/braninscr.html
            x1 = 15 * x[..., 0] - 5
            x2 = 15 * x[..., 1]
            term1 = x2 - 5.1 * x1**2 / (4 * jnp.pi**2) + 5 * x1 / jnp.pi - 6
            term2 = (10 - 10 / (8 * jnp.pi)) * jnp.cos(x1)
            y = (term1**2 + term2 - 44.81) / 51.95
        elif self.modified:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/braninmodifr.html
            x1, x2 = x[..., 0], x[..., 1]
            term1 = self.a * (x2 - self.b * x1**2 + self.c * x1 - self.r) ** 2
            term2 = self.s * (1 - self.t) * jnp.cos(x1)
            y = term1 + term2 + self.s + 5 * x1
        else:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/braninr.html
            x1, x2 = x[..., 0], x[..., 1]
            term1 = self.a * (x2 - self.b * x1**2 + self.c * x1 - self.r) ** 2
            term2 = self.s * (1 - self.t) * jnp.cos(x1)
            y = term1 + term2 + self.s
        return y


class Colville(TestFunction):
    """
    The Colville function is a 4D function with several local minima.
    See https://www.sfu.ca/~ssurjano/colville.html for original implementation and more details.
    """

    d: ClassVar[int] = 4

    @property
    def domain(self) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
        lo = jnp.full(self.d, -10.0)
        hi = jnp.full(self.d, 10.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... 4"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/colviller.html
        x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        term1 = 100 * (x1**2 - x2) ** 2
        term2 = (x1 - 1) ** 2
        term3 = (x3 - 1) ** 2
        term4 = 90 * (x3**2 - x4) ** 2
        term5 = 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
        term6 = 19.8 * (x2 - 1) * (x4 - 1)
        y = term1 + term2 + term3 + term4 + term5 + term6
        return y


class Forrester(TestFunction):
    """
    This function is a simple one-dimensional test function.
    It is multimodal, with one global minimum, one local minimum and a zero-gradient inflection point.
    See https://www.sfu.ca/~ssurjano/forretal08.html for original implementation and more details.
    """

    d: ClassVar[int] = 1

    @property
    def domain(self) -> tuple[Float[Array, "1"], Float[Array, "1"]]:
        lo = jnp.full(self.d, 0.0)
        hi = jnp.full(self.d, 1.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-6.020740055767083)

    def f(self, x: Float[Array, "... 1"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/forretal08r.html
        x = x[..., 0]
        fact1 = (6 * x - 2) ** 2
        fact2 = jnp.sin(12 * x - 4)
        y = fact1 * fact2
        return y


class ForresterLowFidelity(TestFunction):
    """
    The low-fidelity companion to the Forrester function, used for multi-fidelity analysis.
    See https://www.sfu.ca/~ssurjano/forretal08.html for original implementation and more details.
    """

    d: ClassVar[int] = 1
    A: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=0.5)
    B: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=10.0)
    C: Float[Array, ""] = eqx.field(converter=jnp.asarray, default=-5.0)

    @property
    def domain(self) -> tuple[Float[Array, "1"], Float[Array, "1"]]:
        lo = jnp.full(self.d, 0.0)
        hi = jnp.full(self.d, 1.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        # only valid for the default (A, B, C)
        return jnp.asarray(0.6650951)

    def f(self, x: Float[Array, "... 1"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/forretal08lcr.html
        x = x[..., 0]
        high_fidelity = (6 * x - 2) ** 2 * jnp.sin(12 * x - 4)
        term1 = self.A * high_fidelity
        term2 = self.B * (x - 0.5)
        y = term1 + term2 - self.C
        return y


class GoldsteinPrice(TestFunction):
    """
    The Goldstein-Price function has several local minima.
    `rescaled=True` is the logarithmic form of Picheny et al. (2012), on [0, 1]^2.
    See https://www.sfu.ca/~ssurjano/goldpr.html for original implementation and more details.
    """

    d: ClassVar[int] = 2
    rescaled: bool = False

    @property
    def domain(self) -> tuple[Float[Array, "2"], Float[Array, "2"]]:
        lo = jnp.full(self.d, 0.0 if self.rescaled else -2.0)
        hi = jnp.full(self.d, 1.0 if self.rescaled else 2.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(
            (float(jnp.log(3.0) - 8.693) / 2.427) if self.rescaled else 3.0
        )

    def f(self, x: Float[Array, "... 2"]) -> Float[Array, "..."]:
        if self.rescaled:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/goldprscr.html
            x1 = 4 * x[..., 0] - 2
            x2 = 4 * x[..., 1] - 2
            fact1a = (x1 + x2 + 1) ** 2
            fact1b = 19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
            fact1 = 1 + fact1a * fact1b
            fact2a = (2 * x1 - 3 * x2) ** 2
            fact2b = 18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
            fact2 = 30 + fact2a * fact2b
            y = (jnp.log(fact1 * fact2) - 8.693) / 2.427
        else:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/goldprr.html
            x1, x2 = x[..., 0], x[..., 1]
            fact1a = (x1 + x2 + 1) ** 2
            fact1b = 19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
            fact1 = 1 + fact1a * fact1b
            fact2a = (2 * x1 - 3 * x2) ** 2
            fact2b = 18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
            fact2 = 30 + fact2a * fact2b
            y = fact1 * fact2
        return y


class Hartmann3(TestFunction):
    """
    The 3-dimensional Hartmann function has 4 local minima.
    See https://www.sfu.ca/~ssurjano/hart3.html for original implementation and more details.
    """

    d: ClassVar[int] = 3

    @property
    def domain(self) -> tuple[Float[Array, "3"], Float[Array, "3"]]:
        lo = jnp.full(self.d, 0.0)
        hi = jnp.full(self.d, 1.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-3.86278)

    @property
    def alpha(self) -> Float[Array, "4"]:
        return jnp.asarray((1.0, 1.2, 3.0, 3.2))

    @property
    def A(self) -> Float[Array, "3 4"]:
        return jnp.asarray(
            (
                (3.0, 0.1, 3.0, 0.1),
                (10.0, 10.0, 10.0, 10.0),
                (30.0, 35.0, 30.0, 35.0),
            )
        )

    @property
    def P(self) -> Float[Array, "3 4"]:
        return jnp.asarray(
            (
                (0.3689, 0.4699, 0.1091, 0.0381),
                (0.1170, 0.4387, 0.8732, 0.5743),
                (0.2673, 0.7470, 0.5547, 0.8828),
            )
        )

    def f(self, x: Float[Array, "... 3"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/hart3r.html
        A, P, alpha = (t.astype(x.dtype) for t in (self.A, self.P, self.alpha))
        inner = jnp.sum(A * (x[..., None] - P) ** 2, axis=-2)
        y = -jnp.sum(alpha * jnp.exp(-inner), axis=-1)
        return y


class Hartmann4(TestFunction):
    """
    The 4-dimensional Hartmann function is multimodal.
    It is given here in the rescaled form of Picheny et al. (2012), as the SFU R code is.
    See https://www.sfu.ca/~ssurjano/hart4.html for original implementation and more details.
    """

    d: ClassVar[int] = 4

    @property
    def domain(self) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
        lo = jnp.full(self.d, 0.0)
        hi = jnp.full(self.d, 1.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-3.1345)

    @property
    def alpha(self) -> Float[Array, "4"]:
        return jnp.asarray((1.0, 1.2, 3.0, 3.2))

    @property
    def A(self) -> Float[Array, "4 4"]:
        return jnp.asarray(
            (
                (10.0, 3.0, 17.0, 3.5),
                (0.05, 10.0, 17.0, 0.1),
                (3.0, 3.5, 1.7, 10.0),
                (17.0, 8.0, 0.05, 10.0),
            )
        )

    @property
    def P(self) -> Float[Array, "4 4"]:
        return jnp.asarray(
            (
                (0.1312, 0.1696, 0.5569, 0.0124),
                (0.2329, 0.4135, 0.8307, 0.3736),
                (0.2348, 0.1451, 0.3522, 0.2883),
                (0.4047, 0.8828, 0.8732, 0.5743),
            )
        )

    def f(self, x: Float[Array, "... 4"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/hart4r.html
        A, P, alpha = (t.astype(x.dtype) for t in (self.A, self.P, self.alpha))
        inner = jnp.sum(A * (x[..., None, :] - P) ** 2, axis=-1)
        outer = jnp.sum(alpha * jnp.exp(-inner), axis=-1)
        y = (1.1 - outer) / 0.839
        return y


class Hartmann6(TestFunction):
    """
    The 6-dimensional Hartmann function has 6 local minima.
    `rescaled=True` is the form of Picheny et al. (2012), which is what the SFU `hart6` R source computes.
    See https://www.sfu.ca/~ssurjano/hart6.html for original implementation and more details.
    """

    d: ClassVar[int] = 6
    rescaled: bool = False

    @property
    def domain(self) -> tuple[Float[Array, "6"], Float[Array, "6"]]:
        lo = jnp.full(self.d, 0.0)
        hi = jnp.full(self.d, 1.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        ymin = -3.0424577 if self.rescaled else -3.3223680
        return jnp.asarray(ymin)

    @property
    def alpha(self) -> Float[Array, "4"]:
        return jnp.asarray((1.0, 1.2, 3.0, 3.2))

    @property
    def A(self) -> Float[Array, "6 4"]:
        return jnp.asarray(
            (
                (10.0, 0.05, 3.0, 17.0),
                (3.0, 10.0, 3.5, 8.0),
                (17.0, 17.0, 1.7, 0.05),
                (3.5, 0.1, 10.0, 10.0),
                (1.7, 8.0, 17.0, 0.1),
                (8.0, 14.0, 8.0, 14.0),
            )
        )

    @property
    def P(self) -> Float[Array, "6 4"]:
        return jnp.asarray(
            (
                (0.1312, 0.2329, 0.2348, 0.4047),
                (0.1696, 0.4135, 0.1451, 0.8828),
                (0.5569, 0.8307, 0.3522, 0.8732),
                (0.0124, 0.3736, 0.2883, 0.5743),
                (0.8283, 0.1004, 0.3047, 0.1091),
                (0.5886, 0.9991, 0.6650, 0.0381),
            )
        )

    def f(self, x: Float[Array, "... 6"]) -> Float[Array, "..."]:
        A, P, alpha = (t.astype(x.dtype) for t in (self.A, self.P, self.alpha))
        if self.rescaled:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/hart6r.html
            inner = jnp.sum(A * (x[..., None] - P) ** 2, axis=-2)
            outer = jnp.sum(alpha * jnp.exp(-inner), axis=-1)
            y = -(2.58 + outer) / 1.94
        else:
            # port of R implementation from https://www.sfu.ca/~ssurjano/Code/hart6scr.html
            inner = jnp.sum(A * (x[..., None] - P) ** 2, axis=-2)
            outer = jnp.sum(alpha * jnp.exp(-inner), axis=-1)
            y = -outer
        return y


class Perm(TestFunction):
    """
    The Perm function is multimodal.
    See https://www.sfu.ca/~ssurjano/permdb.html for original implementation and more details.
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
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/permdbr.html
        ii = jnp.arange(1, x.shape[-1] + 1)
        xi = (x[..., None, :] / ii) ** ii[..., None]
        inner = (ii ** ii[..., None] + self.beta) * (xi - 1)
        y = jnp.sum(jnp.sum(inner, axis=-1) ** 2, axis=-1)
        return y


class Powell(TestFunction):
    """
    The Powell function is multimodal.
    See https://www.sfu.ca/~ssurjano/powell.html for original implementation and more details.
    """

    d: int

    def __check_init__(self):
        assert self.d % 4 == 0, "Powell is only defined for dimensions 4n"

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, -4.0)
        hi = jnp.full(self.d, 5.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(0.0)

    def f(self, x: Float[Array, "... 4*d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/powellr.html
        xx1 = x[..., 0::4]
        xx2 = x[..., 1::4]
        xx3 = x[..., 2::4]
        xx4 = x[..., 3::4]
        term1 = (xx1 + 10 * xx2) ** 2
        term2 = 5 * (xx3 - xx4) ** 2
        term3 = (xx2 - 2 * xx3) ** 4
        term4 = 10 * (xx1 - xx4) ** 4
        y = jnp.sum(term1 + term2 + term3 + term4, axis=-1)
        return y


class Shekel(TestFunction):
    """
    The Shekel function has m local minima.
    See https://www.sfu.ca/~ssurjano/shekel.html for original implementation and more details.
    """

    m: int = 10
    b: Float[Array, "10"] = eqx.field(
        converter=jnp.asarray,
        default=(0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5),
    )
    C: Float[Array, "d 10"] = eqx.field(
        converter=jnp.asarray,
        default=(
            (4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0),
            (4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6),
            (4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0),
            (4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6),
        ),
    )

    def __check_init__(self):
        assert 1 <= self.m <= 10, "Shekel is only defined for m in [1, 10]"

    @property
    def d(self) -> int:
        return self.C.shape[0]

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, 0.0)
        hi = jnp.full(self.d, 10.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        # tabulated per m, from https://www.sfu.ca/~ssurjano/shekel.html
        ymins = [
            -10.0000,
            -10.0277,
            -10.0433,
            -10.1043,
            -10.1532,
            -10.1704,
            -10.4029,
            -10.4226,
            -10.4832,
            -10.5364,
        ]
        return jnp.asarray(ymins[self.m - 1])
        

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/shekelr.html
        C, b = self.C.astype(x.dtype), self.b.astype(x.dtype)
        inner = jnp.sum((x[..., None] - C[:, : self.m]) ** 2, axis=-2)
        y = -jnp.sum(1 / (inner + b[: self.m]), axis=-1)
        return y


class StyblinskiTang(TestFunction):
    """
    The The Styblinski-Tang function is multimodal.
    See https://www.sfu.ca/~ssurjano/stybtang.html for original implementation and more details.
    """

    d: int

    @property
    def domain(self) -> tuple[Float[Array, "d"], Float[Array, "d"]]:
        lo = jnp.full(self.d, -5.0)
        hi = jnp.full(self.d, 5.0)
        return lo, hi

    @property
    def ymin(self) -> Float[Array, ""]:
        return jnp.asarray(-39.1661657 * self.d)

    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        # port of R implementation from https://www.sfu.ca/~ssurjano/Code/stybtangr.html
        y = 0.5 * jnp.sum(x**4 - 16 * x**2 + 5 * x, axis=-1)
        return y


__all__ = [
    "Beale",
    "Branin",
    "Colville",
    "Forrester",
    "ForresterLowFidelity",
    "GoldsteinPrice",
    "Hartmann3",
    "Hartmann4",
    "Hartmann6",
    "Perm",
    "Powell",
    "Shekel",
    "StyblinskiTang",
]
