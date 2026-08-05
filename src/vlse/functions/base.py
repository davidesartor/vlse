"""Base class shared by the SFU test functions."""

from abc import abstractmethod
from jaxtyping import Array, Bool, Float
import jax.numpy as jnp
import equinox as eqx


class TestFunction(eqx.Module):
    d: eqx.AbstractVar[int]
    domain: eqx.AbstractVar[tuple[Float[Array, "d"], Float[Array, "d"]]]
    ymin: eqx.AbstractVar[Float[Array, ""]]

    normalized: Bool[Array, ""] = eqx.field(
        default=False, kw_only=True, converter=jnp.asarray
    )

    @abstractmethod
    def f(self, x: Float[Array, "... d"]) -> Float[Array, "..."]: ...

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]:
        assert x.shape[-1] == self.d, f"{type(self).__name__} needs d={self.d} inputs"

        # normalized takes inputs in the unit cube and shifts the minimum to zero
        lo, hi = (jnp.asarray(bound, x.dtype) for bound in self.domain)
        x = jnp.where(self.normalized, lo + (hi - lo) * x, x)

        # evaluate the function on its own domain
        y = self.f(x)

        # normalized takes inputs in the unit cube and shifts the minimum to zero
        ymin = jnp.asarray(self.ymin, y.dtype)
        y = jnp.where(self.normalized, y - ymin, y)
        return y


__all__ = ["TestFunction"]
