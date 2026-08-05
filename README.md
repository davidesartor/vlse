# vlse

> **`0.0.0` is a name placeholder.** Nothing is stable yet — install from git, or pin an exact
> version, until `0.1.0` lands.

The 48 test functions of the [Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/)
(Surjanovic & Bingham, SFU), ported to JAX. Each one is a line-by-line translation of the R source
published on that site, and the test suite re-runs those R sources on every invocation to check
agreement at `rel=1e-12`.

```bash
uv add git+https://github.com/davidesartor/vlse        # equinox, jax, jaxtyping; nothing else
```

`vlse` itself is unavailable on PyPI — it collides with the existing `vise` under PyPI's
confusable-character rule. Only the distribution name carries the prefix; nothing in the code does.

## Use

Every function is an `equinox.Module` — a pytree, so it jits, vmaps and differentiates, and can be
passed into and returned from transformed functions.

```python
import vlse
import jax
import jax.random as jr

f = vlse.Ackley(d=20)
lo, hi = f.domain                                   # (-32.768, 32.768)
x = jr.uniform(jr.key(0), (100, 20), minval=lo, maxval=hi)

f(x)                                                # (100,), batched over leading axes
f(x[0])                                             # scalar
jax.grad(lambda x: f(x).sum())(x)                   # (100, 20)
f.ymin                                              # 0.0, the minimum over `domain`
```

Instantiated bare, a function is exactly as published: native domain, published output. Enable
`jax_enable_x64` before importing `vlse` for f64 (the package never touches the flag itself, and the
Hartmann coefficient tables are built at import).

### Normalized form

`normalized=True` takes the unit cube in and shifts the minimum to 0, which is usually what an
optimizer benchmark wants:

```python
g = vlse.Ackley(d=20, normalized=True)
g(jax.random.uniform(jax.random.key(0), (100, 20)))       # inputs in [0, 1]^20, min at 0.0
```

The two forms are exactly `g(u) == f(lo + (hi - lo) * u) - f.ymin`.

## API

There is no base class and no registry. Every class is standalone and exposes the same members:

```python
class Ackley(eqx.Module):
    d: int
    a: float = 20.0
    b: float = 0.2
    c: float = 2 * math.pi
    normalized: bool = False

    domain: ClassVar[tuple[float, float]] = (-32.768, 32.768)
    ymin: ClassVar[float] = 0.0

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "..."]: ...
```

**`d`** — always present, always asserted on in `__call__`. 19 functions are defined for any
dimension and take it as their first argument (`Ackley(d=5)`, `Powell(d=8)`, `Trid(d=5)`); the rest
are fixed-dimension and take nothing (`Branin()`, `Hartmann6()`).

**`domain`** — a `(lo, hi)` pair, per axis. Plain floats when the axes share bounds, a pair of
length-`d` arrays where they differ (`Bukin6`, `McCormick`, `Branin`, `Camel6`). Either way
`lo, hi = f.domain` unpacks and `lo + (hi - lo) * u` broadcasts.

**`ymin`** — the global minimum over `domain`, as a Python float.

`d`, `domain` and `ymin` are `ClassVar`s where they are constants and properties where they depend on
a field, so read them off the instance either way. Function parameters are ordinary fields carrying
the published defaults (`Ackley.a/b/c`, `Michalewicz.m`, `Shekel.b/C`, `Langermann.c/A`).

Alternate published forms are **arguments, not classes**: `rescaled=True` is the Picheny et al.
(2012) reparametrization (`Sphere`, `Rosenbrock`, `Branin`, `GoldsteinPrice`, `Hartmann6`),
`Branin(modified=True)` the Forrester et al. (2008) modified Branin, `Bohachevsky(variant=1|2|3)` one
of the three Bohachevsky functions, `Shekel(m=5|7|10)` one of the three published truncations.
`Branin(modified=True)` is a different function rather than a reparametrization, so it does not
combine with `rescaled=True`; `ForresterLowFidelity` is a separate class for the same reason.

Nothing is declared `static`. `__call__` wears `@eqx.filter_jit`, which filters by `is_array`, so
every non-array leaf (`d`, `normalized`, `rescaled`, `modified`, `variant`) is held static
automatically — python-level branches never see a tracer, and flipping one retraces:

```python
eqx.filter_jit(lambda f, x: f(x))(vlse.Ackley(d=5, normalized=True), x)
```

## Functions

All 48 import from the top level; `vlse.functions` holds them, split into the SFU site's sections.
`(d)` marks the functions that take a dimension.

| section | functions |
| --- | --- |
| `many_local_minima` | Ackley `(d)`, Bukin6, CrossInTray, DropWave, EggHolder, GramacyLee, Griewank `(d)`, HolderTable, Langermann, Levy `(d)`, Levy13, Rastrigin `(d)`, Schaffer2, Schaffer4, Schwefel `(d)`, Shubert |
| `bowl_shaped` | Bohachevsky, Perm0 `(d)`, RotatedHyperEllipsoid `(d)`, Sphere `(d)`, SumPowers `(d)`, SumSquares `(d)`, Trid `(d)` |
| `plate_shaped` | Booth, Matyas, McCormick, PowerSum `(d)`, Zakharov `(d)` |
| `valley_shaped` | Camel3, Camel6, DixonPrice `(d)`, Rosenbrock `(d)` |
| `steep_drops` | DeJong5, Easom, Michalewicz `(d)` |
| `other` | Beale, Branin, Colville, Forrester, ForresterLowFidelity, GoldsteinPrice, Hartmann3, Hartmann4, Hartmann6, Perm `(d)`, Powell `(d)`, Shekel, StyblinskiTang `(d)` |

## Deviations from the R sources

- **The Hartmann 6-D pages are swapped upstream.** `hart6r.R` computes Picheny's
  `-(2.58 + s) / 1.94` and `hart6sc.R` the plain `-s`. Here `Hartmann6()` is the plain form and `Hartmann6(rescaled=True)` the
  rescaled one; the parity table maps the two files crosswise.
- **`shekelr.R` hardcodes `m <- 10`** and ignores its argument, so `Shekel(m=5)` and `Shekel(m=7)`
  cannot be checked against it. They are checked against an independent reference instead.
- **`spherefmod.R` is published for d=6 only**; `Sphere(rescaled=True)` generalizes it. Its
  normalizing constants are recomputed from `d` and reproduce the published `1745`/`899` at d=6.
- **`Michalewicz.ymin`** has no closed form. It is computed once on a grid — the terms are separable,
  so that is exact up to grid resolution — and cached.
- **`Forrester.ymin`** is the numerically computed `-6.020740055767083` rather than the `-6.02074`
  the page prints, which is rounded up in magnitude and would put the normalized form below 0.
  `ForresterLowFidelity.ymin` is likewise numerical, and valid only for the default `(A, B, C)`.

## Development

```bash
uv sync
uv run --no-sync pytest tests -q                 # ~10s
uv run --no-sync python tools/fetch_sfu.py       # refresh tools/sfu/*.R from the SFU site
```

Parity is checked live: `tests/functions/test_reference.py` shells out to `Rscript`, evaluates the vendored R
sources, and compares — there is no frozen golden data. Without R on `PATH` that one module skips and
the rest of the suite still runs. CI covers 3.10–3.13 plus a `--resolution lowest-direct` job, so the
dependency floors are real ones.

## Benchmarks

Throughput of one function (`Ackley`) in f64, swept across the GPUs and CPUs of one cluster — grown
along the batch axis at `d=20`, and along the dimension axis at batch 128. Median per-call throughput
with a 95% interval on the median shaded, log-log, one line per device, coloured by market segment.

Either figure opens hoverable:

[![evaluations per second against batch size, f64, one line per device](https://raw.githubusercontent.com/davidesartor/vlse/main/bench/functions/scaling-batch-fp64.svg)](https://raw.githack.com/davidesartor/vlse/main/bench/functions/scaling-batch-fp64.html)

[![evaluations per second against dimension, f64, one line per device](https://raw.githubusercontent.com/davidesartor/vlse/main/bench/functions/scaling-dim-fp64.svg)](https://raw.githack.com/davidesartor/vlse/main/bench/functions/scaling-dim-fp64.html)

The f32 panels, where everything but the `hpc` parts runs one to two orders faster:
[batch](https://raw.githack.com/davidesartor/vlse/main/bench/functions/scaling-batch-fp32.html),
[dim](https://raw.githack.com/davidesartor/vlse/main/bench/functions/scaling-dim-fp32.html).

Read the small end of either axis as latency rather than as throughput — JAX is dispatch-bound there.
Nodes are shared and clocks vary with what else is on the machine, so this is an order-of-magnitude
picture, not a certified ranking.

`bench/` holds the sweeps above under [bench/functions/](bench/functions/); what is measured, how,
and where each curve stops is in [bench/README.md](bench/README.md).

## License

GPL-2.0-only, matching the SFU reference implementations this is derived from. See
[LICENSE](LICENSE) and [NOTICE](NOTICE).
