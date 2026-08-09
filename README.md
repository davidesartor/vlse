# vlse

The 48 test functions of the [Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/)
(Surjanovic & Bingham, SFU), ported to JAX, plus an L-BFGS-B solver.

- Each function is a line-by-line port of the R source published on the SFU site. The test suite
  re-runs those R sources and checks agreement at `rel=1e-12`.
- Functions are callable objects and valid pytrees: they `jit`, `vmap`, `grad`, and run unchanged
  on CPU or GPU. Evaluation happens in the dtype of the input.
- `vlse.optim.minimise` is L-BFGS-B compiled to a single XLA program, tested against scipy's
  Fortran implementation.

```bash
uv add jaxvlse        # the distribution is `jaxvlse`; the import stays `import vlse`
```

## Use

```python
import vlse
import jax
import jax.random as jr

f = vlse.Ackley(d=20)
lo, hi = f.domain                                   # per-axis bounds, (-32.768, 32.768)
x = jr.uniform(jr.key(0), (100, 20), minval=lo, maxval=hi)

f(x)                                                # (100,), batched over leading axes
f(x[0])                                             # scalar
jax.grad(lambda x: f(x).sum())(x)                   # (100, 20)
f.ymin                                              # 0.0, the minimum over `domain`
```

## API

Every function has the same shape:

```python
class SomeFunction:
    d: int              # input dimension; a constructor argument for the any-d functions
    domain: (lo, hi)    # per-axis bounds, each of length d
    ymin: float         # global minimum over `domain`

    def __call__(self, x):   # (..., d) -> (...)
        ...
```

Published parameters and variants are constructor arguments, never separate classes:
`Ackley(a=, b=, c=)`, `Michalewicz(m=)`, `Shekel(m=5|7|10, b=, C=)`,
`Bohachevsky(variant=1|2|3)`, `rescaled=True` for the Picheny et al. (2012) forms,
`Branin(modified=True)` for the Forrester et al. (2008) modified Branin.

`normalized=True` takes the unit cube in and shifts the minimum to 0:
`g(u) == f(lo + (hi - lo) * u) - f.ymin`. Instantiated bare, a function is exactly as
published. For f64, enable `jax_enable_x64` before importing `vlse`.

## Functions

The dimension, domain and minimum of `any`-dimensional functions may depend on `d`; the values
below that do are given as formulas.

### Many local minima

| function | `d` | domain | minimum | parameters |
| --- | --- | --- | --- | --- |
| `Ackley` | any | [-32.768, 32.768] | 0 | `a`, `b`, `c` |
| `Bukin6` | 2 | [-15, -5] × [-3, 3] | 0 | |
| `CrossInTray` | 2 | [-10, 10] | -2.06261 | |
| `DropWave` | 2 | [-5.12, 5.12] | -1 | |
| `EggHolder` | 2 | [-512, 512] | -959.641 | |
| `GramacyLee` | 1 | [0.5, 2.5] | -0.869011 | |
| `Griewank` | any | [-600, 600] | 0 | |
| `HolderTable` | 2 | [-10, 10] | -19.2085 | |
| `Langermann` | 2 | [0, 10] | -4.1558 | `c`, `A` |
| `Levy` | any | [-10, 10] | 0 | |
| `Levy13` | 2 | [-10, 10] | 0 | |
| `Rastrigin` | any | [-5.12, 5.12] | 0 | |
| `Schaffer2` | 2 | [-100, 100] | 0 | |
| `Schaffer4` | 2 | [-100, 100] | 0.292579 | |
| `Schwefel` | any | [-500, 500] | 0 | |
| `Shubert` | 2 | [-5.12, 5.12] | -186.731 | |

### Bowl-shaped

| function | `d` | domain | minimum | parameters |
| --- | --- | --- | --- | --- |
| `Bohachevsky` | 2 | [-100, 100] | 0 | `variant` (required, 1\|2\|3) |
| `Perm0` | any | [-d, d] | 0 | `beta` |
| `RotatedHyperEllipsoid` | any | [-65.536, 65.536] | 0 | |
| `Sphere` | any | [-5.12, 5.12] | 0 | `rescaled` |
| `SumPowers` | any | [-1, 1] | 0 | |
| `SumSquares` | any | [-5.12, 5.12] | 0 | |
| `Trid` | any | [-d², d²] | -d(d+4)(d-1)/6 | |

### Plate-shaped

| function | `d` | domain | minimum | parameters |
| --- | --- | --- | --- | --- |
| `Booth` | 2 | [-10, 10] | 0 | |
| `Matyas` | 2 | [-10, 10] | 0 | |
| `McCormick` | 2 | [-1.5, 4] × [-3, 4] | -1.9133 | |
| `PowerSum` | any | [0, d] | 0 | |
| `Zakharov` | any | [-5, 10] | 0 | |

### Valley-shaped

| function | `d` | domain | minimum | parameters |
| --- | --- | --- | --- | --- |
| `Camel3` | 2 | [-5, 5] | 0 | |
| `Camel6` | 2 | [-3, 3] × [-2, 2] | -1.03163 | |
| `DixonPrice` | any | [-10, 10] | 0 | |
| `Rosenbrock` | any | [-5, 10] | 0 | `rescaled` |

### Steep ridges and drops

| function | `d` | domain | minimum | parameters |
| --- | --- | --- | --- | --- |
| `DeJong5` | 2 | [-65.536, 65.536] | 0.998 | |
| `Easom` | 2 | [-100, 100] | -1 | |
| `Michalewicz` | any | [0, π] | grid-computed | `m` |

### Other

| function | `d` | domain | minimum | parameters |
| --- | --- | --- | --- | --- |
| `Beale` | 2 | [-4.5, 4.5] | 0 | |
| `Branin` | 2 | [-5, 10] × [0, 15] | 0.397887 | `a`, `b`, `c`, `r`, `s`, `t`, `rescaled`, `modified` |
| `Colville` | 4 | [-10, 10] | 0 | |
| `Forrester` | 1 | [0, 1] | -6.02074 | |
| `ForresterLowFidelity` | 1 | [0, 1] | 0.665095 | `A`, `B`, `C` |
| `GoldsteinPrice` | 2 | [-2, 2] | 3 | `rescaled` |
| `Hartmann3` | 3 | [0, 1] | -3.86278 | |
| `Hartmann4` | 4 | [0, 1] | -3.1345 | |
| `Hartmann6` | 6 | [0, 1] | -3.32237 | `rescaled` |
| `Perm` | any | [-d, d] | 0 | `beta` |
| `Powell` | any | [-4, 5] | 0 | |
| `Shekel` | 4 | [0, 10] | -10.5364 | `m` (5\|7\|10), `b`, `C` |
| `StyblinskiTang` | any | [-5, 5] | -39.166 d | |

### Deviations from the R sources

- The Hartmann 6-D pages are swapped upstream: `hart6r.R` computes Picheny's rescaled form and
  `hart6sc.R` the plain one. `Hartmann6()` is the plain form here; the parity table maps the two
  files crosswise.
- `shekelr.R` hardcodes `m <- 10`, so `Shekel(m=5)` and `Shekel(m=7)` are checked against an
  independent reference instead.
- `spherefmod.R` is published for d=6 only; `Sphere(rescaled=True)` generalizes it, reproducing
  the published constants at d=6.
- `Michalewicz.ymin` has no closed form; it is computed per-axis on a grid (the terms are
  separable) and cached.
- `Forrester.ymin` is the numerically computed `-6.020740055767083` rather than the page's
  rounded `-6.02074`, which would put the normalized form below 0.

## Optimizer

`vlse.optim.minimise` is L-BFGS-B (Byrd, Lu, Nocedal & Zhu 1995) written as a single
`jax.lax.while_loop`: a solve is one XLA dispatch, a multistart batch is a `vmap`, and the whole
thing jits, differentiates and runs on any backend.

```python
from vlse.optim import minimise

f = vlse.Ackley(d=20)
state = minimise(f, x0, bounds=f.domain, tol=1e-9)      # state.x, state.f, state.error, ...

starts = jr.uniform(jr.key(0), (256, 20), minval=lo, maxval=hi)
jax.jit(jax.vmap(lambda x0: minimise(f, x0, f.domain)))(starts)
```

The signature follows `scipy.optimize.minimize`: `minimise(fun, x0, bounds=None, args=(),
tol=1e-5, ftol=0.0, max_iterations=100, history_length=10, ...)`. `x0` may be any pytree —
the solve runs on the raveled vector and `x`/`grad` come back in `x0`'s structure, with
`bounds` mirroring it. Stopping is scipy's `pgtol` (plus `ftol` on relative decrease when
set), the line search follows scipy's `lnsrlb`/`dcsrch`, and every tolerance is read off the
working dtype, so f32 and f64 both run — each to the precision it has. Parity against scipy is
checked in f64, since scipy's Fortran has no single precision path.

The result is not byte-identical to scipy — XLA reassociates the arithmetic, and on a
multimodal box the last bit can decide a basin. The test suite (`tests/optim/`) therefore
solves all 48 functions from 32 shared starts in six modes (sequential / `lax.map` / `vmap`
× CPU / GPU): outside the multimodal set every start must land exactly where scipy lands;
on it, a Šidák-corrected one-sided Wilcoxon signed-rank must not show the solver landing in
worse minima than scipy.

## Benchmarks

Both benches sweep every GPU and CPU of one cluster over batch size and dimension, climbing
powers of two until the device gives out. Median per-call throughput, 95% interval shaded,
log-log; figures open hoverable.

Function evaluation (`Ackley`, f32; the
[f64 batch](https://raw.githack.com/davidesartor/vlse/main/bench/functions/scaling-batch-fp64.html)
and [f64 dim](https://raw.githack.com/davidesartor/vlse/main/bench/functions/scaling-dim-fp64.html)
panels run one to two orders slower on everything but the `hpc` parts):

[![evaluations per second against batch size, f32, one line per device](https://raw.githubusercontent.com/davidesartor/vlse/main/bench/functions/scaling-batch-fp32.svg)](https://raw.githack.com/davidesartor/vlse/main/bench/functions/scaling-batch-fp32.html)

[![evaluations per second against dimension, f32, one line per device](https://raw.githubusercontent.com/davidesartor/vlse/main/bench/functions/scaling-dim-fp32.svg)](https://raw.githack.com/davidesartor/vlse/main/bench/functions/scaling-dim-fp32.html)

L-BFGS-B, against scipy's Fortran on the same objective, starts and box (f32, stopping at the 1e-3
projected gradient that precision can resolve; the
[f64 batch](https://raw.githack.com/davidesartor/vlse/main/bench/optim/scaling-batch-fp64.html)
and [f64 dim](https://raw.githack.com/davidesartor/vlse/main/bench/optim/scaling-dim-fp64.html)
panels solve the same problem in double precision, to 1e-9):

[![solves per second against batch size, f32, one line per device](https://raw.githubusercontent.com/davidesartor/vlse/main/bench/optim/scaling-batch-fp32.svg)](https://raw.githack.com/davidesartor/vlse/main/bench/optim/scaling-batch-fp32.html)

[![solves per second against dimension, f32, one line per device](https://raw.githubusercontent.com/davidesartor/vlse/main/bench/optim/scaling-dim-fp32.svg)](https://raw.githack.com/davidesartor/vlse/main/bench/optim/scaling-dim-fp32.html)

Read the small end of either axis as latency, not throughput — JAX is dispatch-bound there.
What is measured, how, and where each curve stops: [bench/README.md](bench/README.md).

## Development

```bash
uv sync
uv run --no-sync pytest tests/functions -q       # ~15s, parity + API
uv run --no-sync pytest tests/optim -q           # ~2min, the solver against scipy
uv run --no-sync python tools/fetch_sfu.py       # refresh tools/sfu/*.R from the SFU site
```

Parity is checked live: `tests/functions/test_reference.py` shells out to `Rscript` against the
vendored sources — no frozen golden data. Without R on `PATH` that module skips.

## License

GPL-2.0-only, matching the SFU reference implementations this is derived from. See
[LICENSE](LICENSE) and [NOTICE](NOTICE).
