"""L-BFGS-B: limited-memory BFGS under box constraints (Byrd, Lu, Nocedal & Zhu 1995)."""

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve
from jaxtyping import Array, Bool, Float, Int, Scalar

MAX_STEP = 1e10  # scipy's `big` in lnsrlb, the step cap when nothing bounds the ray
EXPANSION = 4.0  # scipy's xtrapu in dcsrch


class LBFGSBState(NamedTuple):
    x: Float[Array, "p"]
    f: Scalar
    grad: Float[Array, "p"]
    s_history: Float[Array, "m p"]
    y_history: Float[Array, "m p"]
    theta: Scalar
    n_updates: Int[Array, ""]
    iteration: Int[Array, ""]
    n_fun_eval: Int[Array, ""]
    error: Scalar
    failed_linesearch: Bool[Array, ""]


class _Hessian(NamedTuple):
    """Compact form B = theta*I - W @ M @ W.T (eqs. 3.2-3.4), M held as an LU factorization."""

    theta: Scalar
    w: Float[Array, "p l"]
    m: Float[Array, "l l"]
    m_lu: tuple[Float[Array, "l l"], Int[Array, "l"]]

    def apply_m(self, v: Float[Array, "l ..."]) -> Float[Array, "l ..."]:
        return lu_solve(self.m_lu, v)


class _CauchyPoint(NamedTuple):
    x: Float[Array, "p"]
    c: Float[Array, "l"]  # W.T @ (x_cauchy - x), reused by the subspace step
    free: Bool[Array, "p"]  # variables not driven onto a bound by the Cauchy path


class _Endpoint(NamedTuple):
    """One end of the line-search bracket: the step and everything evaluated at it."""

    alpha: Scalar
    f: Scalar
    df: Scalar
    x: Float[Array, "p"]
    grad: Float[Array, "p"]


def projected_gradient_norm(
    x: Float[Array, "p"],
    grad: Float[Array, "p"],
    lower: Float[Array, "p"],
    upper: Float[Array, "p"],
) -> Scalar:
    """Infinity norm of the projected gradient, the same criterion as scipy's pgtol."""
    return jnp.max(jnp.abs(jnp.clip(x - grad, lower, upper) - x))


def _compact_hessian(state: LBFGSBState) -> _Hessian:
    """W and M of the compact representation, eqs. 3.3-3.4."""
    s_history, y_history, theta = state.s_history, state.y_history, state.theta
    history_length = s_history.shape[0]

    s_dot_y = s_history @ y_history.T
    s_dot_s = s_history @ s_history.T
    # unfilled slots get diagonal filler so M stays invertible; their W columns are zero
    slots = jnp.arange(history_length)
    empty = slots < history_length - state.n_updates
    unfilled = jnp.diag(empty.astype(s_dot_y.dtype))
    newest_sy = jnp.where(state.n_updates > 0, s_dot_y[-1, -1], 1.0)
    newest_ss = jnp.where(state.n_updates > 0, s_dot_s[-1, -1], 1.0)
    d_block = -jnp.diag(jnp.diag(s_dot_y)) + unfilled * newest_sy
    l_block = jnp.tril(s_dot_y, -1)
    ss_block = theta * (s_dot_s + unfilled * newest_ss)
    m = jnp.block([[d_block, l_block.T], [l_block, ss_block]])
    w = jnp.concatenate([y_history.T, theta * s_history.T], axis=1)
    return _Hessian(theta, w, m, lu_factor(m))


def _cauchy_point(
    x: Float[Array, "p"],
    grad: Float[Array, "p"],
    lower: Float[Array, "p"],
    upper: Float[Array, "p"],
    hessian: _Hessian,
    eps: float,
) -> _CauchyPoint:
    """Generalized Cauchy point (Algorithm CP): first minimiser along the projected gradient path."""
    theta, w = hessian.theta, hessian.w

    # each coordinate's bound hit, in crossing order
    bound_gap = jnp.where(grad < 0.0, x - upper, x - lower)
    breakpoint_of = jnp.where(jnp.abs(grad) < eps, jnp.inf, bound_gap / grad)
    direction = jnp.where(breakpoint_of < eps, 0.0, -grad)
    x_bound = jnp.where(direction > 0.0, upper, jnp.where(direction < 0.0, lower, x))
    order = jnp.argsort(breakpoint_of, axis=-1)
    breakpoints = breakpoint_of[order]
    widths = jnp.diff(jnp.pad(breakpoints, (1, 0), "constant"))

    class Walk(NamedTuple):
        """Path quadratic at the current segment: c = W.T @ (x_path - x), p = W.T @ d."""

        crossed: Int[Array, ""]
        df: Scalar
        ddf: Scalar
        c: Float[Array, "l"]
        p: Float[Array, "l"]

    def keep_crossing(walk: Walk) -> Bool[Array, ""]:
        return (-walk.df / walk.ddf >= widths[walk.crossed]) & (
            walk.crossed < x.shape[-1]
        )

    def cross_breakpoint(walk: Walk) -> Walk:
        j = order[walk.crossed]
        width = widths[walk.crossed]
        c = walk.c + width * walk.p
        # M is symmetric, so one solve against w[j] serves every quadratic term
        wj_m = hessian.apply_m(w[j])
        df = (
            walk.df
            + width * walk.ddf
            + grad[j] ** 2
            + theta * grad[j] * (x_bound[j] - x[j])
            - grad[j] * jnp.dot(wj_m, c)
        )
        ddf = (
            walk.ddf
            - theta * grad[j] ** 2
            - 2.0 * grad[j] * jnp.dot(wj_m, walk.p)
            - grad[j] ** 2 * jnp.dot(wj_m, w[j])
        )
        return Walk(
            walk.crossed + 1, df, jnp.maximum(eps, ddf), c, walk.p + grad[j] * w[j]
        )

    # few breakpoints are ever crossed, so the sequential walk beats vectorizing (measured on GPU)
    start = jnp.argmax(jnp.concatenate([breakpoints > eps, jnp.ones([1], dtype=bool)]))
    p0 = w.T @ direction
    df0 = -jnp.dot(direction, direction)
    ddf0 = -theta * df0 - jnp.dot(hessian.apply_m(p0), p0)
    c0 = jnp.zeros(w.shape[-1:], dtype=w.dtype)
    walk = jax.lax.while_loop(
        keep_crossing, cross_breakpoint, Walk(start, df0, ddf0, c0, p0)
    )

    # the stop segment holds its own minimiser; everything crossed before it sits on a bound
    final_width = jnp.nan_to_num(jnp.maximum(-walk.df / walk.ddf, 0.0), nan=0.0)
    last_breakpoint = breakpoints[jnp.maximum(walk.crossed - 1, 0)]
    t = jnp.where(walk.crossed > 0, last_breakpoint, 0.0) + final_width
    c = walk.c + final_width * walk.p
    # scatter back through the permutation rather than argsort it (GPU scatter restriction)
    free_sorted = jnp.arange(x.shape[-1]) >= walk.crossed
    free = jnp.zeros_like(free_sorted).at[order].set(free_sorted)
    x_cauchy = jnp.where(free, x + t * direction, x_bound)
    return _CauchyPoint(x_cauchy, c, free)


def _subspace_minimum(
    x: Float[Array, "p"],
    grad: Float[Array, "p"],
    lower: Float[Array, "p"],
    upper: Float[Array, "p"],
    hessian: _Hessian,
    cauchy: _CauchyPoint,
) -> Float[Array, "p"]:
    """Direct primal minimisation over the free variables, eqs. 5.4-5.11."""
    theta, w = hessian.theta, hessian.w

    # Newton step, eqs. 5.10-5.11 with the nested Woodbury solves collapsed into one
    w_free = jnp.where(cauchy.free[:, jnp.newaxis], w, 0.0)
    residual = grad + theta * (cauchy.x - x) - w_free @ hessian.apply_m(cauchy.c)
    reduced = hessian.m - w_free.T @ w_free / theta
    correction = w_free @ jnp.linalg.solve(reduced, w_free.T @ residual) / theta**2
    newton = -residual / theta - correction

    # truncate the step at the box, eq. 5.8
    moving = cauchy.free & (jnp.abs(newton) > 0.0)
    safe_newton = jnp.where(jnp.abs(newton) > 0.0, newton, 1.0)
    to_bound = jnp.maximum(
        (upper - cauchy.x) / safe_newton, (lower - cauchy.x) / safe_newton
    )
    scale = jnp.minimum(jnp.min(jnp.where(moving, to_bound, 1.0), axis=-1), 1.0)
    return jnp.where(cauchy.free, cauchy.x + scale * newton, cauchy.x)


def _step_sizes(
    x: Float[Array, "p"],
    direction: Float[Array, "p"],
    lower: Float[Array, "p"],
    upper: Float[Array, "p"],
    first_iteration: Bool[Array, ""],
) -> tuple[Scalar, Scalar]:
    """How long the first trial step is and how far the search ray may run, scipy's lnsrlb."""
    constrained = jnp.any(jnp.isfinite(lower) | jnp.isfinite(upper))
    boxed = jnp.all(jnp.isfinite(lower) & jnp.isfinite(upper))

    # longest step that stays in the box
    advancing = direction != 0.0
    gap = jnp.where(direction > 0.0, upper - x, lower - x)
    safe_direction = jnp.where(advancing, direction, 1.0)
    feasible = jnp.min(jnp.where(advancing, gap / safe_direction, jnp.inf))

    # a constrained problem never overshoots its own subspace minimiser on the first iteration
    max_step = jnp.where(
        constrained, jnp.where(first_iteration, 1.0, feasible), MAX_STEP
    )
    max_step = jnp.clip(max_step, 0.0, MAX_STEP)
    unit_step = 1.0 / jnp.linalg.norm(direction)
    initial_step = jnp.where(first_iteration & ~boxed, unit_step, 1.0)
    return jnp.minimum(initial_step, max_step), max_step


def _wolfe_search(
    evaluate: Callable[[Scalar], _Endpoint],
    start: _Endpoint,
    initial_step: Scalar,
    max_step: Scalar,
    max_evaluations: int | Int[Array, ""],
    c1: float | Float[Array, ""],
    c2: float | Float[Array, ""],
    eps: float,
) -> tuple[_Endpoint, Int[Array, ""]]:
    """Strong-Wolfe search, Nocedal & Wright Alg. 3.5-3.6 as one loop.

    Expansion pushes the step out until the minimum is bracketed, zoom then shrinks
    the bracket around it. Returns the best endpoint and the evaluation count.
    """

    class Search(NamedTuple):
        n_eval: Int[Array, ""]
        trial: Scalar
        best: _Endpoint  # the low end: the best point seen, which the bracket keeps
        hi: Scalar
        f_hi: Scalar
        df_hi: Scalar
        bracketed: Bool[Array, ""]
        done: Bool[Array, ""]

    def searching(search: Search) -> Bool[Array, ""]:
        return (search.n_eval < max_evaluations) & ~search.done

    def refine(search: Search) -> Search:
        # zoom's trial, scipy's dcstep: the Hermite cubic minimiser when it sits closer to the
        # best end than the quadratic's, otherwise their midpoint; bisection if degenerate
        width = search.hi - search.best.alpha
        safe_width = jnp.where(width != 0.0, width, 1.0)
        d1 = (
            search.best.df
            + search.df_hi
            + 3.0 * (search.best.f - search.f_hi) / safe_width
        )
        radicand = d1 * d1 - search.best.df * search.df_hi
        d2 = jnp.sign(width) * jnp.sqrt(jnp.maximum(radicand, 0.0))
        cubic_denominator = search.df_hi - search.best.df + 2.0 * d2
        cubic = search.hi - width * (search.df_hi + d2 - d1) / jnp.where(
            cubic_denominator != 0.0, cubic_denominator, 1.0
        )
        cubic_fraction = jnp.where(
            (radicand >= 0.0) & (cubic_denominator != 0.0),
            (cubic - search.best.alpha) / safe_width,
            0.5,
        )
        quad_denominator = 2.0 * (search.f_hi - search.best.f - search.best.df * width)
        quad_fraction = jnp.where(
            quad_denominator > 0.0, -search.best.df * width / quad_denominator, 0.5
        )
        fraction = jnp.where(
            jnp.abs(cubic_fraction) < jnp.abs(quad_fraction),
            cubic_fraction,
            0.5 * (cubic_fraction + quad_fraction),
        )
        fraction = jnp.clip(jnp.nan_to_num(fraction, nan=0.5), 0.1, 0.9)
        inside = search.best.alpha + fraction * width
        trial = evaluate(jnp.where(search.bracketed, inside, search.trial))

        # expansion: an Armijo violation closes the bracket (negated so a non-finite value
        # counts as a violation), a still-negative slope pushes the step out further
        overshoots = ~(trial.f <= start.f + c1 * trial.alpha * start.df) | (
            (trial.f >= search.best.f) & (search.n_eval > 0)
        )
        turned = trial.df >= 0.0
        satisfied = jnp.abs(trial.df) <= -c2 * start.df
        stalled = trial.alpha >= max_step
        closed = search._replace(
            hi=trial.alpha, f_hi=trial.f, df_hi=trial.df, bracketed=jnp.array(True)
        )
        advanced = search._replace(
            best=trial,
            hi=search.best.alpha,
            f_hi=search.best.f,
            df_hi=search.best.df,
            bracketed=turned & ~satisfied & ~stalled,
            done=satisfied | stalled,
        )
        expanded = jax.tree.map(
            lambda a, b: jnp.where(overshoots, a, b), closed, advanced
        )
        expanded = expanded._replace(
            trial=jnp.minimum(EXPANSION * trial.alpha, max_step)
        )

        # zoom: an improving trial becomes the new best end, a failing one the new far end;
        # a bracket whose ends are indistinguishable in the working dtype cannot shrink further
        improves = (trial.f <= start.f + c1 * trial.alpha * start.df) & (
            trial.f < search.best.f
        )
        flip = trial.df * (search.hi - search.best.alpha) >= 0.0
        exhausted = jnp.abs(trial.f - search.best.f) <= eps * (
            1.0 + jnp.abs(search.best.f)
        )
        accepted = search._replace(
            best=trial,
            hi=jnp.where(flip, search.best.alpha, search.hi),
            f_hi=jnp.where(flip, search.best.f, search.f_hi),
            df_hi=jnp.where(flip, search.best.df, search.df_hi),
            done=jnp.abs(trial.df) <= -c2 * start.df,
        )
        rejected = search._replace(
            hi=trial.alpha, f_hi=trial.f, df_hi=trial.df, done=exhausted
        )
        shrunk = jax.tree.map(
            lambda a, b: jnp.where(improves, a, b), accepted, rejected
        )
        shrunk = shrunk._replace(trial=trial.alpha)

        chosen = jax.tree.map(
            lambda a, b: jnp.where(search.bracketed, a, b), shrunk, expanded
        )
        return chosen._replace(n_eval=search.n_eval + 1)

    init = Search(
        n_eval=jnp.array(0),
        trial=initial_step,
        best=start,
        hi=max_step,
        f_hi=start.f,
        df_hi=start.df,
        bracketed=jnp.array(False),
        done=jnp.array(False),
    )
    search = jax.lax.while_loop(searching, refine, init)
    return search.best, search.n_eval


def _update_memory(
    state: LBFGSBState, s: Float[Array, "p"], y: Float[Array, "p"], eps: float
) -> LBFGSBState:
    """History update, scipy's matupd: a pair too flat to trust would corrupt the whole memory."""
    curvature = jnp.dot(y, s)
    keep = curvature > eps * jnp.dot(y, y)
    push = lambda history, new: jnp.where(
        keep, jnp.roll(history, -1, axis=0).at[-1].set(new), history
    )
    return state._replace(
        s_history=push(state.s_history, s),
        y_history=push(state.y_history, y),
        theta=jnp.where(keep, jnp.dot(y, y) / curvature, state.theta),
        n_updates=jnp.where(
            keep,
            jnp.minimum(state.n_updates + 1, state.s_history.shape[0]),
            state.n_updates,
        ),
    )


def minimise(
    fun: Callable[..., Scalar],
    x0: Float[Array, "p"],
    bounds: Optional[tuple[Float[Array, "p"], Float[Array, "p"]]] = None,
    args: tuple = (),
    tol: float | Float[Array, ""] = jnp.array(1e-5),
    max_iterations: int | Int[Array, ""] = jnp.array(100),
    history_length: int = 10,
    max_linesearch: int | Int[Array, ""] = jnp.array(30),
    c1: float | Float[Array, ""] = jnp.array(1e-4),
    c2: float | Float[Array, ""] = jnp.array(0.9),
) -> LBFGSBState:
    """Minimise `fun(x, *args)` over a box, stopping when the projected gradient is below tol.

    `args` is scipy's: a tuple of extras held fixed, differentiated through but not with respect to.
    """
    lower, upper = (
        (-jnp.inf * jnp.ones_like(x0), jnp.inf * jnp.ones_like(x0))
        if bounds is None
        else bounds
    )
    lower, upper = jnp.broadcast_to(lower, x0.shape), jnp.broadcast_to(upper, x0.shape)

    def value_and_grad(x):
        return jax.value_and_grad(fun)(x, *args)

    x = jnp.clip(x0, lower, upper)
    f, grad = value_and_grad(x)
    eps = float(jnp.finfo(x.dtype).eps)
    init = LBFGSBState(
        x=x,
        f=f,
        grad=grad,
        s_history=jnp.zeros((history_length, x.shape[0]), dtype=x.dtype),
        y_history=jnp.zeros((history_length, x.shape[0]), dtype=x.dtype),
        theta=jnp.array(1.0, dtype=x.dtype),
        n_updates=jnp.array(0),
        iteration=jnp.array(0),
        n_fun_eval=jnp.array(1),
        error=projected_gradient_norm(x, grad, lower, upper),
        failed_linesearch=jnp.array(False),
    )

    def unconverged(state: LBFGSBState) -> Bool[Array, ""]:
        return (
            (state.error > tol)
            & (state.iteration < max_iterations)
            & ~state.failed_linesearch
        )

    def step(state: LBFGSBState) -> LBFGSBState:
        x, f, grad = state.x, state.f, state.grad
        hessian = _compact_hessian(state)
        cauchy = _cauchy_point(x, grad, lower, upper, hessian, eps)
        x_subspace = _subspace_minimum(x, grad, lower, upper, hessian, cauchy)

        direction = x_subspace - x
        initial_step, max_step = _step_sizes(
            x, direction, lower, upper, state.iteration == 0
        )

        def evaluate(alpha: Scalar) -> _Endpoint:
            x_trial = jnp.clip(x + alpha * direction, lower, upper)
            f_trial, grad_trial = value_and_grad(x_trial)
            return _Endpoint(
                alpha, f_trial, jnp.dot(grad_trial, direction), x_trial, grad_trial
            )

        start = _Endpoint(
            jnp.zeros((), dtype=x.dtype), f, jnp.dot(grad, direction), x, grad
        )
        best, n_eval = _wolfe_search(
            evaluate, start, initial_step, max_step, max_linesearch, c1, c2, eps
        )
        # an alpha too short to move x cannot improve: s and y stay zero, so the memory skips
        # the pair and the next iteration recomputes this one exactly, to the iteration cap
        failed = (best.alpha == 0.0) | jnp.all(best.x == x)

        state = _update_memory(state, best.x - x, best.grad - grad, eps)
        return state._replace(
            x=best.x,
            f=best.f,
            grad=best.grad,
            iteration=state.iteration + 1,
            n_fun_eval=state.n_fun_eval + n_eval,
            error=projected_gradient_norm(best.x, best.grad, lower, upper),
            failed_linesearch=failed,
        )

    return jax.lax.while_loop(unconverged, step, init)
