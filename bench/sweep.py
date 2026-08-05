"""One power-of-two point of the axis both benches sweep, and its timing."""

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

Setup = Callable[[int, int], tuple[Callable[[], object], Callable[[], None]]]
"""(size, seed) -> (timed call, release)."""


@dataclass
class Run:
    """One sweep of one bench: what to time at each point, and its constants."""

    label: str
    device: object
    sweep: str
    setup: Setup
    work_at: Callable[[int], float]
    constants: dict[str, object] = field(default_factory=dict)


def block(call, calls: int) -> float:
    """Time `calls` calls as one block, so scheduler jitter averages down inside it."""
    t0 = time.perf_counter()
    for _ in range(calls):
        call()
    return time.perf_counter() - t0


def calls_per_block(call, target: float) -> int:
    """Calls per block, enough that one block clears the timer and scheduler jitter floor."""
    calls = 1
    while calls < 1 << 16:
        elapsed = block(call, calls)
        if elapsed >= target:
            return calls
        calls = max(calls * 2, int(calls * target / max(elapsed, 1e-9)))
    return calls


def time_one(
    setup: Setup, size: int, seed: int, work: float, unit: str, args
) -> float | None:
    """Median seconds per call at one point, or None if the device cannot hold it."""
    release, times, calls = None, None, None
    try:
        call, release = setup(size, seed)
        block(call, args.warmup)
        calls = calls_per_block(call, args.block_seconds)
        times = [block(call, calls) / calls for _ in range(args.reps)]
    except Exception as err:
        print(f"{size}: {type(err).__name__}", flush=True)
    finally:
        if release is not None:
            release()

    if times is None:
        return None
    median = statistics.median(times)
    spread = statistics.stdev(times) / median if len(times) > 1 else 0.0
    print(
        f"{size}#{seed}: {args.reps} blocks of {calls}, {work / median:.4g} {unit}, rsd {spread:.1%}",
        flush=True,
    )
    return median


def point(run: Run, unit: str, args) -> list[dict[int, float]]:
    """Every repeat of `2**args.exponent`, one row each."""
    size = 2**args.exponent
    rows: list[dict[int, float]] = []
    for seed in range(args.repeats):
        median = time_one(run.setup, size, seed, run.work_at(size), unit, args)
        if median is None:
            break
        rows.append({size: median})
        if median > args.max_seconds:
            print(f"{size}: {median:.1f}s a call, stopping", flush=True)
            break
    return rows


def add_arguments(p: argparse.ArgumentParser, max_seconds: float = 10.0) -> None:
    p.add_argument(
        "--repeats", type=int, default=12, help="sequential runs of this point"
    )
    p.add_argument(
        "--reps", type=int, default=5, help="timed blocks per repeat, median kept"
    )
    p.add_argument(
        "--block-seconds", type=float, default=0.05, help="duration of one timed block"
    )
    p.add_argument(
        "--warmup", type=int, default=5, help="untimed calls before each sweep point"
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=max_seconds,
        help="stop repeating once a call costs this",
    )
    p.add_argument("--exponent", type=int, default=0, help="the point run: 2**this")
