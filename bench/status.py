"""Terminal status board: per-config done/running/scheduled/missing as colour bars.

A task counts as done once its rows are in the results, or `sacct` saw it complete inside the
lookback.
"""

import argparse
import csv
import getpass
import glob
import importlib
import os
import re
import subprocess
import time

from .style import SEGMENTS
from .submit import SKIP_CHIPS

BENCHES = ("functions", "optim")
BAR_WIDTH = 24
COLORS = {
    "done": "\033[32m",
    "running": "\033[33m",
    "scheduled": "\033[34m",
    "missing": "\033[90m",
}
RESET = "\033[0m"
ARRAY_RANGE = re.compile(r"^\d+_\[(\d+)-(\d+)\]$")


def chip_universe() -> list[str]:
    chips = [c for segment in ("consumer", "server", "hpc") for c in SEGMENTS[segment]]
    return [c for c in chips if c not in SKIP_CHIPS]


def configs(bench) -> list[tuple[str, tuple[str, ...]]]:
    """(config name, its task configs in array-id order)."""
    widths = [(f"cpu{c}", bench.cpu_configs(int(c))) for c in bench.CORES.split(",")]
    return widths + [(chip, bench.GPU_CONFIGS) for chip in chip_universe()]


def device_of(config_name: str) -> str:
    """The results label a config writes under; a CPU width names itself by core count."""
    cores = config_name.removeprefix("cpu")
    return f"cpu-{cores}core" if cores != config_name else config_name


def landed(directory: str) -> set[tuple[str, str, str, int]]:
    """Every (label, sweep, dtype, point) that has a timing on disk."""
    rows = set()
    for path in glob.glob(os.path.join(directory, "*.csv")):
        label = os.path.basename(path).removesuffix(".csv")
        if label == "configs":
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                rows.add((label, row["sweep"], row["dtype"], int(row["size"])))
    return rows


def squeue_counts(user: str) -> dict[str, dict[str, int]]:
    """Live pending/running task counts per job name, array ranges expanded."""
    out = subprocess.run(
        ["squeue", "-u", user, "-h", "-o", "%i|%j|%t"], capture_output=True, text=True
    ).stdout
    counts: dict[str, dict[str, int]] = {}
    for line in out.splitlines():
        job_id, name, state = line.split("|")
        if state not in ("PD", "R"):
            continue
        if m := ARRAY_RANGE.match(job_id):
            n = int(m.group(2)) - int(m.group(1)) + 1
        else:
            n = 1
        bucket = counts.setdefault(name, {"PD": 0, "R": 0})
        bucket[state] += n
    return counts


def sacct_done(user: str, since: str) -> dict[str, set[int]]:
    """Completed array task ids per job name over the lookback window."""
    out = subprocess.run(
        [
            "sacct",
            "-a",
            "-u",
            user,
            "--starttime",
            since,
            "--format=JobID,JobName,State",
            "-P",
        ],
        capture_output=True,
        text=True,
    ).stdout
    done: dict[str, set[int]] = {}
    for line in out.splitlines()[1:]:
        parts = line.split("|")
        if len(parts) != 3:
            continue
        job_id, name, state = parts
        task = job_id.split("_", 1)[1] if "_" in job_id else ""
        if not task.isdigit() or state != "COMPLETED":
            continue
        done.setdefault(name, set()).add(int(task))
    return done


def bar(done: int, running: int, scheduled: int, missing: int, total: int) -> str:
    if total == 0:
        return " " * BAR_WIDTH
    remaining = BAR_WIDTH
    out = []
    for key, n in (
        ("done", done),
        ("running", running),
        ("scheduled", scheduled),
        ("missing", missing),
    ):
        width = min(round(n / total * BAR_WIDTH), remaining)
        out.append(f"{COLORS[key]}{'█' * width}{RESET}")
        remaining -= width
    return "".join(out)


def render(user: str, since: str) -> str:
    pending_running = squeue_counts(user)
    completed = sacct_done(user, since)
    lines = [
        f"{'config':<10} {'bar':<{BAR_WIDTH}}  done running sched missing total",
        f"{'':10} {COLORS['done']}done{RESET} {COLORS['running']}running{RESET} "
        f"{COLORS['scheduled']}scheduled{RESET} {COLORS['missing']}missing{RESET}",
    ]
    grand = {"done": 0, "running": 0, "pending": 0, "missing": 0, "total": 0}
    for bench_name in BENCHES:
        bench = importlib.import_module(f".{bench_name}", __package__)
        results = landed(os.path.join(os.path.dirname(bench.__file__), "results"))
        lines.append(f"\n{bench_name}")
        totals = {"done": 0, "running": 0, "pending": 0, "missing": 0, "total": 0}
        for name, tasks in configs(bench):
            job = f"{bench_name}-{name}"
            total = len(tasks)
            pending = pending_running.get(job, {}).get("PD", 0)
            running = pending_running.get(job, {}).get("R", 0)
            device = device_of(name)
            # a task is done if it left rows, or sacct still remembers it completing
            written = {
                task
                for task, config in enumerate(tasks, 1)
                if bench.task_key(device, config) in results
            }
            done = min(len(written | completed.get(job, set())), total)
            missing = max(total - done - running - pending, 0)
            lines.append(
                f"  {name:<10} {bar(done, running, pending, missing, total)}  "
                f"{done:>4} {running:>7} {pending:>5} {missing:>7} {total:>5}"
            )
            for key, n in (
                ("done", done),
                ("running", running),
                ("pending", pending),
                ("missing", missing),
                ("total", total),
            ):
                totals[key] += n
                grand[key] += n
        lines.append(
            f"  {'total':<10} {bar(totals['done'], totals['running'], totals['pending'], totals['missing'], totals['total'])}  "
            f"{totals['done']:>4} {totals['running']:>7} {totals['pending']:>5} {totals['missing']:>7} {totals['total']:>5}"
        )
    lines.append(
        f"\n{'grand total':<10} {bar(grand['done'], grand['running'], grand['pending'], grand['missing'], grand['total'])}  "
        f"{grand['done']:>4} {grand['running']:>7} {grand['pending']:>5} {grand['missing']:>7} {grand['total']:>5}"
    )
    return "\n".join(lines)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--watch", action="store_true", help="clear and refresh in place")
    p.add_argument(
        "--interval", type=float, default=10.0, help="seconds between refreshes"
    )
    p.add_argument(
        "--since",
        default="now-2days",
        help="sacct lookback for completed tasks, a Slurm --starttime value",
    )


def run(args) -> None:
    user = getpass.getuser()
    if not args.watch:
        print(render(user, args.since))
        return
    try:
        while True:
            print("\033[2J\033[H" + render(user, args.since), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
