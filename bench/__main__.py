"""One entry point for both benches: `python -m bench <functions|optim> <run|plot|submit>`."""

import argparse
import importlib
import os

from . import status, submit, sweep
from .plot import figure
from .results import RUN, Results

BENCHES = ("dispatch", "functions", "optim")


def directory(bench, *parts: str) -> str:
    return os.path.join(os.path.dirname(bench.__file__), *parts)


def run(args) -> None:
    """One point, `--repeats` rows: every array task runs one size of one config."""
    sweeping = args.bench.prepare(args)
    results = Results(directory(args.bench, "results"))
    results.write(sweeping.label, sweeping.sweep, sweeping.device, **sweeping.constants)

    rows = sweep.point(sweeping, args.bench.UNIT, args)
    if not rows:
        raise SystemExit(f"2**{args.exponent} does not fit on {sweeping.label}")
    dtype = sweeping.constants["dtype"]
    path = results.append(sweeping.label, sweeping.sweep, dtype, rows)
    print(f"wrote {len(rows)} rows as {RUN} to {path}")


def plot(args) -> None:
    for page in args.bench.pages(args, directory(args.bench, "results")):
        figure(page, directory(args.bench, page.name))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bench", description=__doc__)
    benches = parser.add_subparsers(required=True)
    for name in BENCHES:
        bench = importlib.import_module(f".{name}", __package__)
        commands = benches.add_parser(
            name, help=bench.__doc__.splitlines()[0]
        ).add_subparsers(required=True)
        bench.add_run_arguments(
            sweeper := commands.add_parser("run", description=bench.__doc__)
        )
        sweep.add_arguments(sweeper)
        bench.add_plot_arguments(commands.add_parser("plot"))
        submit.add_arguments(
            commands.add_parser("submit"), bench.CORES, bench.REPEATS, bench.WALLTIME
        )

        for command, action in (("run", run), ("plot", plot), ("submit", submit.run)):
            commands.choices[command].set_defaults(
                bench=bench, bench_name=name, action=action
            )

    status.add_arguments(
        benches.add_parser("status", help="terminal sweep status board")
    )
    benches.choices["status"].set_defaults(action=status.run)

    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
