"""Run the correctness suite on a few GPU chips, since the GPU modes only run where a GPU is.

Not every chip the cluster has: the ones whose compute capability or CPU architecture sends XLA
down its own code path, plus a modern baseline to read the rest against.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# cluster knowledge rather than bench knowledge, but it lives with the bench that needed it first
sys.path.insert(0, os.path.join(ROOT, "bench"))
from submit import gpu_chips, sbatch  # noqa: E402

# the chips that have gone wrong on a bench run, which is what earns a chip a place here: titan_x
# fails the optim sweep outright (cuBLASLt gemms on sm_52), 1080_ti has thrown on one. Add a chip
# when it breaks; the rest are covered by CI and by the benches themselves
DEFAULT_CHIPS = ("titan_x", "1080_ti")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--chips",
        default=",".join(DEFAULT_CHIPS),
        help=f"comma separated chips to test, default {','.join(DEFAULT_CHIPS)}",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="every chip the cluster offers instead, which is a wide ask on a busy queue",
    )
    p.add_argument(
        "--cores",
        type=int,
        default=16,
        help="cores to ask for; one per xdist worker, which is where the compiles parallelize",
    )
    p.add_argument(
        "--pytest-args",
        default="-rf tests/optim",
        help="passed through to pytest, target included; tests/optim is the half with a GPU mode",
    )
    # the suite plus a venv build and every compile, with headroom for the slower chips; a job
    # this short still backfills
    p.add_argument("--time", default="01:00:00")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logs = os.path.join(HERE, "logs")
    os.makedirs(logs, exist_ok=True)

    available = gpu_chips()
    wanted = list(available) if args.all else args.chips.split(",")
    for chip in wanted:
        if chip not in available:
            print(f"# no {chip} on this cluster, skipping")
            continue
        # through the environment rather than `--export=ALL,PYTEST_ARGS=...`, which sbatch splits
        # on commas and would cut a pytest `-k` expression in half
        os.environ["PYTEST_ARGS"] = args.pytest_args
        sbatch(
            [
                "sbatch",
                "--chdir",
                ROOT,
                "--export=ALL",
                "-p",
                ",".join(available[chip]),
                "--gres=gpu:1",
                f"--constraint={chip}",
                "-c",
                str(args.cores),
                "--mem",
                "48G",
                "-t",
                args.time,
                "-J",
                f"vlse-test-{chip}",
                "-o",
                os.path.join(logs, f"tests-{chip}.out"),
                "tools/slurm/pytest_gpu.sh",
                chip,
            ],
            args.dry_run,
        )


if __name__ == "__main__":
    main()
