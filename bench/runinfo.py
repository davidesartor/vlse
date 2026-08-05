"""results/ bookkeeping shared by both benches: the constants table, and the row every job appends.

Both benches write the same two files -- `configs.csv` holding what was constant for a run, and one
`<stem>.csv` per device whose rows are the sweeps that landed on it -- so the machinery lives here
and each bench points a `Results` at its own directory.
"""

import csv
import fcntl
import os
import platform
import socket
import subprocess
import sys

# the columns both benches share, in the order they read best; anything else a bench records
# follows them, in the order it first appeared
COMMON_FIELDS = "file label device machine dtype host fn d batch version".split()

HOST = socket.gethostname()
# what identifies this job's row: the Slurm job it is an array task of, or host and pid off cluster
RUN = os.environ.get("SLURM_JOB_ID") or f"{HOST}-{os.getpid()}"


def cpu_model() -> str:
    """Best-effort CPU name, for labelling the machine a CPU run happened on."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if out.returncode == 0:
            return out.stdout.strip()
    return platform.processor() or platform.machine()


def device_label(device) -> str:
    """<platform>-<device kind>, lowered into something that can sit in a file name."""
    return f"{device.platform}-{device.device_kind}".lower().replace(" ", "_")


class Results:
    """One bench's results directory."""

    def __init__(self, directory: str):
        self.directory = directory
        self.configs = os.path.join(directory, "configs.csv")

    def read(self) -> dict[str, dict[str, str]]:
        """File stem -> its constants. Empty when no run has been recorded yet."""
        if not os.path.exists(self.configs):
            return {}
        with open(self.configs) as fh:
            return {r["file"]: r for r in csv.DictReader(fh)}

    def write(self, stem: str, label: str, device, **fields: object) -> None:
        """Upsert one run's constants, rewriting the table so a rerun replaces its own row.

        Locked like the row append: every task of every array writes this same file.
        """
        run = dict(
            label=label,
            device=device.device_kind,
            machine=cpu_model() if device.platform == "cpu" else device.device_kind,
            host=HOST,
            **fields,
        )
        os.makedirs(self.directory, exist_ok=True)
        with open(self.configs, "a+", newline="") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            configs = {r["file"]: r for r in csv.DictReader(fh)}
            configs[stem] = {"file": stem, **{k: str(v) for k, v in run.items()}}
            seen = [
                k for row in configs.values() for k in row if k not in COMMON_FIELDS
            ]
            fh.seek(0)
            fh.truncate()
            w = csv.DictWriter(
                fh, COMMON_FIELDS + list(dict.fromkeys(seen)), restval=""
            )
            w.writeheader()
            w.writerows(configs[key] for key in sorted(configs))

    def append(self, stem: str, seconds: dict[int, float]) -> str:
        """Append this run's sweep as one row, under a lock: parallel jobs share the one file.

        Rewrites rather than appends bytes because a job that reached further than the last one
        widens the table, and the column set is the union over every row.
        """
        path = os.path.join(self.directory, f"{stem}.csv")
        os.makedirs(self.directory, exist_ok=True)
        with open(path, "a+", newline="") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            rows = list(csv.DictReader(fh))
            rows.append(
                {"run": RUN, **{str(n): f"{s:.9g}" for n, s in seconds.items()}}
            )
            sizes = sorted({int(col) for r in rows for col in r if col != "run"})
            fh.seek(0)
            fh.truncate()
            w = csv.DictWriter(fh, ["run", *(str(n) for n in sizes)], restval="")
            w.writeheader()
            w.writerows(rows)
        return path
