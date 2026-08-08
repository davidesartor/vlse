"""The results directory a bench writes: a table of run constants, and one row per timing."""

import csv
import fcntl
import os
import platform
import socket
import subprocess
import sys

COMMON_FIELDS = "label sweep dtype device machine host fn d batch version".split()
ROW_FIELDS = "run repeat sweep dtype size seconds".split()
CONFIG_KEY = ("label", "sweep", "dtype")

HOST = socket.gethostname()
RUN = os.environ.get("SLURM_JOB_ID") or f"{HOST}-{os.getpid()}"


def cpu_model() -> str:
    """Best-effort CPU name, for labelling the machine a CPU run happened on."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
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


def config_key(row: dict) -> tuple[str, ...]:
    return tuple(row[field] for field in CONFIG_KEY)


def timings(path: str) -> list[dict[str, str]]:
    """Timing rows of one label's file, tolerating a header repeated by a concurrent writer."""
    with open(path) as fh:
        return [row for row in csv.DictReader(fh) if row["size"] != "size"]


def device_label(device) -> str:
    """<platform>-<device kind>, lowered into something that can sit in a file name."""
    return f"{device.platform}-{device.device_kind}".lower().replace(" ", "_")


class Results:
    """One bench's results directory."""

    def __init__(self, directory: str):
        self.directory = directory
        self.configs = os.path.join(directory, "configs.csv")

    def write(self, label: str, sweep: str, device, **fields: object) -> None:
        """Upsert one run's constants, rewriting the table so a rerun replaces its own row."""
        run = dict(
            label=label,
            sweep=sweep,
            device=device.device_kind,
            machine=cpu_model() if device.platform == "cpu" else device.device_kind,
            host=HOST,
            **fields,
        )
        os.makedirs(self.directory, exist_ok=True)
        # locked: every task of every array writes this same file
        with open(self.configs, "a+", newline="") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            configs = {config_key(r): r for r in csv.DictReader(fh)}
            row = {k: str(v) for k, v in run.items()}
            configs[config_key(row)] = row
            extra = [
                k for row in configs.values() for k in row if k not in COMMON_FIELDS
            ]
            fh.seek(0)
            fh.truncate()
            w = csv.DictWriter(
                fh, COMMON_FIELDS + list(dict.fromkeys(extra)), restval=""
            )
            w.writeheader()
            w.writerows(configs[key] for key in sorted(configs))

    def append(
        self, label: str, sweep: str, dtype: str, runs: list[dict[int, float]]
    ) -> str:
        """Append this job's repeats, one row per timing, under a lock on the label's own file."""
        path = os.path.join(self.directory, f"{label}.csv")
        os.makedirs(self.directory, exist_ok=True)
        with open(path, "a+", newline="") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            # a read under the lock, not a cached st_size: NFS can report this file empty
            # to a second node while the first node's header is still only server-side
            fh.seek(0)
            already_headed = bool(fh.readline())
            fh.seek(0, os.SEEK_END)
            w = csv.DictWriter(fh, ROW_FIELDS)
            if not already_headed:
                w.writeheader()
            w.writerows(
                {
                    "run": RUN,
                    "repeat": repeat,
                    "sweep": sweep,
                    "dtype": dtype,
                    "size": size,
                    "seconds": f"{seconds:.9g}",
                }
                for repeat, point in enumerate(runs)
                for size, seconds in point.items()
            )
        return path
