"""Submit one Slurm job array per GPU chip type this cluster exposes, plus the CPU jobs."""

import argparse
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ppc64le: no uv or jax cuda wheels
SKIP_PARTITIONS = ("power9",)
# blackwell is refused at submit time; h200_nvl pends forever
SKIP_CHIPS = (
    "nvidia_rtx_pro_6000_blackwell_server_edition_2g.48gb",
    "h200_nvl",
)
# named on every GPU job: nodes move between partitions, and these carry most of the cluster
ALWAYS_PARTITIONS = ("gpu", "gpu-preempt")
CPU_FEATURE = "intel8352y"
CPU_PARTITIONS = "cpu,cpu-preempt"


def add_arguments(
    p: argparse.ArgumentParser, cores: str, repeats: int, walltime: str
) -> None:
    p.add_argument(
        "--cores", default=cores, help="comma separated core counts, one CPU job each"
    )
    p.add_argument(
        "--repeats", type=int, default=repeats, help="sequential runs per point"
    )
    p.add_argument(
        "--chips", default=None, help="comma separated subset of the GPU chip types"
    )
    p.add_argument(
        "--dtypes", default=None, help="comma separated subset of the dtypes swept"
    )
    p.add_argument(
        "--cpu-feature",
        default=CPU_FEATURE,
        help=f"CPU node feature the pinned node is picked by, default {CPU_FEATURE}",
    )
    p.add_argument(
        "--cpu-node",
        default=None,
        help="the node every CPU job runs on, default an idle one",
    )
    p.add_argument("--gpu-time", default=walltime)
    p.add_argument("--cpu-time", default=walltime)
    p.add_argument("--skip-gpu", action="store_true")
    p.add_argument("--skip-cpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")


def submittable(partition: str) -> bool:
    """Whether this account may queue here; naming an unreachable partition fails the submission."""
    probe = subprocess.run(
        ["sbatch", "--test-only", "-p", partition, "--gres=gpu:1", "--wrap", "true"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def gpu_chips() -> dict[str, list[str]]:
    """Chip type -> every submittable partition offering it."""
    out = subprocess.run(
        ["sinfo", "-h", "-o", "%P|%G"], capture_output=True, text=True, check=True
    ).stdout

    offered: dict[str, set[str]] = {}
    for line in out.splitlines():
        partition, gres = (field.strip().rstrip("*") for field in line.split("|"))
        if gres.startswith("gpu:") and not partition.startswith(SKIP_PARTITIONS):
            offered.setdefault(gres.split(":")[1], set()).add(partition)

    candidates = set().union(*offered.values())
    allowed = {p for p in candidates if submittable(p)} | set(ALWAYS_PARTITIONS)
    return {
        chip: sorted((partitions | set(ALWAYS_PARTITIONS)) & allowed)
        for chip, partitions in sorted(offered.items())
        if partitions & allowed and chip not in SKIP_CHIPS
    }


def cpu_nodes() -> list[tuple[bool, str, set[str], set[str]]]:
    """(busy, node, partitions, features) for every CPU node."""
    out = subprocess.run(
        ["sinfo", "-h", "-N", "-p", CPU_PARTITIONS, "-o", "%N|%P|%T|%f"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    nodes: dict[str, tuple[bool, set[str], set[str]]] = {}
    for line in out.splitlines():
        node, partition, state, features = (f.strip() for f in line.split("|"))
        busy, partitions, seen = nodes.setdefault(node, (False, set(), set()))
        nodes[node] = (
            busy or state.rstrip("*") != "idle",
            partitions | {partition.rstrip("*")},
            seen | set(features.split(",")),
        )
    return [
        (busy, node, partitions, features)
        for node, (busy, partitions, features) in nodes.items()
    ]


def cpu_node(feature: str) -> tuple[str, str]:
    """An idle node carrying the feature, and its own partitions (naming others is rejected)."""
    carrying = sorted(
        (busy, node, partitions)
        for busy, node, partitions, features in cpu_nodes()
        if feature in features
    )
    if not carrying:
        raise SystemExit(f"no node in {CPU_PARTITIONS} carries {feature}")
    _, node, partitions = carrying[0]
    return node, ",".join(sorted(partitions))


def partitions_of(node: str) -> str:
    for _, listed, partitions, _ in cpu_nodes():
        if listed == node:
            return ",".join(sorted(partitions))
    raise SystemExit(f"{node} is in none of {CPU_PARTITIONS}")


def selected(configs: tuple[str, ...], dtypes: set[str] | None) -> tuple[str, ...]:
    """Every config names its dtype in one of its colon fields, whatever else it carries."""
    if dtypes is None:
        return configs
    return tuple(config for config in configs if set(config.split(":")) & dtypes)


def sbatch(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if dry_run:
        return
    done = subprocess.run(command, capture_output=True, text=True)
    print((done.stdout or done.stderr).strip())


def run(args) -> None:
    """One array per GPU chip and one per CPU core count; each task is one point."""
    bench = args.bench_name
    logs = os.path.join(ROOT, "bench", bench, "logs")
    os.makedirs(logs, exist_ok=True)
    export = f"--export=ALL,REPEATS={args.repeats}"
    wanted = args.chips.split(",") if args.chips else None
    dtypes = set(args.dtypes.split(",")) if args.dtypes else None

    if not args.skip_gpu:
        configs = selected(args.bench.GPU_CONFIGS, dtypes)
        for chip, partitions in gpu_chips().items():
            if wanted is not None and chip not in wanted:
                continue
            # pinned by node feature rather than gres, so any carrying partition may start it
            sbatch(
                [
                    "sbatch",
                    "--chdir",
                    ROOT,
                    "-a",
                    f"1-{len(configs)}",
                    export,
                    "-p",
                    ",".join(partitions),
                    "--gres=gpu:1",
                    f"--constraint={chip}",
                    "-c",
                    "4",
                    "--mem",
                    "16G",
                    "-t",
                    args.gpu_time,
                    "-J",
                    f"{bench}-{chip}",
                    "-o",
                    os.path.join(logs, f"gpu-{chip}-%a.out"),
                    f"bench/{bench}/slurm/gpu.sh",
                    chip,
                    *configs,
                ],
                args.dry_run,
            )

    if args.skip_cpu:
        return
    # every width on the same node, held exclusively: only the width moves between jobs
    if args.cpu_node:
        node, partitions = args.cpu_node, partitions_of(args.cpu_node)
    else:
        node, partitions = cpu_node(args.cpu_feature)
    for cores in args.cores.split(","):
        configs = selected(args.bench.cpu_configs(int(cores)), dtypes)
        sbatch(
            [
                "sbatch",
                "--chdir",
                ROOT,
                "-a",
                f"1-{len(configs)}",
                "-p",
                partitions,
                f"--nodelist={node}",
                "--exclusive",
                "-c",
                cores,
                "--mem",
                "64G",
                "-t",
                args.cpu_time,
                "-J",
                f"{bench}-cpu{cores}",
                "-o",
                os.path.join(logs, f"cpu{cores}-%a.out"),
                export,
                f"bench/{bench}/slurm/cpu.sh",
                *configs,
            ],
            args.dry_run,
        )
