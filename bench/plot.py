"""Drawing shared by the benches: one line per device with a 95% band around the median."""

import glob
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from . import results
from .style import CHIP_SEGMENT, INK, RAMPS

Curves = dict[int, list[float]]

DISPATCH = os.path.join(os.path.dirname(__file__), "dispatch", "results")


@dataclass
class Page:
    """One figure: every series a median line with its interval shaded, both axes log."""

    name: str
    title: str
    axis_name: str
    y_name: str
    note: str
    series: dict[str, Curves]
    dashed: tuple[str, ...] = ()


def hardware_class(label: str) -> str:
    """Which hue a device gets; an unlisted chip is drawn as a consumer card."""
    if label == "scipy":
        return "scipy"
    if label.startswith("cpu"):
        return "cpu"
    return CHIP_SEGMENT.get(label, "consumer")


def latencies() -> dict[str, float]:
    """label -> the host's per-call dispatch latency, as the dispatch bench measured it.

    Minimum of that bench's samples: timing noise is one-sided, so the smallest is the latency.
    """
    floors = {}
    for path in sorted(glob.glob(os.path.join(DISPATCH, "*.csv"))):
        label = os.path.basename(path).removesuffix(".csv")
        if label == "configs":
            continue
        seconds = [float(row["seconds"]) for row in results.timings(path)]
        if seconds:
            floors[label] = min(seconds)
    return floors


def floor_of(label: str, floors: dict[str, float]) -> float:
    """This label's latency; scipy dispatches nothing, so it has none to subtract."""
    return floors.get(label, 0.0)


def throughputs(
    timings: dict[int, list[float]], work_at: Callable[[int], float], floor: float
) -> Curves:
    """Timings as throughputs, less the dispatch latency; a point at or under it is dropped whole."""
    return {
        point: [work_at(point) / (seconds - floor) for seconds in samples]
        for point, samples in timings.items()
        if min(samples) > floor
    }


def load(
    directory: str, sweep: str, work_at: Callable[[int], float]
) -> dict[tuple[str, str], Curves]:
    """(label, dtype) -> {axis point: throughputs}, from this sweep's rows of every `<label>.csv`."""
    timings: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        label = os.path.basename(path).removesuffix(".csv")
        if label == "configs":
            continue
        # a rerun appends, so the last row for a timing is the live one
        latest = {}
        for row in results.timings(path):
            if row["sweep"] == sweep:
                key = (row["run"], row["repeat"], row["dtype"], row["size"])
                latest[key] = row["seconds"]
        for (_, _, dtype, size), seconds in latest.items():
            timings[label, dtype][int(size)].append(float(seconds))

    floors = latencies()
    curves = {
        (label, dtype): throughputs(points, work_at, floor_of(label, floors))
        for (label, dtype), points in timings.items()
    }
    return {key: curve for key, curve in curves.items() if curve}


def band(samples: list[float]) -> tuple[float, float, float]:
    """Median and its 95% interval from the binomial order statistics."""
    ordered = sorted(samples)
    rank = max(int(len(ordered) / 2 - 1.96 * len(ordered) ** 0.5 / 2), 0)
    return statistics.median(ordered), ordered[rank], ordered[len(ordered) - 1 - rank]


def peak(curve: Curves) -> float:
    return max(statistics.median(v) for v in curve.values())


def translucent(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def by_class(labels) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        grouped[hardware_class(label)].append(label)
    return grouped


def colors(peaks: dict[str, float]) -> dict[str, str]:
    """One hue per hardware class, stepped within the class by rank in peak throughput."""
    within = by_class(sorted(peaks, key=peaks.get))
    return {
        label: RAMPS[group][
            round(rank * (len(RAMPS[group]) - 1) / max(len(peers) - 1, 1))
        ]
        for group, peers in within.items()
        for rank, label in enumerate(peers)
    }


def fastest_first(peaks: dict[str, float]) -> list[str]:
    """Legend order: classes fastest first, and inside a class its devices."""
    within = by_class(peaks)
    return [
        label
        for group in sorted(within, key=lambda g: -max(peaks[l] for l in within[g]))
        for label in sorted(within[group], key=lambda label: -peaks[label])
    ]


def figure(page: Page, out: str) -> None:
    """Write one page twice: the interactive `.html`, and an `.svg` still for the READMEs."""
    import plotly.graph_objects as go

    if not page.series:
        print(f"no curves yet, skipping {page.name}")
        return

    peaks = {label: peak(curve) for label, curve in page.series.items()}
    hue = colors(peaks)
    fig = go.Figure()

    for label in fastest_first(peaks):
        points = sorted(page.series[label].items())
        sizes = [n for n, _ in points]
        stats = [band(samples) for _, samples in points]
        group = hardware_class(label)
        fig.add_trace(
            go.Scatter(
                x=[*sizes, *reversed(sizes)],
                y=[s[2] for s in stats] + [s[1] for s in reversed(stats)],
                fill="toself",
                fillcolor=translucent(hue[label], 0.22),
                mode="lines",
                line={"width": 0},
                legendgroup=group,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sizes,
                y=[s[0] for s in stats],
                name=label,
                mode="lines",
                line={
                    "color": hue[label],
                    "width": 2,
                    "dash": "dash" if label in page.dashed else "solid",
                },
                legendgroup=group,
                legendgrouptitle_text=group,
                hovertemplate=(
                    f"<b>{label}</b><br>{page.axis_name} %{{x:,}}"
                    f"<br>%{{y:,.4g}} {page.y_name}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title={"text": f"{page.title}<br><sub>{page.note}</sub>", "x": 0.02},
        template="plotly_dark",
        paper_bgcolor=INK["surface"],
        plot_bgcolor=INK["surface"],
        font={"color": INK["secondary"], "size": 12},
        title_font={"color": INK["primary"], "size": 17},
        hovermode="closest",
        legend={"groupclick": "togglegroup", "traceorder": "grouped"},
        margin={"l": 70, "r": 30, "t": 90, "b": 60},
        height=680,
    )
    axis = {
        "type": "log",
        "gridcolor": INK["grid"],
        "zeroline": False,
        "linecolor": INK["grid"],
    }
    fig.update_xaxes(title_text=page.axis_name, **axis)
    fig.update_yaxes(title_text=page.y_name, **axis)

    # cdn rather than inlined: the bundle is 3 MB, and these files are committed
    fig.write_html(f"{out}.html", include_plotlyjs="cdn", full_html=True)
    fig.write_image(f"{out}.svg", width=1100, height=680)
    print(f"wrote {out}.html and {out}.svg")
