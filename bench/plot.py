"""Drawing shared by both benches: one line per device with a 95% band around the median.

Each bench builds its own `Page`s -- axes, titles and which curves belong together -- and what
lives here is how a device is coloured, how the band is computed, and the figure itself.
"""

import csv
import glob
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from .style import CHIP_SEGMENT, INK, RAMPS

Curves = dict[int, list[float]]
# how far clear of its host's dispatch latency a point must be to survive the subtraction
FLOOR_CLEARANCE = 2.0


@dataclass
class Page:
    """One figure: every series drawn as a median line with its interval shaded, both axes log."""

    name: str
    title: str
    axis_name: str
    y_name: str
    note: str
    series: dict[str, Curves]
    dashed: tuple[str, ...] = ()


def hardware_class(label: str) -> str:
    """Which hue a device gets. An unlisted chip is drawn as a consumer card, the safe default."""
    chip = label.removeprefix("jax-")
    # scipy is the baseline rather than another device, so it gets its own group
    if chip.startswith("scipy"):
        return "scipy"
    if chip.startswith("cpu"):
        return "cpu"
    return CHIP_SEGMENT.get(chip, "consumer")


def throughputs(
    timings: dict[int, list[float]], work_at: Callable[[int], float]
) -> Curves:
    """One device's timings as throughputs, less the host's per-call dispatch latency.

    Every call is blocked on, so the round trip to the device adds to the kernel rather than
    overlapping it, and the cheapest point of the sweep is nearly all round trip: that is the
    latency. It is host-side, so it differs by an order of magnitude between an x86 and an arm
    host and would otherwise be the only thing the flat left of these curves shows.

    A point not clear of the floor by `FLOOR_CLEARANCE` is the difference of two near-equal noisy
    numbers, so it is dropped rather than drawn.
    """
    # pooled over every point that is still flat rather than read off one of them: the flat run is
    # noisy enough that its cheapest point is a dip, and subtracting a dip inflates what is left
    cheapest = min(statistics.median(samples) for samples in timings.values())
    floor = statistics.median(
        [
            seconds
            for samples in timings.values()
            if statistics.median(samples) < 1.5 * cheapest
            for seconds in samples
        ]
    )
    return {
        point: [
            work_at(point) / (seconds - floor) for seconds in samples if seconds > floor
        ]
        for point, samples in timings.items()
        if statistics.median(samples) > FLOOR_CLEARANCE * floor
    }


def load(
    directory: str, sweep: str, work_at: Callable[[int], float]
) -> dict[tuple[str, str], Curves]:
    """(label, dtype) -> {axis point: throughputs}, from this sweep's rows of every `<label>.csv`.

    Label comes off the file name rather than configs.csv, so a device still plots while its jobs
    are mid-flight and its constants have not landed yet.
    """
    timings: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in sorted(glob.glob(os.path.join(directory, "*.csv"))):
        label = os.path.basename(path).removesuffix(".csv")
        if label == "configs":
            continue
        with open(path) as fh:
            # a rerun appends rather than replacing, so the last row for a timing is the live one
            latest = {}
            for row in csv.DictReader(fh):
                if row["sweep"] == sweep:
                    key = (row["run"], row["repeat"], row["dtype"], row["size"])
                    latest[key] = row["seconds"]
        for (_, _, dtype, size), seconds in latest.items():
            timings[label, dtype][int(size)].append(float(seconds))

    # a device whose every point sat in its own dispatch latency has no curve left to draw
    curves = {key: throughputs(points, work_at) for key, points in timings.items()}
    return {key: curve for key, curve in curves.items() if curve}


def band(samples: list[float]) -> tuple[float, float, float]:
    """Median throughput and the 95% interval on it, from the binomial order statistics.

    Median rather than mean because timing noise is one-sided -- contention only ever slows a
    call down -- so one stalled block drags the mean off the curve and the median ignores it.
    """
    ordered = sorted(samples)
    rank = max(int(len(ordered) / 2 - 1.96 * len(ordered) ** 0.5 / 2), 0)
    return statistics.median(ordered), ordered[rank], ordered[len(ordered) - 1 - rank]


def peak(curve: Curves) -> float:
    return max(statistics.median(v) for v in curve.values())


def translucent(hex_color: str, alpha: float) -> str:
    """The same colour as an rgba string, which is the only form a plotly fill takes."""
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
    """Legend order under `grouped`: classes fastest first, and inside a class its devices."""
    within = by_class(peaks)
    return [
        label
        for group in sorted(within, key=lambda g: -max(peaks[l] for l in within[g]))
        for label in sorted(within[group], key=lambda label: -peaks[label])
    ]


def figure(page: Page, out: str) -> None:
    """Write one page twice: the interactive `.html`, and an `.svg` still of it for the READMEs."""
    import plotly.graph_objects as go

    if not page.series:  # a partial run, before that dtype's sweeps have landed
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
                # explicit: plotly defaults a short trace to lines+markers, and this is a polygon
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
