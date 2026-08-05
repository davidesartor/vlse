"""Drawing shared by both benches: rows in, one line per device with a band around the median.

Each bench keeps its own axes and titles; what lives here is how a device is coloured, how the
band is computed, and the figure itself.
"""

import csv
import glob
import os
import statistics
from collections import defaultdict

import plotly.graph_objects as go
from style import CHIP_SEGMENT, INK, RAMPS


def hardware_class(label: str) -> str:
    """Which hue a device gets. An unlisted chip is drawn as a consumer card, the safe default."""
    if label.startswith("scipy"):
        return "scipy"
    chip = label.removeprefix("jax-")
    if chip.startswith("cpu"):
        return "cpu"
    return CHIP_SEGMENT.get(chip, "consumer")


def load(
    directory: str, stem: str, work_per_call
) -> dict[tuple[str, str], dict[int, list[float]]]:
    """(label, dtype) -> {axis point: throughputs}, from every `<stem>-<label>-<dtype>.csv`.

    One column per point of the axis and one row per job, blank where that job stopped earlier
    than the rest. Label and dtype come off the file name rather than configs.csv, so a device
    still plots while its jobs are mid-flight and its constants have not landed yet.
    """
    curves: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in sorted(glob.glob(os.path.join(directory, f"{stem}-*.csv"))):
        label, dtype = (
            os.path.basename(path).removesuffix(".csv").removeprefix(f"{stem}-")
        ).rsplit("-", 1)
        with open(path) as fh:
            for r in csv.DictReader(fh):
                for column, seconds in r.items():
                    if column != "run" and seconds:
                        point = int(column)
                        curves[label, dtype][point].append(
                            work_per_call(point) / float(seconds)
                        )
    return curves


def band(samples: list[float]) -> tuple[float, float, float]:
    """Median throughput and the 95% interval on it, from the binomial order statistics.

    Median rather than mean because timing noise is one-sided -- contention only ever slows a
    call down -- so one stalled block drags the mean off the curve and the median ignores it.
    """
    ordered = sorted(samples)
    median = statistics.median(ordered)
    rank = max(int(len(ordered) / 2 - 1.96 * len(ordered) ** 0.5 / 2), 0)
    return median, ordered[rank], ordered[len(ordered) - 1 - rank]


def peak(curve: dict[int, list[float]]) -> float:
    return max(statistics.median(v) for v in curve.values())


def translucent(hex_color: str, alpha: float) -> str:
    """The same colour as an rgba string, which is the only form a plotly fill takes."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def colors(peaks: dict[str, float]) -> dict[str, str]:
    """One hue per hardware class, stepped within the class by rank in peak throughput."""
    within: dict[str, list[str]] = defaultdict(list)
    for label in sorted(peaks, key=peaks.get):
        within[hardware_class(label)].append(label)
    return {
        label: RAMPS[group][
            round(rank * (len(RAMPS[group]) - 1) / max(len(peers) - 1, 1))
        ]
        for group, peers in within.items()
        for rank, label in enumerate(peers)
    }


def figure(
    series: dict[str, dict[int, list[float]]],
    out: str,
    title: str,
    axis_name: str,
    y_name: str,
    note: str,
    dashed: tuple[str, ...] = (),
) -> None:
    """One page: every series as a median line with its interval shaded, both axes log."""
    if not series:  # a partial run, before that dtype's sweeps have landed
        print(f"no curves yet, skipping {os.path.basename(out)}")
        return

    ink = INK["dark"]
    peaks = {label: peak(curve) for label, curve in series.items()}
    hue = colors(peaks)
    fig = go.Figure()

    # legend order is trace order under `grouped`: classes fastest first, and inside a class its
    # devices fastest first
    within: dict[str, list[str]] = defaultdict(list)
    for label in peaks:
        within[hardware_class(label)].append(label)
    fastest_first = [
        label
        for group in sorted(within, key=lambda g: -max(peaks[l] for l in within[g]))
        for label in sorted(within[group], key=lambda label: -peaks[label])
    ]

    for label in fastest_first:
        points = sorted(series[label].items())
        sizes = [n for n, _ in points]
        stats = [band(samples) for _, samples in points]
        group = hardware_class(label)
        fig.add_trace(
            go.Scatter(
                x=[*sizes, *reversed(sizes)],
                y=[s[2] for s in stats] + [s[1] for s in reversed(stats)],
                fill="toself",
                fillcolor=translucent(hue[label], 0.22),
                # explicit: plotly defaults a short trace to lines+markers, and the band is a polygon
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
                    "dash": "dash" if label in dashed else "solid",
                },
                legendgroup=group,
                legendgrouptitle_text=group,
                hovertemplate=f"<b>{label}</b><br>{axis_name} %{{x:,}}<br>%{{y:,.4g}} {y_name}<extra></extra>",
            )
        )

    fig.update_layout(
        title={"text": f"{title}<br><sub>{note}</sub>", "x": 0.02},
        template="plotly_dark",
        paper_bgcolor=ink["surface"],
        plot_bgcolor=ink["surface"],
        font={"color": ink["secondary"], "size": 12},
        title_font={"color": ink["primary"], "size": 17},
        hovermode="closest",
        legend={"groupclick": "togglegroup", "traceorder": "grouped"},
        margin={"l": 70, "r": 30, "t": 90, "b": 60},
        height=680,
    )
    axis = {
        "type": "log",
        "gridcolor": ink["grid"],
        "zeroline": False,
        "linecolor": ink["grid"],
    }
    fig.update_xaxes(title_text=axis_name, **axis)
    fig.update_yaxes(title_text=y_name, **axis)

    # cdn rather than inlined: the bundle is 3 MB, and these files are committed
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print(f"wrote {out}")

    # the same panel as a still, for the READMEs: github and pypi render an image, not a script
    still = out.removesuffix(".html") + ".svg"
    fig.write_image(still, width=1100, height=680)
    print(f"wrote {still}")
