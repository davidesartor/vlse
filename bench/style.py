"""Palettes for the bench plots, and the hardware classes they are keyed by."""

# the market segment a chip was sold into, which is also roughly the memory system it got:
# `consumer` is GeForce and Quadro, GDDR on a narrow bus, `server` GDDR on a wide one, `hpc` HBM
SEGMENTS = {
    "cpu": (),
    "consumer": (
        "titan_x",
        "1080_ti",
        "2080_ti",
        "rtx_8000",
        "a4000",
        "a5000",
        "a6000",
    ),
    "server": ("m40", "a16", "a40", "l4", "l40s"),
    "hpc": ("v100", "a100", "h100", "h200_nvl", "gh200"),
}
CHIP_SEGMENT = {chip: segment for segment, chips in SEGMENTS.items() for chip in chips}

# one hue per segment, stepped for the devices inside it, slowest first; lighter with speed, since
# the figures are drawn on the dark surface. consumer gets seven steps: it merges GeForce and Quadro
RAMPS = {
    # seven steps: the optim bench draws a CPU curve per core count, 1 through 64
    "cpu": [
        "#0b4f38",
        "#0f6647",
        "#158a60",
        "#199e70",
        "#4ec49b",
        "#8ee0bd",
        "#c4eedb",
    ],
    "scipy": ["#8a8a85"],
    "consumer": [
        "#9c360f",
        "#b34115",
        "#c9491a",
        "#d95926",
        "#e37344",
        "#ee8f61",
        "#f6bfa2",
    ],
    "server": ["#8f5e00", "#c98500", "#e0a41f", "#eec369", "#f7dda1"],
    "hpc": ["#1c5cab", "#2a78d6", "#3987e5", "#6da7ec", "#a8ccf6"],
}
INK = {
    "surface": "#1a1a19",
    "primary": "#ffffff",
    "secondary": "#c3c2b7",
    "grid": "#3a3a37",
}
