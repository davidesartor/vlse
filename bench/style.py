"""Palettes for the bench plots, and the hardware classes they are keyed by."""

# the market segment each chip was sold into, which is also roughly the memory system it got:
# `consumer` is GeForce and Quadro, GDDR on a narrow bus, `server` GDDR on a wide one, `hpc` HBM
SEGMENTS = {
    "cpu": (),
    "scipy": (),
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

# one hue per hardware class, stepped for the devices inside that class, slowest first -- as many
# steps as the class has chips. Lighter with speed, since the figures are drawn on the dark surface
RAMPS = {
    "cpu": ["#0f6647", "#158a60", "#199e70", "#4ec49b", "#8ee0bd"],
    # the reference rather than a device: neutral, so it reads as the line the rest are read against
    "scipy": ["#6f6f68", "#8a8a80", "#a5a599", "#c3c2b7", "#dedcd0"],
    # seven steps rather than five: the merged class carries GeForce and Quadro both
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
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "grid": "#3a3a37",
    },
}
