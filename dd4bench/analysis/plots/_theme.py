"""Visual style constants shared across all plot modules."""

from __future__ import annotations

_TEMPLATE = "plotly_white"

_BLUE = "#1f77b4"
_RED  = "#d62728"

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_METRIC_UNITS: dict[str, str] = {
    "wall_time_s":    "(s)",
    "peak_rss_mb":    "(MB)",
    "user_cpu_s":     "(s)",
    "sys_cpu_s":      "(s)",
    "events_per_sec": "(ev/s)",
    "output_size_mb": "(MB)",
}

_UNACCOUNTED_COLOR = "#999999"
_OTHER_COLOR       = "#d0d0d0"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color string to an rgba() string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
