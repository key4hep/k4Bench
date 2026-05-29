"""Pure trend-window resolution (no Streamlit), so it can be unit-tested.

The sidebar offers a set of look-back presets plus a custom range; this module
turns a chosen preset into a concrete inclusive ``(start, end)`` date window that
the caller uses to filter the run dates it downloads.
"""

from __future__ import annotations

from datetime import date, timedelta

# Trend-window presets → look-back length in days; ``None`` means special handling.
WINDOW_PRESETS: dict[str, int | None] = {
    "Last 7 days":   7,
    "Last 30 days":  30,
    "Last 90 days":  90,
    "Last 6 months": 182,
    "All":           None,
    "Custom…":       None,
}


def resolve_window(
    preset: str,
    all_dates: list[date],
    custom_range: tuple[date, date] | None,
) -> tuple[date, date]:
    """Resolve a preset (or custom range) to an inclusive ``(start, end)`` window.

    The window is anchored on the latest available run date, not today, so the
    default preset always shows data even if the nightly has not run recently.
    A preset of *N* days yields an inclusive window spanning exactly *N* calendar
    days (``end`` back through ``end - (N - 1)``), so the label matches the range.
    """
    lo, hi = min(all_dates), max(all_dates)
    if preset == "All":
        return lo, hi
    if preset == "Custom…":
        if custom_range is None:
            return lo, hi
        start, end = custom_range
        return start, end
    days = WINDOW_PRESETS[preset] or 0
    return hi - timedelta(days=max(days - 1, 0)), hi
