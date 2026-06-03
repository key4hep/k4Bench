"""Parse the output of ``/usr/bin/time -v``.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_time_output(text: str) -> dict:
    """Return a dict of metrics extracted from ``/usr/bin/time -v`` output.

    All values are ``None`` when the corresponding line is absent or
    cannot be converted to the expected type.

    Parameters
    ----------
    text:
        Full stdout/stderr text of the timed process, as produced by
        ``/usr/bin/time -v``.

    Returns
    -------
    dict with keys:
        wall_time_raw, wall_time_s, user_cpu_s, sys_cpu_s,
        peak_rss_mb, major_page_faults,
        voluntary_ctx_switches, involuntary_ctx_switches
    """
    metrics: dict = {
        "wall_time_raw": None,
        "wall_time_s": None,
        "user_cpu_s": None,
        "sys_cpu_s": None,
        "peak_rss_mb": None,
        "major_page_faults": None,
        "voluntary_ctx_switches": None,
        "involuntary_ctx_switches": None,
    }

    for line in text.splitlines():
        # Split on the *last* colon so h:mm:ss wall times are handled
        # correctly.  Each /usr/bin/time -v line has the format
        # "    Label (detail): value".
        key_part, _, value_part = line.rpartition(":")
        value = value_part.strip()

        if "Elapsed (wall clock)" in key_part:
            # The wall-clock line uses ": " as separator before the time,
            # so we must re-join key_part + ":" + value and re-extract.
            # Example line:
            #   "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:16.23"
            match = re.search(r":\s*(\d+:\d{2}(?::\d{2})?(?:\.\d+)?)$", line)
            if match:
                raw = match.group(1)
                metrics["wall_time_raw"] = raw
                metrics["wall_time_s"] = _wall_to_seconds(raw)

        elif "User time" in key_part:
            metrics["user_cpu_s"] = _to_float(value)

        elif "System time" in key_part:
            metrics["sys_cpu_s"] = _to_float(value)

        elif "Maximum resident set size" in key_part:
            kb = _to_int(value)
            metrics["peak_rss_mb"] = kb / 1024.0 if kb is not None else None

        elif "Major (requiring I/O) page faults" in key_part:
            metrics["major_page_faults"] = _to_int(value)

        # "Voluntary context switches" must come before the next check
        # because "Involuntary" also contains "voluntary".
        elif "Voluntary context switches" in key_part:
            metrics["voluntary_ctx_switches"] = _to_int(value)

        elif "Involuntary context switches" in key_part:
            metrics["involuntary_ctx_switches"] = _to_int(value)

    return metrics


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _wall_to_seconds(wall: str) -> float | None:
    """Convert ``m:ss.ss`` or ``h:mm:ss.ss`` to total seconds.

    Returns ``None`` if *wall* does not match the expected format.
    """
    parts = wall.split(":")
    try:
        if len(parts) == 2:                          # m:ss.ss
            return float(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:                          # h:mm:ss.ss
            return (
                float(parts[0]) * 3600
                + float(parts[1]) * 60
                + float(parts[2])
            )
    except ValueError:
        pass
    return None


def _to_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None
