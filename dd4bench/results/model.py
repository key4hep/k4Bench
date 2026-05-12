"""Data model for a single ddsim benchmark run."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class RunResult:
    """Metrics collected from one ddsim execution."""

    # --- identity ---
    label: str
    returncode: int
    n_events: int

    # --- process-level timing (from /usr/bin/time -v) ---
    # Covers the entire process including geometry initialisation.
    wall_time_raw: str | None = None   # e.g. "0:16.23" or "1:02:45.10"
    wall_time_s: float | None = None
    user_cpu_s: float | None = None
    sys_cpu_s: float | None = None

    # --- per-event metrics (from DD4benchTimingAction C++ plugin) ---
    # Recorded per event, excluding geometry initialisation.
    # Empty lists when the plugin is unavailable or the run failed.
    event_numbers: list[int] = field(default_factory=list)
    event_times_s: list[float] = field(default_factory=list)
    event_rss_peak_mb: list[float] = field(default_factory=list)
    event_rss_delta_mb: list[float] = field(default_factory=list)

    # --- memory ---
    peak_rss_mb: float | None = None

    # --- OS-level diagnostics ---
    major_page_faults: int | None = None
    voluntary_ctx_switches: int | None = None
    involuntary_ctx_switches: int | None = None

    # --- output ---
    output_size_mb: float | None = None
    events_per_sec: float | None = None

    # ---------------------------------------------------------------------------
    # Computed properties
    # ---------------------------------------------------------------------------

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    @property
    def total_cpu_s(self) -> float | None:
        if self.user_cpu_s is not None and self.sys_cpu_s is not None:
            return self.user_cpu_s + self.sys_cpu_s
        return None

    @property
    def rss_peak_mean_mb(self) -> float | None:
        """Mean per-event peak RSS in MB."""
        if not self.event_rss_peak_mb:
            return None
        return statistics.mean(self.event_rss_peak_mb)

    @property
    def rss_delta_mean_mb(self) -> float | None:
        """Mean per-event RSS delta in MB (can be negative if memory is freed)."""
        if not self.event_rss_delta_mb:
            return None
        return statistics.mean(self.event_rss_delta_mb)

    @property
    def rss_delta_std_mb(self) -> float | None:
        """Std dev of per-event RSS delta (None for < 2 events)."""
        if len(self.event_rss_delta_mb) < 2:
            return None
        return statistics.stdev(self.event_rss_delta_mb)

    @property
    def time_per_event_mean_s(self) -> float | None:
        """Mean per-event wall time from the C++ plugin (excludes init)."""
        if not self.event_times_s:
            return None
        return statistics.mean(self.event_times_s)

    @property
    def time_per_event_std_s(self) -> float | None:
        """Std dev of per-event wall time (None for < 2 events)."""
        if len(self.event_times_s) < 2:
            return None
        return statistics.stdev(self.event_times_s)

    @property
    def init_time_s(self) -> float | None:
        """Estimated initialisation time = total wall time - sum of event times."""
        if self.wall_time_s is None or not self.event_times_s:
            return None
        return max(0.0, self.wall_time_s - sum(self.event_times_s))

    def __str__(self) -> str:
        wall = f"{self.wall_time_s:.1f}s" if self.wall_time_s is not None else "N/A"
        rss = f"{self.peak_rss_mb:.0f} MB" if self.peak_rss_mb is not None else "N/A"
        eps = f"{self.events_per_sec:.3f} ev/s" if self.events_per_sec is not None else "N/A"
        mean = (
            f"{self.time_per_event_mean_s:.2f}s/ev"
            if self.time_per_event_mean_s is not None
            else "N/A"
        )
        status = "ok" if self.succeeded else f"FAILED (rc={self.returncode})"
        return f"RunResult({self.label!r}, {status}, wall={wall}, rss={rss}, {eps}, mean_ev={mean})"
