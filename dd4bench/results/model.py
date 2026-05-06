"""Data model for a single benchmark run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunResult:
    """Metrics collected from one execution."""

    # --- identity ---
    label: str
    returncode: int
    n_events: int

    # --- timing (from /usr/bin/time -v) ---
    wall_time_raw: str | None = None
    wall_time_s: float | None = None
    user_cpu_s: float | None = None
    sys_cpu_s: float | None = None

    # --- memory ---
    peak_rss_mb: float | None = None

    # --- OS-level diagnostics ---
    major_page_faults: int | None = None
    voluntary_ctx_switches: int | None = None
    involuntary_ctx_switches: int | None = None

    # --- output ---
    output_size_mb: float | None = None
    events_per_sec: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    @property
    def total_cpu_s(self) -> float | None:
        if self.user_cpu_s is not None and self.sys_cpu_s is not None:
            return self.user_cpu_s + self.sys_cpu_s
        return None

    def __str__(self) -> str:
        wall = f"{self.wall_time_s:.1f}s" if self.wall_time_s is not None else "N/A"
        rss = f"{self.peak_rss_mb:.0f} MB" if self.peak_rss_mb is not None else "N/A"
        eps = f"{self.events_per_sec:.3f} ev/s" if self.events_per_sec is not None else "N/A"
        status = "ok" if self.succeeded else f"FAILED (rc={self.returncode})"
        return f"RunResult({self.label!r}, {status}, wall={wall}, rss={rss}, {eps})"
