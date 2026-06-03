"""Data model for a single ddsim benchmark run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunResult:
    """Metrics collected from one ddsim execution."""

    # -----------------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------------

    label: str
    returncode: int
    n_events: int

    # -----------------------------------------------------------------------
    # Process-level timing (/usr/bin/time -v)
    # Includes geometry initialisation and full application runtime.
    # -----------------------------------------------------------------------

    wall_time_raw: str | None = None
    wall_time_s: float | None = None

    user_cpu_s: float | None = None
    sys_cpu_s: float | None = None

    # -----------------------------------------------------------------------
    # Memory
    # -----------------------------------------------------------------------

    peak_rss_mb: float | None = None

    # -----------------------------------------------------------------------
    # OS diagnostics
    # -----------------------------------------------------------------------

    major_page_faults: int | None = None

    voluntary_ctx_switches: int | None = None
    involuntary_ctx_switches: int | None = None

    # -----------------------------------------------------------------------
    # Output metrics
    # -----------------------------------------------------------------------

    output_size_mb: float | None = None
    events_per_sec: float | None = None

    # -----------------------------------------------------------------------
    # Computed properties
    # -----------------------------------------------------------------------

    @property
    def succeeded(self) -> bool:
        """Return True if the process exited successfully."""
        return self.returncode == 0

    @property
    def total_cpu_s(self) -> float | None:
        """Return total CPU time = user + system."""
        if self.user_cpu_s is None or self.sys_cpu_s is None:
            return None

        return self.user_cpu_s + self.sys_cpu_s

    @property
    def cpu_efficiency(self) -> float | None:
        """Return CPU / wall ratio."""
        if self.total_cpu_s is None or self.wall_time_s is None:
            return None

        if self.wall_time_s <= 0:
            return None

        return self.total_cpu_s / self.wall_time_s

    def __str__(self) -> str:

        wall = f"{self.wall_time_s:.1f}s" if self.wall_time_s is not None else "N/A"

        rss = f"{self.peak_rss_mb:.0f} MB" if self.peak_rss_mb is not None else "N/A"

        eps = (
            f"{self.events_per_sec:.3f} ev/s"
            if self.events_per_sec is not None
            else "N/A"
        )

        status = "ok" if self.succeeded else f"FAILED (rc={self.returncode})"

        return (
            f"RunResult("
            f"{self.label!r}, "
            f"{status}, "
            f"wall={wall}, "
            f"rss={rss}, "
            f"{eps}"
            f")"
        )
