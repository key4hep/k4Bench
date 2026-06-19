"""Conservative pass/fail reliability check for a benchmark run.

The goal here is **not** to score performance quality on a continuous scale, but
to detect runs that were *likely* affected by environmental interference and
should be excluded from comparisons.

A run is judged ``RELIABLE`` only when every *hard* criterion either passes or
has no data to judge it. Any single hard failure makes the run ``UNRELIABLE``.
The guiding principle is conservative and asymmetric:

* only **positive evidence** of interference can reject a run — a criterion with
  no data (``UNKNOWN``) never fails it;
* some signals are advisory (``WARN``) and are reported but never reject a run.

Hard criteria (a single ``FAIL`` rejects the run)
    * **CPU efficiency** ≥ 95% of the ideal for the thread count. For a
      single-threaded run that is ``total_cpu / wall ≥ 0.95``; for an
      ``n``-threaded run the floor scales to ``0.95 * n``. Low efficiency means
      the thread spent time off-CPU — waiting, contended, preempted, or doing
      I/O. This is the primary contention detector for a single-core workload.
    * **Load average** ≤ physical core count. Load above the physical core count
      means the host was oversubscribed.
    * **No swap activity** — any swap-in/swap-out *during* the run rejects it.
    * **No thermal throttling** — any throttle event during the run rejects it.

Advisory criteria (reported, never reject)
    * **Involuntary context switches** ≤ ``10×`` a baseline established from prior
      clean runs, normalised per CPU-second so it is robust to run length and
      core count. With no baseline the value is reported only (``WARN``) — never
      a sole cause for rejection.
    * **RAM utilisation** > 90% — a warning only. High RAM use alone is not
      harmful as long as no swapping occurs (which the swap criterion covers).

This module is intentionally dependency-free (no pandas/streamlit) so it can be
unit-tested in isolation; the dashboard consumes it today, and the runner/reporter
can reuse it without pulling in UI dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ── Thresholds ──────────────────────────────────────────────────────────────────

#: Minimum CPU efficiency *per thread* for a run to pass (total_cpu / wall).
MIN_CPU_EFFICIENCY = 0.95

#: Involuntary context switches may exceed the baseline by at most this factor.
CTX_SWITCH_BASELINE_MULTIPLIER = 10.0

#: 1-minute load average must not exceed (this factor × physical cores).
MAX_LOAD_PER_CORE = 1.0

#: RAM utilisation above this fraction raises a warning (never a rejection).
RAM_WARN_FRACTION = 0.90


class Status(str, Enum):
    """Outcome of a single reliability criterion."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    UNKNOWN = "unknown"  # no data available to judge this criterion


@dataclass(frozen=True)
class Criterion:
    """The evaluated result of one reliability criterion."""

    name: str
    status: Status
    detail: str
    #: The measured value, formatted for display (e.g. ``"99%"``, ``"0 pages"``).
    measured: str = "—"
    #: The threshold the value is judged against (e.g. ``"≥ 95%"``, ``"0 pages"``).
    limit: str = "—"
    #: ``True`` if a ``FAIL`` here rejects the run; ``False`` for advisory checks.
    hard: bool = True

    @property
    def rejecting(self) -> bool:
        """Whether this criterion is a hard failure that rejects the run."""
        return self.hard and self.status is Status.FAIL


@dataclass(frozen=True)
class ReliabilityVerdict:
    """Aggregate verdict over all criteria for a single run."""

    criteria: tuple[Criterion, ...]

    @property
    def failures(self) -> list[Criterion]:
        """Hard criteria that failed (the reasons a run is rejected)."""
        return [c for c in self.criteria if c.rejecting]

    @property
    def warnings(self) -> list[Criterion]:
        """Advisory criteria that flagged a concern without rejecting the run."""
        return [c for c in self.criteria if c.status is Status.WARN]

    @property
    def reliable(self) -> bool | None:
        """Overall verdict.

        ``False`` if any hard criterion failed. ``None`` (unknown) if *no* hard
        criterion could be evaluated at all — there is no evidence either way.
        ``True`` otherwise (at least one hard criterion passed and none failed).
        """
        if self.failures:
            return False
        judged_hard = [
            c for c in self.criteria
            if c.hard and c.status in (Status.PASS, Status.FAIL)
        ]
        if not judged_hard:
            return None
        return True


# ── Per-criterion evaluation ────────────────────────────────────────────────────

def _eval_cpu_efficiency(cpu_efficiency: float | None, n_threads: int) -> Criterion:
    name = "CPU efficiency"
    limit = f"≥ {MIN_CPU_EFFICIENCY * 100:.0f}%"
    if cpu_efficiency is None:
        return Criterion(name, Status.UNKNOWN, "No CPU/wall-time data recorded.", limit=limit)
    floor = MIN_CPU_EFFICIENCY * max(1, n_threads)
    pct = cpu_efficiency / max(1, n_threads) * 100
    measured = f"{pct:.0f}%"
    if cpu_efficiency >= floor:
        return Criterion(
            name, Status.PASS,
            "The thread had the CPU essentially to itself.",
            measured=measured, limit=limit,
        )
    return Criterion(
        name, Status.FAIL,
        "The thread spent time off-CPU, indicating contention, preemption or I/O waiting.",
        measured=measured, limit=limit,
    )


def _eval_ctx_switches(
    involuntary_ctx_switches: int | None,
    total_cpu_s: float | None,
    baseline_per_cpu_s: float | None,
) -> Criterion:
    name = "Involuntary context switches"
    if involuntary_ctx_switches is None:
        return Criterion(name, Status.UNKNOWN, "Not recorded.", hard=False)
    if not total_cpu_s or total_cpu_s <= 0:
        per_cpu_s = None
    else:
        per_cpu_s = involuntary_ctx_switches / total_cpu_s
    if baseline_per_cpu_s is None or per_cpu_s is None:
        measured = (
            f"{per_cpu_s:.1f}/CPU-s" if per_cpu_s is not None
            else f"{involuntary_ctx_switches:,.0f}"
        )
        return Criterion(
            name, Status.WARN,
            "No clean-run baseline exists yet — reported only, not used to reject the run.",
            measured=measured, limit="no baseline", hard=False,
        )
    limit_val = CTX_SWITCH_BASELINE_MULTIPLIER * baseline_per_cpu_s
    measured = f"{per_cpu_s:.1f}/CPU-s"
    limit = f"≤ {limit_val:.1f}/CPU-s"
    if per_cpu_s <= limit_val:
        return Criterion(
            name, Status.PASS,
            f"Within {CTX_SWITCH_BASELINE_MULTIPLIER:.0f}× the clean-run baseline "
            f"of {baseline_per_cpu_s:.1f}/CPU-s.",
            measured=measured, limit=limit, hard=False,
        )
    return Criterion(
        name, Status.WARN,
        f"Exceeds {CTX_SWITCH_BASELINE_MULTIPLIER:.0f}× the baseline of "
        f"{baseline_per_cpu_s:.1f}/CPU-s — strong sign of preemption.",
        measured=measured, limit=limit, hard=False,
    )


def _eval_load(
    load_avg_1m_pre: float | None,
    load_avg_1m_post: float | None,
    physical_cores: int | None,
) -> Criterion:
    name = "System load"
    limit = "≤ 100%"
    samples = [v for v in (load_avg_1m_pre, load_avg_1m_post) if v is not None]
    if not samples or not physical_cores:
        return Criterion(name, Status.UNKNOWN, "Load average or core count not recorded.",
                         limit=limit)
    peak = max(samples)
    util_pct = peak / physical_cores * 100
    measured = f"{util_pct:.0f}%"
    # 100% means the run-queue is as deep as the physical core count: every core
    # has a runnable task, leaving no idle core for the single-threaded benchmark.
    context = (
        f"peak 1-min load {peak:.2f} across {physical_cores} physical cores "
        f"({util_pct:.0f}% of cores subscribed)"
    )
    if peak <= MAX_LOAD_PER_CORE * physical_cores:
        return Criterion(
            name, Status.PASS,
            f"Not oversubscribed — {context}.",
            measured=measured, limit=limit,
        )
    return Criterion(
        name, Status.FAIL,
        f"Host was oversubscribed — {context}.",
        measured=measured, limit=limit,
    )


def _eval_swap(swap_in_pages: int | None, swap_out_pages: int | None) -> Criterion:
    name = "Swap activity"
    limit = "0 pages"
    if swap_in_pages is None and swap_out_pages is None:
        return Criterion(name, Status.UNKNOWN, "Swap activity not recorded.", limit=limit)
    if swap_in_pages is None or swap_out_pages is None:
        return Criterion(
            name,
            Status.UNKNOWN,
            "Swap activity only partially recorded.",
            limit=limit,
        )
    total = swap_in_pages + swap_out_pages
    if total == 0:
        return Criterion(name, Status.PASS, "No paging to disk during the run.",
                         measured="0 pages", limit=limit)
    return Criterion(
        name, Status.FAIL,
        "The kernel paged to disk during the run — memory pressure may have affected timings.",
        measured=f"{total:,} pages", limit=limit,
    )


def _eval_thermal(thermal_throttle_events: int | None) -> Criterion:
    name = "Thermal throttling"
    limit = "none"
    if thermal_throttle_events is None:
        return Criterion(name, Status.UNKNOWN,
                         "Throttle counters unavailable (e.g. container/VM).", limit=limit)
    if thermal_throttle_events == 0:
        return Criterion(name, Status.PASS, "CPU was not throttled for heat during the run.",
                         measured="none", limit=limit)
    return Criterion(
        name, Status.FAIL,
        "CPU was thermally throttled during the run — clock speed was reduced, inflating timings.",
        measured="detected", limit=limit,
    )


def _eval_ram(ram_used_fraction: float | None) -> Criterion:
    name = "RAM utilisation"
    limit = f"≤ {RAM_WARN_FRACTION * 100:.0f}%"
    if ram_used_fraction is None:
        return Criterion(name, Status.UNKNOWN, "RAM availability not recorded.",
                         limit=limit, hard=False)
    measured = f"{ram_used_fraction * 100:.0f}%"
    if ram_used_fraction <= RAM_WARN_FRACTION:
        return Criterion(name, Status.PASS, "Comfortable memory headroom.",
                         measured=measured, limit=limit, hard=False)
    return Criterion(
        name, Status.WARN,
        "High, but not harmful on its own while no swapping occurs.",
        measured=measured, limit=limit, hard=False,
    )


def evaluate_reliability(
    *,
    cpu_efficiency: float | None = None,
    n_threads: int = 1,
    involuntary_ctx_switches: int | None = None,
    total_cpu_s: float | None = None,
    ctx_switch_baseline_per_cpu_s: float | None = None,
    load_avg_1m_pre: float | None = None,
    load_avg_1m_post: float | None = None,
    physical_cores: int | None = None,
    swap_in_pages: int | None = None,
    swap_out_pages: int | None = None,
    thermal_throttle_events: int | None = None,
    ram_used_fraction: float | None = None,
) -> ReliabilityVerdict:
    """Evaluate all reliability criteria and return an aggregate verdict.

    All inputs are optional; any that are ``None`` yield an ``UNKNOWN`` criterion
    that cannot reject the run. See the module docstring for the full model.

    Parameters
    ----------
    cpu_efficiency:
        ``total_cpu_s / wall_time_s`` for the run (≈ 1.0 for a clean single thread).
    n_threads:
        Number of benchmark threads; scales the efficiency floor (default 1).
    involuntary_ctx_switches, total_cpu_s, ctx_switch_baseline_per_cpu_s:
        The switch count is normalised by ``total_cpu_s`` and compared against the
        baseline (switches per CPU-second from prior clean runs). Advisory only.
    load_avg_1m_pre, load_avg_1m_post:
        1-minute load average sampled before / after the run; the peak is tested.
    physical_cores:
        Physical core count the load is compared against.
    swap_in_pages, swap_out_pages:
        Pages swapped in/out *during* the run (deltas, not absolute levels).
    thermal_throttle_events:
        Throttle events accumulated during the run.
    ram_used_fraction:
        Fraction of RAM in use (0–1); advisory warning above 90%.
    """
    criteria = (
        _eval_cpu_efficiency(cpu_efficiency, n_threads),
        _eval_load(load_avg_1m_pre, load_avg_1m_post, physical_cores),
        _eval_swap(swap_in_pages, swap_out_pages),
        _eval_thermal(thermal_throttle_events),
        _eval_ctx_switches(involuntary_ctx_switches, total_cpu_s, ctx_switch_baseline_per_cpu_s),
        _eval_ram(ram_used_fraction),
    )
    return ReliabilityVerdict(criteria=criteria)
