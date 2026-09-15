"""The metrics k4Bench shares across views, and what is intrinsic to each.

Scope: every metric the regression report records, plus those a shared view
(dashboard tab, e-group mail, analysis figure) names, formats or ranks. It is
not an exhaustive schema of the results files: raw diagnostics that no shared
view displays (e.g. ``major_page_faults``, ``voluntary_ctx_switches``) and
view-local derived statistics (e.g. Event Memory's ``median_rss_mb``) are not
listed, and fall back to their raw name like any unknown column.

One :class:`MetricSpec` per metric holds what does not depend on who displays
it: the human name, the stored unit, how a value of it reads (a memory size, a
fraction, a count) and which direction is better. The consumers above read
these facts from here, so a metric they share is named and unit-labelled once.

Decisions that belong to one view or one policy stay with that view or policy:
which metrics are judged or reported-only and their regression family
(:mod:`k4bench.regression.report_builder`), which metrics a tab plots and in
what order, a tab's statistic panel titles, short control labels.

Like :mod:`k4bench.labels` this is a leaf module: its consumers sit in layers
that must not import each other, so it imports nothing from k4bench. An
unrecognized column falls back to its raw name and no unit rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: How a value reads, beyond its unit:
#:
#: - ``"quantity"`` — a number followed by its unit (seconds, MB/event, ev/s);
#: - ``"megabytes"`` — a size stored in MB that may be shown in GB when large;
#: - ``"ratio"`` — a fraction, shown as a percentage;
#: - ``"count"`` — an integer.
MetricKind = Literal["quantity", "megabytes", "ratio", "count"]


@dataclass(frozen=True)
class MetricSpec:
    """Intrinsic description of one measured column."""

    #: Sentence-case name, so a caller can drop it into a title or list item.
    label: str
    #: Unit of the stored values; empty for dimensionless columns.
    unit: str = ""
    kind: MetricKind = "quantity"
    #: ``True`` if smaller is better, ``False`` if larger is, ``None`` if
    #: neither direction is an improvement.
    lower_is_better: bool | None = True


METRICS: dict[str, MetricSpec] = {
    "wall_time_s": MetricSpec("Wall time", "s"),
    "user_cpu_s": MetricSpec("User CPU time", "s"),
    "sys_cpu_s": MetricSpec("System CPU time", "s"),
    "cpu_efficiency": MetricSpec("CPU efficiency", kind="ratio", lower_is_better=False),
    "peak_rss_mb": MetricSpec("Peak RSS", "MB", "megabytes"),
    "peak_vmem_mb": MetricSpec("Peak virtual memory", "MB", "megabytes"),
    "mean_rss_anon_mb": MetricSpec("Mean event anonymous RSS", "MB", "megabytes"),
    "mean_rss_file_mb": MetricSpec("Mean event file-backed RSS", "MB", "megabytes"),
    "rss_anon_slope_mb_per_event": MetricSpec("Anonymous RSS growth per event", "MB/event"),
    "mean_rss_mb": MetricSpec("Mean event RSS", "MB", "megabytes"),
    "mean_time_s": MetricSpec("Mean event time", "s"),
    "median_time_s": MetricSpec("Median event time", "s"),
    "trimmed_mean_time_s": MetricSpec("Trimmed mean event time", "s"),
    "output_size_mb": MetricSpec("Output size", "MB", "megabytes"),
    "events_per_sec": MetricSpec("Throughput", "ev/s", lower_is_better=False),
    "involuntary_ctx_switches": MetricSpec("Involuntary context switches", kind="count"),
    "returncode": MetricSpec("Return code", kind="count", lower_is_better=None),
}

#: ``metric -> label`` view of :data:`METRICS`, for callers that only need names.
METRIC_LABELS: dict[str, str] = {name: spec.label for name, spec in METRICS.items()}


def metric_label(metric: str) -> str:
    """Human-readable name of *metric*, or the raw column name if unknown."""
    spec = METRICS.get(metric)
    return spec.label if spec else metric


def metric_unit(metric: str) -> str:
    """Stored unit of *metric*; empty if dimensionless or unknown."""
    spec = METRICS.get(metric)
    return spec.unit if spec else ""


def metric_title(metric: str, unit: str | None = None) -> str:
    """Axis or panel title such as ``Wall time (s)``. *unit* overrides the
    stored unit, for a view that rescales (e.g. memory shown in GB)."""
    unit = metric_unit(metric) if unit is None else unit
    label = metric_label(metric)
    return f"{label} ({unit})" if unit else label


def is_megabytes(metric: str) -> bool:
    """Whether *metric* is a size stored in MB, which a view may show in GB."""
    spec = METRICS.get(metric)
    return spec is not None and spec.kind == "megabytes"


def pretty_metric(metric: str, sub_detector: str | None = None) -> str:
    """Human-readable metric name, suffixed with the sub-detector for a
    region-level row, e.g. ``("mean_rss_mb", "EMEC_turbine")`` ->
    ``Mean event RSS · EMEC_turbine``.

    The sub-detector keeps its raw name: it is a DD4hep DetElement identifier,
    which is what the dashboard labels the series with and what someone
    searching the geometry types."""
    name = metric_label(metric)
    return f"{name} · {sub_detector}" if sub_detector else name
