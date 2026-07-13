"""End-to-end tests for the Overview tab's Streamlit render flow.

Drives ``detectors_overview.render`` through ``streamlit.testing.v1.AppTest``
with the remote_cache fetchers stubbed (no network), covering what the pure
helper tests in ``test_dashboard_detectors_overview.py`` cannot: widget
wiring, session-state keys, the reliability warning/toggle, and rerenders on
control changes.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from k4bench.regression.models import (  # noqa: E402
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)
from k4bench.regression.render import to_json  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"


def _verdict(det: str, metric: str, value: float, **kw) -> MetricVerdict:
    base = dict(
        detector=det, platform="PLAT", sample="single_e_10GeV", label="baseline_all",
        metric_family="time", metric=metric, sub_detector=None,
        run_id="2026-07-11", run_date="2026-07-11", value=value,
        baseline_median=value, baseline_mad=0.1, pct_change=0.0, z_score=0.0,
        severity=Severity.OK, direction=Direction.NONE, reason="ok",
    )
    base.update(kw)
    return MetricVerdict(**base)


def _report(night: str, *, scale: float = 1.0, reliable: bool | None = True) -> dict:
    groups = []
    for det, f in (("CLD_o2_v08", 1.0), ("IDEA_o1_v03", 1.4), ("SiD", 0.7)):
        groups.append(RunGroupReport(
            detector=det, platform="PLAT", sample="single_e_10GeV",
            k4h_release=f"key4hep-{night}", run_date=night, run_id=night,
            reliable=reliable,
            verdicts=[
                _verdict(det, "wall_time_s", 100.0 * f * scale),
                _verdict(det, "user_cpu_s", 90.0 * f * scale),
                _verdict(det, "peak_rss_mb", 1500.0 * f),
                _verdict(det, "median_time_s", 0.5 * f * scale),
                _verdict(det, "mean_time_s", 0.6 * f * scale),
                _verdict(det, "mean_rss_mb", 1200.0 * f),
            ],
        ))
    return to_json(NightlyReport(generated_at=f"{night}T06:00:00+00:00", groups=groups))


DATES = ["2026-07-11", "2026-07-10", "2026-07-09"]
#: The middle night failed the host reliability check — exercises the
#: exclude-unreliable warning/toggle.
REPORTS = {
    "2026-07-11": _report("2026-07-11"),
    "2026-07-10": _report("2026-07-10", scale=1.05, reliable=False),
    "2026-07-09": _report("2026-07-09", scale=1.1),
}


#: Covers the fixture's full 3-day span — the window itself is the sidebar's
#: shared Trend window, resolved by ``app.py`` and passed in as a plain
#: (start, end) tuple, not a control this tab owns.
_WINDOW = (date(2026, 7, 9), date(2026, 7, 11))


def _app(dashboard_dir, dates, reports, window) -> None:
    # No type annotations here: AppTest.from_function execs this body as a
    # bare script (the "def _app(...):" line is stripped), so a parameter
    # annotation referencing an unimported name would fail at exec time.
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    from tabs import detectors_overview as ov

    ov._cached_list_report_dates = lambda url: dates
    ov._cached_fetch_reports = lambda url, nights: {
        n: reports[n] for n in nights if n in reports
    }
    ov.render("https://example.invalid", "PLAT", "single_e_10GeV", window)


def _run(window: tuple[date, date] | None = _WINDOW) -> AppTest:
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), DATES, REPORTS, window), default_timeout=30
    )
    at.run()
    assert not at.exception, at.exception
    return at


def test_page_renders_controls_and_figure():
    at = _run()
    assert len(at.get("plotly_chart")) == 2  # history figure + landscape figure
    # Palette/opacity controls and the tab's own history-window dropdown were
    # dropped — the window now comes from the sidebar's shared Trend window.
    # Metric selectors and toggles share a single, evenly-spaced row.
    assert {s.label for s in at.selectbox} == {"Time metric", "Memory metric"}
    assert not at.slider
    assert {t.label for t in at.toggle} == {"Exclude unreliable runs"}
    assert {p.label for p in at.pills} == {"Regressions"}
    assert {c.label for c in at.segmented_control} == {"Scale"}
    captions = "\n".join(str(c.value) for c in at.caption)
    assert "Latest night: **2026-07-11**" in captions
    assert "**2026-07-09** → **2026-07-11** (3 nights)" in captions
    # The selected comparison is deep-linkable (AppTest stores param values
    # as lists).
    assert at.query_params["tmetric"] == ["mean_time_s"]
    assert at.query_params["mmetric"] == ["mean_rss_mb"]


def test_unreliable_night_warned_and_excludable():
    at = _run()
    warnings = "\n".join(str(w.value) for w in at.warning)
    # One night × three detectors failed the host check, on by default.
    assert "3 unreliable runs detected" in warnings
    assert "2026-07-10" in warnings
    toggle = at.toggle(key="det_ov_exclude_unreliable")
    assert toggle.value is True
    # Including them re-renders without error.
    toggle.set_value(False).run()
    assert not at.exception, at.exception


def _report_unjudged(night: str) -> dict:
    """The real-world shape of a failed night: it is *not judged*, so its group
    carries reliable=False with an empty verdict list — the case that
    regressed, since the flag then lives only on the group, not on any metric
    row that report_metrics_frame would surface."""
    groups = [RunGroupReport(
        detector="CLD_o2_v08", platform="PLAT", sample="single_e_10GeV",
        k4h_release=f"key4hep-{night}", run_date=night, run_id=night,
        reliable=False, verdicts=[],
        notes=["tonight's run failed the host reliability check"],
    )]
    return to_json(NightlyReport(generated_at=f"{night}T06:00:00+00:00", groups=groups))


def test_unreliable_night_with_no_verdicts_still_warns():
    # Regression guard: an unreliable night contributes no metric verdict rows,
    # yet the warning must still fire — it now reads the group-level flag, not
    # the (absent) verdict rows. The latest night has data so the tab renders.
    dates = ["2026-07-11", "2026-07-10"]
    reports = {
        "2026-07-11": _report("2026-07-11"),
        "2026-07-10": _report_unjudged("2026-07-10"),
    }
    at = AppTest.from_function(
        _app, args=(str(_DASHBOARD_DIR), dates, reports,
                    (date(2026, 7, 10), date(2026, 7, 11))),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    warnings = "\n".join(str(w.value) for w in at.warning)
    assert "1 unreliable run detected" in warnings
    assert "2026-07-10" in warnings


def test_control_changes_rerender():
    at = _run()
    at.selectbox(key="det_ov_time_metric").set_value("wall_time_s").run()
    assert not at.exception, at.exception
    at.segmented_control(key="det_ov_scale").set_value("Relative %").run()
    assert not at.exception, at.exception
    at.pills(key="det_ov_flags").set_value(["⚠️ Watch"]).run()
    assert not at.exception, at.exception


def test_narrower_window_still_renders():
    # A window that excludes the older nights renders cleanly (still has the
    # latest night for the snapshot) — the window is just a passed-in tuple
    # now, resolved upstream by the sidebar.
    at = _run(window=(date(2026, 7, 11), date(2026, 7, 11)))
    assert not at.exception, at.exception
    assert len(at.get("plotly_chart")) == 2


def test_no_window_falls_back_to_latest_nights():
    # ``window=None`` (e.g. the sidebar hasn't resolved one yet) falls back
    # to the latest nights via nights_in_window, not an error.
    at = _run(window=None)
    assert not at.exception, at.exception
    assert len(at.get("plotly_chart")) == 2
