"""Tests for the Regressions tab's sidebar-scoped rendering.

The tab shows exactly the run group matching the sidebar's (detector,
platform, sample) — the cross-detector picture lives in the Overview tab —
opens the trend preview on the most severe flag, and keys the report night on
the sidebar's release: the newest release shows the latest report, an older
one the report of its last run. All remote calls are stubbed; nothing touches
the network.
"""

from __future__ import annotations

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

PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
NIGHT = "2026-07-10"
STACK = f"key4hep-{NIGHT}"


def _verdict(metric: str, severity: Severity, pct: float | None, **kw) -> MetricVerdict:
    base = dict(
        detector="CLD", platform=PLAT, sample="single_e", label="baseline",
        metric_family="time", metric=metric, sub_detector=None,
        run_id=NIGHT, run_date=NIGHT, value=6.0, baseline_median=5.0,
        baseline_mad=0.1, pct_change=pct, z_score=10.0, severity=severity,
        direction=Direction.UP if (pct or 0) >= 0 else Direction.DOWN,
        reason="step" if severity is not Severity.OK else "within baseline",
    )
    base.update(kw)
    return MetricVerdict(**base)


def _report(groups: list[RunGroupReport]) -> dict:
    return to_json(NightlyReport(generated_at=f"{NIGHT}T06:00:00+00:00", groups=groups))


def _group(detector: str = "CLD", sample: str = "single_e", *, verdicts=(),
           job_failures=()) -> RunGroupReport:
    return RunGroupReport(
        detector=detector, platform=PLAT, sample=sample,
        k4h_release=STACK, run_date=NIGHT, run_id=NIGHT,
        verdicts=list(verdicts), job_failures=list(job_failures),
    )


#: One flagged group: two confirmed (peak_rss the larger |Δ|), one watch whose
#: |Δ| beats both — the preview must still open on a *confirmed* metric.
_FLAGGED = [
    _verdict("wall_time_s", Severity.CONFIRMED, 0.20),
    _verdict("peak_rss_mb", Severity.CONFIRMED, -0.35),
    _verdict("mean_time_s", Severity.WATCH, 0.50),
    _verdict("user_cpu_s", Severity.OK, 0.01),
]


def _app(dashboard_dir, reports_map, blame_map, dates, stacks_dates,
         detector, platform, sample, stack):
    """The tab, rendered standalone with every remote call stubbed.

    ``AppTest.from_function`` re-executes this source in its own script
    context, so it can close over nothing: imports and stubs all live inside
    it, set on the ``tabs.regressions`` module the script itself imports.
    """
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)

    from tabs import regressions as _tab

    _tab._cached_list_report_dates = lambda url: list(dates)
    _tab._cached_fetch_report = lambda url, night: reports_map.get(night)
    # Absent on most nights; the blame sidecar is best-effort, so the stub
    # returns None unless a test supplies one.
    _tab._cached_fetch_blame = lambda url, night: blame_map.get(night)
    # Doubles as both the report-night lookup (which release ran when) and the
    # trend preview's history fetch; the windowed-run fetch below returns
    # nothing so the drill-down warns instead of downloading anything.
    _tab._cached_list_run_dates = (
        lambda url, det, plat, samp: {k: list(v) for k, v in stacks_dates}
    )
    _tab._cached_fetch_runs_windowed = lambda *a, **k: ()
    # The forward blame card's live provenance lookup; stubbed to avoid the
    # network so a windowed verdict's card renders without a real EOS fetch.
    _tab.packages_for_release = lambda data_url, platform, release: None
    _tab.render(
        "https://example.invalid", "/tmp/cache", detector, platform, sample, stack,
    )


def _run(report_json=None, detector="CLD", platform=PLAT, sample="single_e",
         stack=STACK, dates=(NIGHT,), stacks_dates=None,
         reports_map=None, blame_map=None) -> AppTest:
    if reports_map is None:
        reports_map = {d: report_json for d in dates}
    at = AppTest.from_function(
        _app,
        args=(
            str(_DASHBOARD_DIR), reports_map, blame_map or {}, tuple(dates),
            tuple((k, tuple(v)) for k, v in (stacks_dates or {}).items()),
            detector, platform, sample, stack,
        ),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    return at


# ── the sidebar scope selects one run group ───────────────────────────────────

def test_scoped_group_renders_flat_without_expanders():
    at = _run(_report([_group(verdicts=_FLAGGED),
                       _group(detector="IDEA", verdicts=[])]))
    # One group, rendered flat: the flag table shows, nothing hides in an
    # expander, and the other detector's group leaves no trace.
    assert not at.expander
    assert len(at.dataframe) == 1
    flags = at.dataframe[0].value
    assert len(flags) == 3  # the OK verdict earns no row
    assert "IDEA" not in at.markdown[0].value if at.markdown else True


def test_banner_counts_the_groups_verdicts():
    at = _run(_report([_group(verdicts=_FLAGGED, job_failures=["no results"])]))
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["🔴 Regressed"] == "2"
    assert by_label["⚠️ Watch"] == "1"
    assert by_label["❌ Failures"] == "1"
    assert by_label["✅ Within baseline"] == "1"
    assert "Detectors checked" not in by_label  # cross-detector count is Overview's


def test_trend_preview_defaults_to_the_worst_confirmed_flag():
    at = _run(_report([_group(verdicts=_FLAGGED)]))
    # The night dropdown is gone (the report is keyed on the sidebar's
    # release), so the trend preview is the tab's only selectbox. It opens on
    # the largest-|Δ| CONFIRMED metric — not "—", and not the larger-|Δ| WATCH.
    (preview,) = at.selectbox
    assert "peak_rss_mb" in preview.value
    assert preview.value.startswith("🔴")
    # The drill-down actually rendered (its history fetch found no runs).
    assert any("No history could be loaded" in w.value for w in at.warning)


def test_scope_miss_names_the_detectors_other_groups():
    at = _run(
        _report([_group(sample="p8_ee_Zbb_ecm91", verdicts=_FLAGGED)]),
        sample="single_e",
    )
    body = " ".join(i.value for i in at.info)
    assert "single_e" in body            # what the sidebar asked for
    assert "Z → bb" in body              # where the report has data instead
    assert not at.metric                 # no banner for a scope miss


def test_missing_detector_names_the_covered_ones():
    at = _run(_report([_group(detector="IDEA", verdicts=[])]), detector="CLD")
    body = " ".join(i.value for i in at.info)
    assert "CLD" in body and "IDEA" in body


# ── the report night is keyed on the sidebar's release ───────────────────────

def test_newest_release_shows_the_latest_report():
    # The selected release owns the triple's newest run → the latest report,
    # even when it is newer than that run (a missing-run night's failure must
    # stay visible).
    at = _run(
        reports_map={"2026-07-11": _report([_group(
            verdicts=[], job_failures=["no run uploaded for 2026-07-11"],
        )])},
        dates=("2026-07-11", NIGHT),
        stacks_dates={STACK: [NIGHT]},
    )
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["❌ Failures"] == "1"
    assert not any("Historical view" in c.value for c in at.caption)


def test_older_release_shows_the_report_of_its_last_run():
    # Two nights re-benchmarked the older release; its *last* run's report is
    # the most settled judgement of that fixed stack, so that is the night
    # shown — flagged as a historical view.
    old_stack = "key4hep-2026-07-05"
    old_group = RunGroupReport(
        detector="CLD", platform=PLAT, sample="single_e",
        k4h_release=old_stack, run_date="2026-07-07", run_id="2026-07-07",
        verdicts=[_verdict("wall_time_s", Severity.OK, 0.0,
                           run_id="2026-07-07", run_date="2026-07-05")],
    )
    at = _run(
        reports_map={
            NIGHT: _report([_group(verdicts=_FLAGGED)]),
            "2026-07-07": to_json(NightlyReport(generated_at="", groups=[old_group])),
        },
        dates=(NIGHT, "2026-07-07", "2026-07-06"),
        stacks_dates={old_stack: ["2026-07-06", "2026-07-07"], STACK: [NIGHT]},
        stack=old_stack,
    )
    captions = " ".join(c.value for c in at.caption)
    assert "Historical view" in captions and "2026-07-07" in captions
    # The older night's (quiet) report rendered, not the latest one's flags.
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["🔴 Regressed"] == "0"
    assert by_label["✅ Within baseline"] == "1"


def test_run_listing_failure_falls_back_with_a_visible_warning():
    # A network/listing failure must not silently swap in the latest report
    # under an old-release selection without saying so.
    def _app(dashboard_dir, report_json, night, platform):
        import sys as _sys
        if dashboard_dir not in _sys.path:
            _sys.path.insert(0, dashboard_dir)
        import requests as _requests
        from tabs import regressions as _tab

        _tab._cached_list_report_dates = lambda url: [night]
        _tab._cached_fetch_report = lambda url, n: report_json

        # Only the report-night resolution (the first call) should fail here;
        # the trend preview's own (unrelated) history fetch, triggered by the
        # same auto-opened drilldown, must still be free to run and come up
        # empty on its own terms.
        calls = {"n": 0}
        def _raise_once(url, det, plat, samp):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _requests.ConnectionError("EOS unreachable")
            return {}
        _tab._cached_list_run_dates = _raise_once
        _tab._cached_fetch_runs_windowed = lambda *a, **k: ()
        _tab.render(
            "https://example.invalid", "/tmp/cache", "CLD", platform, "single_e",
            f"key4hep-{night}",
        )

    at = AppTest.from_function(
        _app,
        args=(str(_DASHBOARD_DIR), _report([_group(verdicts=_FLAGGED)]), NIGHT, PLAT),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    assert any("Could not check" in w.value for w in at.warning)
    by_label = {m.label: m.value for m in at.metric}
    assert by_label["🔴 Regressed"] == "2"  # still rendered, from the fallback


def test_release_older_than_the_first_report_says_so():
    at = _run(
        reports_map={NIGHT: _report([_group(verdicts=_FLAGGED)])},
        dates=(NIGHT,),
        stacks_dates={"key4hep-2026-06-01": ["2026-06-01"], STACK: [NIGHT]},
        stack="key4hep-2026-06-01",
    )
    body = " ".join(i.value for i in at.info)
    assert "No nightly report covers" in body and "2026-06-01" in body
    assert not at.metric


# ── candidate PRs from the blame sidecar ─────────────────────────────────────

def _windowed_confirmed() -> MetricVerdict:
    """A confirmed wall-time regression with a real (baseline, onset] window."""
    return _verdict(
        "wall_time_s", Severity.CONFIRMED, 0.20,
        onset_run_id="2026-07-04", onset_run_date="2026-07-04",
        last_accepted_run_id="2026-07-01", last_accepted_run_date="2026-07-01",
    )


def _blame_json(candidates) -> dict:
    from k4bench.blame.models import BlameEntry, BlameReport, RepoBlame
    entry = BlameEntry(
        detector="CLD", platform=PLAT, sample="single_e", label="baseline",
        metric="wall_time_s", sub_detector=None,
        base_release="2026-07-01", onset_release="2026-07-04",
        repos=(RepoBlame(
            package="k4geo", repo="key4hep/k4geo",
            base_commit="a" * 40, head_commit="c" * 40,
            compare_url="https://github.com/key4hep/k4geo/compare/a...c",
            status="changed", candidates=tuple(candidates),
        ),),
        n_unchanged=60,
    )
    return BlameReport(
        generated_at=f"{NIGHT}T06:00:00+00:00", report_night=NIGHT, entries=(entry,)
    ).to_json()


def test_ranked_candidate_prs_render_when_a_blame_sidecar_is_present():
    from k4bench.blame.models import CandidatePR
    cand = CandidatePR(
        repo="key4hep/k4geo", number=1234, title="Lower the tracker step limit",
        author="alice", url="https://github.com/key4hep/k4geo/pull/1234",
        merged_at="2026-07-04T10:00:00Z", files=("FCCee/CLD/compact/x.xml",),
        additions=20, deletions=4, score=72.0,
        description="raises tracker step count, plausibly slower",
    )
    at = _run(
        _report([_group(verdicts=[_windowed_confirmed()])]),
        blame_map={NIGHT: _blame_json([cand])},
    )
    captions = " ".join(c.value for c in at.caption)
    assert "Suggested cause" in captions
    pr_frames = [d.value for d in at.dataframe if "Pull request" in d.value.columns]
    assert len(pr_frames) == 1
    assert "key4hep/k4geo#1234" in list(pr_frames[0]["Pull request"])


def test_unranked_candidates_show_no_table():
    # The interim: the blame sidecar carries the candidate PRs but the ranking
    # stage has not scored/described them yet. The forward package-diff card
    # still renders; the candidate ledger stays hidden until there is a ranking.
    from k4bench.blame.models import CandidatePR
    cand = CandidatePR(
        repo="key4hep/k4geo", number=1234, title="Some PR", author="alice",
        url="https://github.com/key4hep/k4geo/pull/1234",
    )
    at = _run(
        _report([_group(verdicts=[_windowed_confirmed()])]),
        blame_map={NIGHT: _blame_json([cand])},
    )
    assert not any("Pull request" in d.value.columns for d in at.dataframe)


def test_no_candidate_table_without_a_blame_sidecar():
    # The common case: a confirmed regression whose night has no blame.json.
    at = _run(_report([_group(verdicts=[_windowed_confirmed()])]))
    assert not any("Pull request" in d.value.columns for d in at.dataframe)


def test_malformed_blame_sidecar_hides_blame_instead_of_crashing():
    # Valid JSON, wrong shape (entries missing required fields): the page must
    # render as if the night had no sidecar at all.
    at = _run(
        _report([_group(verdicts=[_windowed_confirmed()])]),
        blame_map={NIGHT: {"entries": [{"detector": "CLD"}]}},
    )
    assert not any("Pull request" in d.value.columns for d in at.dataframe)


def test_stale_blame_with_a_different_window_is_not_joined():
    # Same verdict identity, but the sidecar attributes an older window (e.g. a
    # rerun re-anchored the step): its ranking must not attach.
    from k4bench.blame.models import CandidatePR
    cand = CandidatePR(
        repo="key4hep/k4geo", number=1234, title="Some PR", author="alice",
        url="https://github.com/key4hep/k4geo/pull/1234",
        score=90.0, description="from another window",
    )
    stale = _blame_json([cand])
    stale["entries"][0]["base_release"] = "2026-06-20"
    stale["entries"][0]["onset_release"] = "2026-06-25"
    at = _run(
        _report([_group(verdicts=[_windowed_confirmed()])]),
        blame_map={NIGHT: stale},
    )
    assert not any("Pull request" in d.value.columns for d in at.dataframe)
