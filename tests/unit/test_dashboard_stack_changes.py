"""Tests for the Stack Changes tab.

Covers the release-vs-release contract the tab rests on (never run dates), the
identical-stack case that is the tab's most useful answer, and the app-level
registration — including the remote-only section slice, which is an off-by-one
away from exposing a tab that cannot work in local mode.

All remote calls are stubbed; nothing touches the network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"

PLAT = "x86_64-almalinux9-gcc14.2.0-opt"


def _load_module():
    if str(_DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(_DASHBOARD_DIR))
    spec = importlib.util.spec_from_file_location(
        "k4bench_dashboard_stack_changes", _DASHBOARD_DIR / "tabs" / "stack_changes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stack_changes = _load_module()


def _pkg(commit: str, url: str = "https://github.com/key4hep/k4geo.git") -> dict:
    return {"commit": commit, "version": "develop", "repo_url": url}


#: 07-08 and 07-09 are the identical stack (the nightly-lag case); k4geo moves
#: in 07-10, so the tab's default view (the two newest) shows a real diff.
_STACKS = {
    "key4hep-2026-07-08": {"k4geo": _pkg("a" * 40), "dd4hep": _pkg("b" * 40)},
    "key4hep-2026-07-09": {"k4geo": _pkg("a" * 40), "dd4hep": _pkg("b" * 40)},
    "key4hep-2026-07-10": {"k4geo": _pkg("c" * 40), "dd4hep": _pkg("b" * 40)},
}


def _app(dashboard_dir, stack_names, packages, from_release, to_release,
         report_dates, reports_map, detector, sample):
    """The tab, rendered standalone with every remote call stubbed.

    ``AppTest.from_function`` re-executes this source in its own script
    context, so it can close over nothing: the imports, the stubs, and the
    platform literal all have to live inside it. The stubs are set on the
    ``tabs.stack_changes`` module the script itself imports — patching any
    other instance of it would not be seen from here.
    """
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as _st

    from tabs import stack_changes as _tab

    _tab._cached_list_detectors = lambda url: ["IDEA"]
    _tab._cached_list_stacks = lambda url, detector, platform: stack_names
    _tab._cached_fetch_stack_packages = (
        lambda url, detector, platform, stack: packages.get(stack)
    )
    # The reverse view reads the nightly reports; stub those too so the tab is
    # hermetic (default: no reports, so the reverse section is empty).
    _tab._cached_list_report_dates = lambda url: report_dates
    _tab._cached_fetch_report = lambda url, date: reports_map.get(date)

    if from_release:
        _st.query_params["from"] = from_release
    if to_release:
        _st.query_params["to"] = to_release
    _tab.render(
        "https://example.invalid", "x86_64-almalinux9-gcc14.2.0-opt",
        detector, sample,
    )


def _run(stack_names=None, packages=None, from_release=None, to_release=None,
         report_dates=(), reports_map=None, detector="CLD",
         sample="single_e") -> AppTest:
    at = AppTest.from_function(
        _app,
        args=(
            str(_DASHBOARD_DIR),
            sorted(_STACKS, reverse=True) if stack_names is None else stack_names,
            _STACKS if packages is None else packages,
            from_release,
            to_release,
            tuple(report_dates),
            reports_map or {},
            detector,
            sample,
        ),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    return at


# ── the release-vs-release contract ──────────────────────────────────────────

def test_release_strips_the_directory_prefix():
    # The tab talks in nightly tags; EOS stores them as key4hep-{date} dirs.
    assert stack_changes._release("key4hep-2026-07-10") == "2026-07-10"
    assert stack_changes._release("2026-07-10") == "2026-07-10"


def _query(url: str) -> dict:
    from urllib.parse import parse_qs, urlsplit
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def _confirmed(**kw):
    from k4bench.regression.models import Direction, MetricVerdict, Severity
    base = dict(
        detector="CLD", platform=PLAT, sample="single_e", label="baseline",
        metric_family="time", metric="wall_time_s", sub_detector=None,
        run_id="2026-06-27", run_date="2026-06-27", value=6.0, baseline_median=5.0,
        baseline_mad=0.1, pct_change=0.2, z_score=10.0, severity=Severity.CONFIRMED,
        direction=Direction.UP, reason="step", onset_run_id="2026-06-26",
        onset_run_date="2026-06-25", last_accepted_run_id="2026-06-25",
        last_accepted_run_date="2026-06-24",
    )
    base.update(kw)
    return MetricVerdict(**base)


def _raw_report(verdicts, night="2026-06-27"):
    from k4bench.regression.models import NightlyReport, RunGroupReport
    from k4bench.regression.render import to_json
    group = RunGroupReport(
        detector="CLD", platform=PLAT, sample="single_e",
        k4h_release=f"key4hep-{night}", run_date=night, run_id=night,
        verdicts=verdicts,
    )
    return to_json(NightlyReport(generated_at="", groups=[group]))


def test_regressions_in_range_filters_by_range_and_platform():
    reports = [_raw_report([
        _confirmed(metric="wall_time_s"),                     # in range
        _confirmed(metric="peak_rss_mb", onset_run_date="2026-07-05"),  # onset after range
        _confirmed(metric="mean_time_s", platform="other-plat"),        # other platform
    ])]
    hits = stack_changes._regressions_in_range(reports, PLAT, "2026-06-24", "2026-06-25")
    assert [v.metric for v in hits] == ["wall_time_s"]


def test_regressions_in_range_dedups_on_the_onset_run_not_its_release():
    reports = [_raw_report([
        _confirmed(onset_run_id="run-A"),
        _confirmed(onset_run_id="run-A"),   # exact duplicate → collapsed
        _confirmed(onset_run_id="run-B"),   # same onset release, different run → kept
    ])]
    hits = stack_changes._regressions_in_range(reports, PLAT, "2026-06-24", "2026-06-25")
    assert {v.onset_run_id for v in hits} == {"run-A", "run-B"}


def test_regressions_in_range_orders_by_magnitude_with_unknown_last():
    reports = [_raw_report([
        _confirmed(metric="mean_time_s", pct_change=0.05),
        _confirmed(metric="cpu_efficiency", pct_change=None),  # absolute-floor: no %
        _confirmed(metric="wall_time_s", pct_change=-0.30),
    ])]
    hits = stack_changes._regressions_in_range(reports, PLAT, "2026-06-24", "2026-06-25")
    # Largest |Δ| first; the None-magnitude one sorts last, not as if it were 0%.
    assert [v.metric for v in hits] == ["wall_time_s", "mean_time_s", "cpu_efficiency"]


def test_regressions_in_range_excludes_same_release_windows():
    # A regression whose onset and baseline are the SAME release had no stack
    # change, so it is not a candidate effect of a diff and must not appear with
    # a nonsensical X → X window.
    reports = [_raw_report([
        _confirmed(metric="wall_time_s", last_accepted_run_date="2026-06-25"),  # == onset
        _confirmed(metric="peak_rss_mb"),  # normal bounded window 06-24 → 06-25
    ])]
    hits = stack_changes._regressions_in_range(reports, PLAT, "2026-06-24", "2026-06-25")
    assert [v.metric for v in hits] == ["peak_rss_mb"]


def test_regressions_in_range_skips_missing_reports():
    # A night whose report could not be fetched is None and must be tolerated.
    hits = stack_changes._regressions_in_range(
        [None, _raw_report([_confirmed()])], PLAT, "2026-06-24", "2026-06-25")
    assert len(hits) == 1


def test_packages_for_release_prefixes_the_release_tag(monkeypatch):
    seen = {}
    monkeypatch.setattr(stack_changes, "_packages",
                        lambda url, plat, stack: seen.setdefault("stack", stack))
    stack_changes.packages_for_release("url", PLAT, "2026-07-10")
    assert seen["stack"] == "key4hep-2026-07-10"


def test_reverse_view_renders_the_regression_table():
    # A confirmed regression whose onset falls in the diffed range, with a
    # relative-% metric and an absolute-floor metric (no %). Both should list;
    # the %-less one must not show +0.0%.
    report = _raw_report([
        _confirmed(metric="wall_time_s", pct_change=0.10,
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
        _confirmed(metric="cpu_efficiency", metric_family="cpu_efficiency_pp",
                   pct_change=None, onset_run_id="run-eff",
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
    ])
    at = _run(from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    # Two dataframes now: [0] the package diff, [1] the reverse regression table.
    assert len(at.dataframe) == 2
    reverse = at.dataframe[1].value
    # Scoped to the sidebar's detector/sample: the constant columns are gone,
    # and the ledger is the Regressions tab's plus the blame window.
    assert list(reverse.columns) == [
        "", "Config", "Metric", "Dir", "Δ vs baseline",
        "Current / baseline", "Blame window",
    ]
    assert list(reverse["Blame window"]) == ["2026-07-09 → 2026-07-10"] * 2
    deltas = list(reverse["Δ vs baseline"])
    assert 10.0 in deltas                     # the relative-% metric (|Δ|)
    assert any(pd.isna(d) for d in deltas)    # the %-less metric is blank, not +0.0%


def test_reverse_view_renders_even_when_stack_provenance_is_missing():
    # Provenance for the two releases has aged off CVMFS (packages empty), so the
    # package diff cannot be built — but the reverse table comes from the reports
    # and must still show, alongside the "cannot be diffed" warning.
    report = _raw_report([_confirmed(
        last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10")])
    at = _run(stack_names=["key4hep-2026-07-10", "key4hep-2026-07-09"], packages={},
              from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    assert any("cannot be diffed" in w.value for w in at.warning)
    assert len(at.dataframe) == 1  # no diff table, but the reverse table renders
    assert list(at.dataframe[0].value["Config"]) == ["baseline"]


def test_multi_release_range_warns_and_shows_each_regressions_own_window():
    # A multi-release diff where a regression's window is a sub-range of it —
    # the per-row window is what stops the cumulative diff being misread as
    # one night's change.
    report = _raw_report([_confirmed(
        last_accepted_run_date="2026-07-04", onset_run_date="2026-07-05")])  # inside 07-01..07-10
    at = _run(
        stack_names=["key4hep-2026-07-10", "key4hep-2026-07-05", "key4hep-2026-07-01"],
        packages={}, from_release="2026-07-01", to_release="2026-07-10",
        report_dates=("2026-07-05",), reports_map={"2026-07-05": report})
    reverse = at.dataframe[0].value  # provenance missing → only the reverse table
    assert list(reverse["Blame window"]) == ["2026-07-04 → 2026-07-05"]
    # …and a prominent warning makes the cumulative range crystal-clear.
    assert any("multi-release" in w.value for w in at.warning)


def test_neighbouring_releases_never_trigger_the_cumulative_warning():
    # The bug: a regression whose baseline predates the selected base (its window
    # is wider than the range) must NOT flip a consecutive comparison to
    # "cumulative". The trigger is the selected range spanning >1 release, not
    # any single regression's window. (The Blame window column itself is always
    # shown — it is the per-regression truth either way.)
    report = _raw_report([_confirmed(
        last_accepted_run_date="2026-07-05",  # older than the selected base (07-09)
        onset_run_date="2026-07-10")])
    at = _run(packages={}, from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    reverse = at.dataframe[0].value
    assert list(reverse["Blame window"]) == ["2026-07-05 → 2026-07-10"]
    assert not any("multi-release" in w.value for w in at.warning)


def test_reverse_view_says_so_when_no_regression_has_onset_in_range():
    at = _run(from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",),
              reports_map={"2026-07-10": _raw_report([
                  _confirmed(onset_run_date="2026-05-01")])})  # onset far before range
    assert any("No confirmed regression" in m.value for m in at.info)


def test_deep_link_carries_detector_so_the_app_can_resolve_the_platform():
    # Regressions is cross-detector; the app resolves the platform list from the
    # selected detector, so a deep link that seeds a platform without a detector
    # offering it would be rejected. Detector must ride along.
    q = _query(stack_changes.deep_link(
        detector="CLD_o2_v08", platform=PLAT,
        base_release="2026-06-24", head_release="2026-06-25",
    ))
    assert q == {
        "tab": "Stack Changes", "detector": "CLD_o2_v08", "platform": PLAT,
        "from": "2026-06-24", "to": "2026-06-25",
    }


def test_deep_link_omits_from_for_an_open_ended_window():
    q = _query(stack_changes.deep_link(
        detector="CLD_o2_v08", platform=PLAT, head_release="2026-06-25",
    ))
    assert "from" not in q
    assert q["to"] == "2026-06-25"


def test_from_default_avoids_a_reversed_range_when_only_to_is_seeded(monkeypatch):
    # Newest-first. An open-ended blame link seeds only ?to=; the From default
    # must be *older* than that To, not the usual second-newest (which would be
    # newer than an old onset and trip the reversed-range warning).
    releases = ["2026-07-10", "2026-07-05", "2026-06-25", "2026-06-20"]
    monkeypatch.setattr(stack_changes.st, "query_params", {"to": "2026-06-25"})
    assert stack_changes._from_default_for(releases) == "2026-06-20"  # one older than To


def test_from_default_is_second_newest_without_a_seed(monkeypatch):
    releases = ["2026-07-10", "2026-07-05", "2026-06-25"]
    monkeypatch.setattr(stack_changes.st, "query_params", {})
    assert stack_changes._from_default_for(releases) == "2026-07-05"


def test_from_default_when_to_is_the_oldest_release_does_not_run_off_the_end(monkeypatch):
    releases = ["2026-07-10", "2026-07-05", "2026-06-25"]
    monkeypatch.setattr(stack_changes.st, "query_params", {"to": "2026-06-25"})
    # No release older than the oldest To — fall back to To itself (the tab then
    # shows "pick two different releases" rather than crashing on an index).
    assert stack_changes._from_default_for(releases) == "2026-06-25"


def test_stacks_are_unioned_across_detectors(monkeypatch):
    # Detectors join and leave the matrix, so no single detector's history is
    # the full set of releases.
    per_detector = {"IDEA": ["key4hep-2026-07-10"], "SiD": ["key4hep-2026-07-09"]}
    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA", "SiD"])
    monkeypatch.setattr(
        stack_changes, "_cached_list_stacks",
        lambda url, detector, platform: per_detector[detector],
    )
    assert stack_changes._stacks_for_platform("u", PLAT) == [
        "key4hep-2026-07-10", "key4hep-2026-07-09",
    ]


def test_a_detector_without_the_platform_is_skipped(monkeypatch):
    import requests

    def _stacks(url, detector, platform):
        if detector == "SiD":
            raise requests.RequestException("404")  # never ran on this platform
        return ["key4hep-2026-07-10"]

    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA", "SiD"])
    monkeypatch.setattr(stack_changes, "_cached_list_stacks", _stacks)
    # One detector missing a platform must not blank the whole tab.
    assert stack_changes._stacks_for_platform("u", PLAT) == ["key4hep-2026-07-10"]


def test_an_unexpected_listing_error_is_not_swallowed(monkeypatch):
    def _stacks(url, detector, platform):
        raise TypeError("a bug in the listing code")

    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA"])
    monkeypatch.setattr(stack_changes, "_cached_list_stacks", _stacks)
    # Only an absent directory is expected here. Anything else is a bug, and
    # must surface rather than read as "this platform has no releases".
    with pytest.raises(TypeError):
        stack_changes._stacks_for_platform("u", PLAT)


def test_packages_fall_back_to_another_detector(monkeypatch):
    # A detector may have skipped a release, or run it before provenance
    # capture; any other detector's run answers for the same stack.
    def _fetch(url, detector, platform, stack):
        return _STACKS[stack] if detector == "SiD" else None

    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA", "SiD"])
    monkeypatch.setattr(stack_changes, "_cached_fetch_stack_packages", _fetch)
    assert stack_changes._packages("u", PLAT, "key4hep-2026-07-10") == _STACKS["key4hep-2026-07-10"]


def test_packages_none_when_no_detector_has_it(monkeypatch):
    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA"])
    monkeypatch.setattr(
        stack_changes, "_cached_fetch_stack_packages", lambda *a: None,
    )
    # Unknown must never read as "an empty stack".
    assert stack_changes._packages("u", PLAT, "key4hep-2026-07-10") is None


# ── how far apart the releases are ───────────────────────────────────────────

def test_consecutive_releases_say_nothing():
    # There is nothing to warn about, and a line restating the heading is noise.
    releases = ["2026-07-10", "2026-07-09", "2026-07-08"]
    assert stack_changes._span(releases, "2026-07-09", "2026-07-10") == ""


def test_a_wide_range_warns_that_the_diff_is_cumulative():
    # A month-wide diff looks identical to one night's in the table; without
    # this it reads as "these 21 packages changed last night".
    releases = [f"2026-07-{d:02d}" for d in range(10, 0, -1)]
    span = stack_changes._span(releases, "2026-07-01", "2026-07-10")
    assert "9 releases apart" in span and "cumulative" in span


# ── render ───────────────────────────────────────────────────────────────────

def test_defaults_to_the_two_newest_releases():
    # "What came in last night?" — defaulting both pickers to the newest would
    # open the tab on "pick two different releases" instead of an answer.
    at = _run()
    assert [s.value for s in at.selectbox] == ["2026-07-09", "2026-07-10"]
    assert at.dataframe, "the default view should show the diff, not a prompt"


def test_renders_the_diff_between_two_releases():
    at = _run(from_release="2026-07-09", to_release="2026-07-10")
    rendered = at.dataframe[0].value

    assert len(rendered) == 1, "only the moved package belongs here"
    row = rendered.iloc[0]
    # Identifiers are plain text; the compare view is the row's one action, and
    # it spans both commits so nothing is lost by being the only link.
    assert row["Package"] == "k4geo"
    assert row["From"] == "a" * 12
    assert row["To"] == "c" * 12
    assert row["Compare"] == (
        f"https://github.com/key4hep/k4geo/compare/{'a' * 40}...{'c' * 40}"
    )


def test_the_diff_reports_how_far_apart_the_releases_are():
    at = _run(from_release="2026-07-08", to_release="2026-07-10")
    captions = " ".join(c.value for c in at.caption)
    assert "2 releases apart" in captions and "cumulative" in captions


def test_the_branch_column_is_not_rendered():
    # Every package in every release sits on `develop`, so a branch column
    # would be one repeated value taking space from the SHAs.
    at = _run(from_release="2026-07-09", to_release="2026-07-10")
    assert "Branch" not in at.dataframe[0].value.columns


def test_packages_on_an_unknown_forge_still_render():
    # No compare view exists for a forge whose URL layout we do not know; the
    # package and its commits are still the answer to what moved.
    stacks = {
        "key4hep-2026-07-09": {"odd": {"commit": "a" * 40, "version": "develop",
                                       "repo_url": "https://git.example.com/a/b"}},
        "key4hep-2026-07-10": {"odd": {"commit": "c" * 40, "version": "develop",
                                       "repo_url": "https://git.example.com/a/b"}},
    }
    at = _run(stack_names=sorted(stacks, reverse=True), packages=stacks,
              from_release="2026-07-09", to_release="2026-07-10")
    row = at.dataframe[0].value.iloc[0]
    assert row["Package"] == "odd"
    assert row["From"] == "a" * 12
    assert row["Compare"] is None


def test_identical_releases_are_called_out_not_left_empty():
    # The nightly-lag case is the tab's most valuable answer: it rules an
    # upstream commit out entirely, rather than showing an empty table.
    at = _run(from_release="2026-07-08", to_release="2026-07-09")
    body = " ".join(s.value for s in at.success)
    assert "identical stack" in body and "nothing upstream changed" in body
    assert not at.dataframe


def test_reversed_range_is_refused_not_sign_flipped():
    at = _run(from_release="2026-07-10", to_release="2026-07-08")
    assert any("swap them" in w.value for w in at.warning)
    assert not at.dataframe


def test_needs_two_releases_to_compare():
    at = _run(stack_names=["key4hep-2026-07-10"])
    assert any("at least two" in i.value for i in at.info)


def test_missing_provenance_is_reported_not_diffed():
    # Releases benchmarked before provenance capture, or whose stack had aged
    # off CVMFS by backfill time, cannot be compared — say so rather than
    # diffing against nothing.
    at = _run(packages={})
    assert any("No stack provenance" in w.value for w in at.warning)
    assert not at.dataframe


# ── app registration ─────────────────────────────────────────────────────────

def _sections():
    """The section registry.

    Imported from ``sections.py`` rather than ``app.py`` on purpose: app.py
    ends in a bare ``main()``, so importing it would run the whole dashboard —
    and, if ``K4BENCH_DATA_URL`` happens to be set, fetch over the network from
    inside a unit test.
    """
    if str(_DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(_DASHBOARD_DIR))
    spec = importlib.util.spec_from_file_location(
        "k4bench_dashboard_sections", _DASHBOARD_DIR / "sections.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_stack_changes_is_registered_and_remote_only():
    sections = _sections()
    # It compares two releases off EOS, so it cannot work without a data_url.
    assert "Stack Changes" in sections.SECTION_NAMES
    assert "Stack Changes" in sections.REMOTE_ONLY


def test_every_remote_only_section_is_a_real_section():
    sections = _sections()
    # A typo would silently fail to hide a section rather than erroring.
    assert sections.REMOTE_ONLY <= set(sections.SECTION_NAMES)


def test_local_mode_keeps_exactly_the_sections_that_work_without_a_data_url():
    sections = _sections()
    assert sections.visible_sections(trends_enabled=False) == [
        "Region Timing", "Event Timing", "Event Memory", "Machine Info", "Logs",
    ]


def test_remote_mode_keeps_every_section_in_display_order():
    sections = _sections()
    assert sections.visible_sections(trends_enabled=True) == sections.SECTION_NAMES


def test_section_order_is_independent_of_data_requirements(monkeypatch):
    """Reordering the bar must not change which sections are hidden.

    Order is a presentation choice and remote-only is a fact about data
    sources; deriving one from the other would let a reorder strand a tab with
    nothing behind it.
    """
    sections = _sections()
    monkeypatch.setattr(sections, "SECTION_NAMES", list(reversed(sections.SECTION_NAMES)))
    assert set(sections.visible_sections(trends_enabled=False)) == {
        "Region Timing", "Event Timing", "Event Memory", "Machine Info", "Logs",
    }


# ── the scoped reverse view ───────────────────────────────────────────────────

def test_regressions_in_range_scopes_by_detector_and_sample():
    reports = [_raw_report([
        _confirmed(metric="wall_time_s"),
        _confirmed(detector="IDEA", metric="wall_time_s", onset_run_id="run-i"),
        _confirmed(sample="other", metric="peak_rss_mb", onset_run_id="run-s"),
    ])]
    hits = stack_changes._regressions_in_range(
        reports, PLAT, "2026-06-24", "2026-06-25",
        detector="CLD", sample="single_e",
    )
    assert [(v.detector, v.sample) for v in hits] == [("CLD", "single_e")]


def test_reverse_view_scopes_to_the_sidebar_and_widens_on_toggle():
    report = _raw_report([
        _confirmed(metric="wall_time_s", pct_change=0.10,
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
        _confirmed(detector="IDEA", sample="p8_ee_Zbb_ecm91", metric="peak_rss_mb",
                   metric_family="memory", pct_change=0.25, onset_run_id="run-idea",
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
    ])
    at = _run(from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    reverse = at.dataframe[1].value
    assert len(reverse) == 1 and "Detector" not in reverse.columns
    captions = " ".join(c.value for c in at.caption)
    assert "1 more in other detectors" in captions
    # Widen to the whole platform: both rows, with the scope columns back.
    at.toggle(key="stack_regr_all").set_value(True).run()
    assert not at.exception, at.exception
    reverse = at.dataframe[1].value
    assert list(reverse["Detector"]) == ["IDEA", "CLD"]  # worst |Δ| first
    assert "Sample" in reverse.columns


def test_scoped_miss_with_hits_elsewhere_points_at_the_toggle():
    report = _raw_report([
        _confirmed(detector="IDEA", metric="wall_time_s", pct_change=0.10,
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
    ])
    at = _run(from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    body = " ".join(i.value for i in at.info)
    assert "elsewhere on this platform" in body
    assert len(at.dataframe) == 1  # only the package diff table


def test_reverse_table_matches_the_regressions_ledger():
    # The reverse view uses the exact same ledger as the Regressions tab (badge,
    # Dir arrow, |Δ| bar, current/baseline) plus the blame window; no chart of
    # its own — with no CPU+memory pair the outlier plane stays closed too.
    report = _raw_report([
        _confirmed(metric="wall_time_s", pct_change=0.10,
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
        _confirmed(metric="cpu_efficiency", metric_family="cpu_efficiency_pp",
                   pct_change=None, onset_run_id="run-eff",
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
    ])
    at = _run(from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    assert not at.get("plotly_chart")
    reverse = at.dataframe[1].value
    assert list(reverse[""]) == ["🔴", "🔴"]
    assert list(reverse["Dir"]) == ["↑", "↑"]
    assert list(reverse["Metric"]) == ["wall time", "CPU efficiency"]
    assert reverse.iloc[0]["Current / baseline"] == "6 / 5"


# ── typical vs outlier ────────────────────────────────────────────────────────

def test_scatter_candidates_prefer_the_both_families_config():
    hits = [
        _confirmed(label="cfgA", metric="wall_time_s", pct_change=0.30),
        _confirmed(label="cfgB", metric="wall_time_s", pct_change=0.10,
                   onset_run_id="b1"),
        _confirmed(label="cfgB", metric="peak_rss_mb", metric_family="memory",
                   pct_change=0.05, onset_run_id="b1"),
    ]
    cands = stack_changes._scatter_candidates(hits)
    # cfgB stepped in CPU *and* memory at the *same* onset → first despite the
    # smaller |Δ|; its axes are its own flagged metrics. cfgA borrows the
    # default memory axis.
    assert [c[2] for c in cands] == ["cfgB", "cfgA"]
    assert cands[0][3:] == ("wall_time_s", "peak_rss_mb", True)
    assert cands[1][3:] == ("wall_time_s", "peak_rss_mb", False)


def test_scatter_candidates_do_not_combine_different_onsets():
    # cfgC stepped in CPU and memory, but at two different onsets — two
    # unrelated regressions, not one diagonal step. It must not outrank cfgA
    # by virtue of "both", nor carry the "both" flag.
    hits = [
        _confirmed(label="cfgA", metric="wall_time_s", pct_change=0.30),
        _confirmed(label="cfgC", metric="wall_time_s", pct_change=0.10,
                   onset_run_id="b1"),
        _confirmed(label="cfgC", metric="peak_rss_mb", metric_family="memory",
                   pct_change=0.05, onset_run_id="b2"),
    ]
    cands = stack_changes._scatter_candidates(hits)
    assert [c[2] for c in cands] == ["cfgA", "cfgC"]  # ranked by |Δ|, not "both"
    assert cands[1][3:] == ("wall_time_s", "peak_rss_mb", False)


def test_series_points_read_both_metrics_per_night_from_reports():
    reports = [
        _raw_report([_confirmed(metric="wall_time_s", value=5.0),
                     _confirmed(metric="peak_rss_mb", value=100.0)],
                    night="2026-06-24"),
        # A night missing one of the two axes contributes no point.
        _raw_report([_confirmed(metric="wall_time_s", value=9.9)],
                    night="2026-06-25"),
        _raw_report([_confirmed(metric="wall_time_s", value=6.0),
                     _confirmed(metric="peak_rss_mb", value=130.0)],
                    night="2026-06-26"),
    ]
    pts = stack_changes._series_points(
        reports, "CLD", PLAT, "single_e", "baseline", "wall_time_s", "peak_rss_mb")
    assert list(pts["night"]) == ["2026-06-24", "2026-06-26"]
    assert list(pts["x"]) == [5.0, 6.0]
    assert list(pts["y"]) == [100.0, 130.0]
    # The onset run id *is* a night, so it splits before/after directly…
    assert stack_changes._onset_night(pts, [_confirmed(onset_run_id="2026-06-26")]) \
        == "2026-06-26"
    # …and a legacy verdict (no run id) falls back to the first night that
    # measured the onset release.
    legacy = [_confirmed(onset_run_id=None, onset_run_date="2026-06-26")]
    assert stack_changes._onset_night(pts, legacy) == "2026-06-26"


def test_outlier_scatter_opens_for_a_cpu_and_memory_step():
    report = _raw_report([
        _confirmed(metric="wall_time_s", pct_change=0.10,
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10",
                   onset_run_id="2026-06-27"),
        _confirmed(metric="peak_rss_mb", metric_family="memory", pct_change=0.20,
                   onset_run_id="2026-06-27",
                   last_accepted_run_date="2026-07-09", onset_run_date="2026-07-10"),
    ])
    at = _run(from_release="2026-07-09", to_release="2026-07-10",
              report_dates=("2026-07-10",), reports_map={"2026-07-10": report})
    # The CPU × memory plane auto-opens (both families stepped) — the range's
    # only chart.
    assert len(at.get("plotly_chart")) == 1
    sel = at.selectbox(key="stack_outlier_cfg")
    assert "CPU + memory stepped" in sel.value
    # The Overview-style axis pickers, defaulting to the config's flagged
    # metrics.
    assert at.selectbox(key="stack_outlier_tmetric_CLD_single_e_baseline").value \
        == "wall_time_s"
    assert at.selectbox(key="stack_outlier_mmetric_CLD_single_e_baseline").value \
        == "peak_rss_mb"
    # Changing an axis re-renders from the same cached reports.
    at.selectbox(key="stack_outlier_tmetric_CLD_single_e_baseline") \
        .set_value("user_cpu_s").run()
    assert not at.exception, at.exception


def test_scatter_candidates_axes_come_from_the_matched_pair_not_each_worst():
    # cfgD: wall_time_s (onset A, the config's worst |Δ|) and user_cpu_s
    # (onset B, smaller |Δ|) are both "time"; peak_rss_mb (onset B) is the
    # only "memory" flag. Picking each family's own worst independently would
    # plot wall_time_s (onset A) against peak_rss_mb (onset B) while still
    # claiming "both stepped" off the user_cpu_s/peak_rss_mb (onset B) pair —
    # two different pairs. The axes must come from the pair that actually
    # shares an onset: user_cpu_s × peak_rss_mb, not wall_time_s.
    hits = [
        _confirmed(label="cfgD", metric="wall_time_s", pct_change=0.50,
                   onset_run_id="A"),
        _confirmed(label="cfgD", metric="user_cpu_s", pct_change=0.10,
                   onset_run_id="B"),
        _confirmed(label="cfgD", metric="peak_rss_mb", metric_family="memory",
                   pct_change=0.20, onset_run_id="B"),
    ]
    cands = stack_changes._scatter_candidates(hits)
    assert cands[0][2:] == ("cfgD", "user_cpu_s", "peak_rss_mb", True)


def test_bound_to_release_excludes_points_after_head_release():
    # Reports are fetched with no upper date bound (a regression can confirm,
    # and so first appear in a report, after its onset), so the plotted points
    # must be capped separately — a historical range must not attribute a
    # later release's values to the diff being viewed.
    pts = pd.DataFrame({
        "night": ["2026-07-09", "2026-07-10", "2026-07-15"],
        "x": [5.0, 6.0, 999.0],
        "y": [100.0, 120.0, 999.0],
        "k4h_release": ["key4hep-2026-07-09", "key4hep-2026-07-10", "key4hep-2026-07-15"],
    })
    bounded = stack_changes._bound_to_release(pts, "2026-07-10")
    assert list(bounded["night"]) == ["2026-07-09", "2026-07-10"]


def test_deep_link_carries_the_sample_when_given():
    q = _query(stack_changes.deep_link(
        detector="CLD", platform=PLAT, sample="single_e",
        base_release="2026-06-24", head_release="2026-06-25",
    ))
    assert q["sample"] == "single_e"
