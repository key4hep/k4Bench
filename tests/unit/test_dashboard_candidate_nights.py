"""Which report nights the Regressions tab offers for the sidebar's release
(:func:`dashboard.tabs.regressions._candidate_nights`)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from tabs import regressions  # noqa: E402

from k4bench.regression.lineage import PLATFORM_SUCCESSORS  # noqa: E402

_ACTIVE, _REPLACED = next(iter(PLATFORM_SUCCESSORS.items()))

_RUNS = ["2026-09-01", "2026-09-02", "2026-09-03"]
_NEWEST_STACK = f"key4hep-{_RUNS[-1]}"
_STACKS_DATES = {f"key4hep-{night}": [night] for night in _RUNS}
#: The replacing platform's first run.
_SUCCESSOR_RUN = "2026-09-06"
#: Report nights: one the replaced platform missed before its successor ran,
#: and one after.
_BEFORE = [*_RUNS, "2026-09-04"]
_REPORTS = [*_BEFORE, "2026-09-10"]


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def listing(url, det, plat, samp):
        if plat == _ACTIVE:
            return {f"key4hep-{_SUCCESSOR_RUN}": [_SUCCESSOR_RUN]}
        return dict(_STACKS_DATES)
    monkeypatch.setattr(regressions, "_cached_list_run_dates", listing)


def _nights(platform, dates, stack=_NEWEST_STACK):
    nights, is_release, _ = regressions._candidate_nights(
        "https://example.invalid", "CLD", platform, "single_e", stack, list(dates),
    )
    assert is_release
    return nights


def test_a_replaced_platforms_newest_release_offers_only_its_own_nights():
    # Its newest release *is* its newest, but the platform is not current: the
    # later reports carry no run group for it, so offering the latest would
    # append a night with nothing to say to the end of its history.
    assert _nights(_REPLACED, _REPORTS) == [_RUNS[-1]]


def test_a_replaced_platform_keeps_the_night_it_missed_before_its_successor_ran():
    # The report still flagged that night as a missing run, so it stays
    # reachable, exactly as the report assembly decided.
    assert _nights(_REPLACED, _BEFORE) == [max(_BEFORE), _RUNS[-1]]


def test_an_active_platform_still_offers_the_latest_report():
    # The control — a platform that simply has not run tonight keeps the latest
    # report on offer, because that is where its "no run uploaded" failure is.
    nights = _nights(_ACTIVE, _REPORTS, stack=f"key4hep-{_SUCCESSOR_RUN}")
    assert nights == [max(_REPORTS)]


def test_an_older_release_never_offered_the_latest_report():
    # Only a platform's newest release ever gets the latest report appended.
    assert _nights(_REPLACED, _BEFORE, stack=f"key4hep-{_RUNS[0]}") == [_RUNS[0]]


def test_a_migration_cards_package_changes_read_each_end_on_its_own_platform(monkeypatch):
    from k4bench.regression.models import Direction, MetricVerdict, Severity

    verdict = MetricVerdict(
        detector="CLD", platform=_ACTIVE, sample="single_e", label="baseline",
        metric_family="memory", metric="peak_rss_mb", sub_detector=None,
        run_id="2026-09-07", run_date="2026-09-07", value=1787.0,
        baseline_median=1669.0, baseline_mad=2.0, pct_change=0.07, z_score=60.0,
        severity=Severity.CONFIRMED, direction=Direction.UP, reason="step",
        onset_run_id="2026-09-06", onset_run_date="2026-09-04",
        last_accepted_run_id="2026-09-03", last_accepted_run_date="2026-09-03",
        last_accepted_platform=_REPLACED,
    )
    provenance = {
        (_REPLACED, "2026-09-03"): {"dd4hep": {"commit": "c880d07" + "0" * 33}},
        (_ACTIVE, "2026-09-04"): {"DD4hep": {"commit": "b30b104"}},
    }
    monkeypatch.setattr(
        regressions, "packages_for_release",
        lambda url, platform, release: provenance.get((platform, release)),
    )
    changes = regressions._window_changes("https://example.invalid", verdict)
    assert [(c.name, c.status) for c in changes] == [("DD4hep", "changed")]
