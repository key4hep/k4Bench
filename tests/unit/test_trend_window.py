"""Unit tests for ``dashboard/trend_window.py``: the release range the window
opens on, the report nights that judged a range, and the run listing cut to
it.

``trend_window`` is a pure, Streamlit-free module, so it is loaded in isolation
by file path (its siblings ``app``/``data`` pull in Streamlit).
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pytest

_TW_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "trend_window.py"


def _load_tw():
    spec = importlib.util.spec_from_file_location("k4bench_dashboard_trend_window", _TW_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tw = _load_tw()


def _rel(day: str) -> str:
    return f"key4hep-2026-{day}"


#: Twelve releases, oldest first, with the gaps of nights that re-ran one.
_RELEASES = [_rel(d) for d in (
    "09-19", "09-21", "09-22", "09-23", "09-24", "09-26",
    "09-28", "09-29", "09-30", "10-01", "10-03", "10-05",
)]


# ── dated_releases ───────────────────────────────────────────────────────────

def test_dated_releases_sorts_oldest_first_dedupes_and_drops_undated_names():
    stacks = [_rel("10-05"), "key4hep-latest", _rel("09-19"), _rel("10-05")]
    assert tw.dated_releases(stacks) == [_rel("09-19"), _rel("10-05")]
    assert tw.dated_releases([]) == []


def test_release_day_reads_the_tag_date():
    assert tw.release_day(_rel("10-05")) == date(2026, 10, 5)
    assert tw.release_day("key4hep-latest") is None


# ── default_releases ─────────────────────────────────────────────────────────

def test_the_window_opens_on_the_newest_ten_releases():
    assert tw.default_releases(_RELEASES, None) == _RELEASES[-10:]
    assert tw.default_releases(_RELEASES, None, count=5) == _RELEASES[-5:]


def test_a_history_shorter_than_the_count_opens_on_all_of_it():
    assert tw.default_releases(_RELEASES, None, count=50) == _RELEASES
    one = [_rel("10-05")]
    assert tw.default_releases(one, None) == one


def test_a_release_ahead_of_the_newest_report_is_added_on_top():
    """A release tonight's batch is still uploading has no report yet. Counting
    it would push the oldest release out of the window until the report lands,
    taking the Overview's oldest night with it; it is added on top instead."""
    uploading = [*_RELEASES, _rel("10-06")]
    newest_report = date(2026, 10, 5)
    assert tw.default_releases(uploading, newest_report, count=5) == [
        *_RELEASES[-5:], _rel("10-06"),
    ]


def test_an_anchor_on_a_rerun_night_counts_the_release_it_reran():
    # The 10-04 report re-ran the 10-03 release: 10-05 has no report yet.
    assert tw.default_releases(
        _RELEASES, date(2026, 10, 4), count=5,
    ) == _RELEASES[-6:]


def test_an_anchor_past_every_release_counts_from_the_newest():
    assert tw.default_releases(_RELEASES, date(2027, 1, 1), count=5) == _RELEASES[-5:]


# ── releases_between ─────────────────────────────────────────────────────────

def test_releases_between_takes_both_ends_and_everything_between():
    assert tw.releases_between(_RELEASES, _rel("09-22"), _rel("09-28")) == [
        _rel("09-22"), _rel("09-23"), _rel("09-24"), _rel("09-26"), _rel("09-28"),
    ]
    assert tw.releases_between(_RELEASES, _rel("10-01"), _rel("10-01")) == [_rel("10-01")]


# ── report_span ──────────────────────────────────────────────────────────────

def test_report_span_of_the_newest_releases_reaches_the_last_night():
    # 10-05's release is still re-run on 10-06 and 10-07 while no newer tag exists.
    selected = _RELEASES[-3:]
    assert tw.report_span(selected, _RELEASES, date(2026, 10, 7)) == (
        date(2026, 10, 1), date(2026, 10, 7),
    )


def test_report_span_ends_the_night_before_the_next_newer_release():
    # The 09-24 release was re-run on 09-25; 09-26 judged the next release.
    selected = [_rel("09-22"), _rel("09-23"), _rel("09-24")]
    assert tw.report_span(selected, _RELEASES, date(2026, 10, 7)) == (
        date(2026, 9, 22), date(2026, 9, 25),
    )


def test_report_span_of_one_release():
    assert tw.report_span([_rel("09-21")], _RELEASES, date(2026, 10, 7)) == (
        date(2026, 9, 21), date(2026, 9, 21),
    )


# ── to_dates / runs_in_window ────────────────────────────────────────────────

def test_to_dates_sorts_dedupes_and_drops_unparseable_names():
    assert tw.to_dates(["2026-05-21", "latest", "2026-01-01", "2026-05-21"]) == [
        date(2026, 1, 1), date(2026, 5, 21),
    ]
    assert tw.to_dates([]) == []


def test_runs_in_window_keeps_every_run_of_the_selected_releases():
    stacks_dates = {
        "key4hep-2026-05-20": ["2026-05-21", "2026-05-20", "not-a-date"],
        "key4hep-2026-03-01": ["2026-03-01", "2026-05-22"],
        "key4hep-2026-01-01": ["2026-01-01"],
        "key4hep-2026-02-01": ["not-a-date"],
    }
    windowed = tw.runs_in_window(
        stacks_dates,
        ["key4hep-2026-02-01", "key4hep-2026-03-01", "key4hep-2026-05-20"],
    )
    # A release re-run long after its tag keeps the late run; a release outside
    # the window is dropped whatever night it ran, and one left with no dated
    # run is dropped rather than left empty.
    assert windowed == (
        ("key4hep-2026-03-01", ("2026-03-01", "2026-05-22")),
        ("key4hep-2026-05-20", ("2026-05-20", "2026-05-21")),
    )
    # The download cache keys on it, so it must hash.
    hash(windowed)


@pytest.mark.parametrize("releases", [[], ["key4hep-2026-01-01"]])
def test_runs_in_window_of_an_empty_listing_is_empty(releases):
    assert tw.runs_in_window({}, releases) == ()
