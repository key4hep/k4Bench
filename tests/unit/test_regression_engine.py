"""Unit tests for the pure regression engine (:mod:`k4bench.regression.engine`)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from k4bench.regression.engine import (
    MIN_BASELINE_RUNS,
    evaluate_series,
    robust_baseline,
)
from k4bench.regression.models import Direction, SeriesId, Severity

_TIME = SeriesId(
    detector="DET", platform="PLAT", sample="single_e",
    label="baseline", metric_family="time", metric="wall_time_s",
)


def _history(values, reliable=None, start="2026-01-01") -> pd.DataFrame:
    """One row per consecutive night; *reliable* defaults to all-True."""
    d0 = date.fromisoformat(start)
    if reliable is None:
        reliable = [True] * len(values)
    dates = [d0 + timedelta(days=i) for i in range(len(values))]
    return pd.DataFrame({
        "run_id":   [d.isoformat() for d in dates],
        "run_date": pd.to_datetime(dates),
        "value":    values,
        "reliable": reliable,
    })


#: A stable baseline with a small deterministic wobble (MAD ≈ 0.4 s on 100 s).
_STEADY = [100.0, 100.4, 99.6, 100.2, 99.8, 100.3, 99.7, 100.1, 99.9, 100.0]


def _severities(verdicts):
    return [v.severity for v in verdicts]


def test_robust_baseline_median_and_scaled_mad():
    med, mad = robust_baseline(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert med == 3.0
    assert mad == pytest.approx(1.4826)  # raw MAD 1.0, scaled


def test_flat_noise_is_ok_throughout():
    verdicts = evaluate_series(_history(_STEADY), series=_TIME)
    assert len(verdicts) == len(_STEADY)
    assert all(v.severity is Severity.UNKNOWN for v in verdicts[:MIN_BASELINE_RUNS])
    assert all(v.severity is Severity.OK for v in verdicts[MIN_BASELINE_RUNS:])
    assert all(v.direction is Direction.NONE for v in verdicts)


def test_outlier_under_effect_floor_is_ok():
    # Perfectly flat baseline → MAD 0 → z is infinite, but +3% is under the 5%
    # time floor: the practical-effect gate must block the flag.
    verdicts = evaluate_series(_history([100.0] * 10 + [103.0]), series=_TIME)
    assert verdicts[-1].severity is Severity.OK


def test_single_night_step_is_watch_then_ok_again():
    verdicts = evaluate_series(_history(_STEADY + [120.0, 100.0, 100.2]), series=_TIME)
    assert verdicts[len(_STEADY)].severity is Severity.WATCH
    assert verdicts[len(_STEADY)].direction is Direction.UP
    # The nights after the spike clear back to OK — no lingering confirm.
    assert _severities(verdicts[len(_STEADY) + 1:]) == [Severity.OK, Severity.OK]
    assert Severity.CONFIRMED not in _severities(verdicts)


def test_persisting_step_is_watch_then_confirmed():
    verdicts = evaluate_series(_history(_STEADY + [120.0, 120.5]), series=_TIME)
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    assert verdicts[-1].direction is Direction.UP
    assert verdicts[-1].pct_change == pytest.approx(0.205, abs=0.01)


def test_step_reverting_after_one_night_never_confirms():
    verdicts = evaluate_series(_history(_STEADY + [120.0, 99.9, 120.0]), series=_TIME)
    # The second spike is a fresh WATCH: the clean night in between reset the
    # pending state, so two non-consecutive spikes never confirm.
    assert _severities(verdicts[-3:]) == [Severity.WATCH, Severity.OK, Severity.WATCH]
    assert Severity.CONFIRMED not in _severities(verdicts)


def test_insufficient_history_is_unknown():
    verdicts = evaluate_series(_history(_STEADY[:MIN_BASELINE_RUNS - 2]), series=_TIME)
    assert all(v.severity is Severity.UNKNOWN for v in verdicts)


def test_downward_step_confirms_same_as_upward():
    # Direction carries no good/bad judgment: a downward step confirms via
    # the exact same two-strike rule as an upward one — no special casing.
    verdicts = evaluate_series(_history(_STEADY + [80.0, 80.2]), series=_TIME)
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    assert verdicts[-1].direction is Direction.DOWN


def test_confirmed_change_reanchors_baseline():
    # A persistent (expected) step: one WATCH, one CONFIRMED — then the
    # baseline re-anchors at the new level instead of re-flagging against the
    # pre-change median for weeks.
    new_level = [120.0, 120.3, 120.1, 119.8, 120.2, 120.4, 119.9, 120.0, 120.3]
    verdicts = evaluate_series(_history(_STEADY + new_level), series=_TIME)
    post = verdicts[len(_STEADY):]
    assert _severities(post[:2]) == [Severity.WATCH, Severity.CONFIRMED]
    # Exactly one CONFIRMED per episode …
    assert _severities(verdicts).count(Severity.CONFIRMED) == 1
    # … and every following night at the new level is OK — judged against the
    # re-anchored (post-change) median, not the pre-change one.
    assert _severities(post[2:]) == [Severity.OK] * 7
    assert all("re-anchoring" in v.reason for v in post[2:7])
    assert post[2].baseline_median == pytest.approx(120.15, abs=0.3)


def test_second_step_after_reanchor_flags_again():
    new_level = [120.0, 120.3, 120.1, 119.8, 120.2, 120.4, 119.9, 120.0, 120.3]
    verdicts = evaluate_series(
        _history(_STEADY + new_level + [140.0, 140.5]), series=_TIME,
    )
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    # The second episode is judged against the re-anchored ~120 s level.
    assert verdicts[-1].baseline_median == pytest.approx(120.1, abs=0.5)


def test_second_step_during_reanchoring_is_still_caught():
    # A further change arriving right after a confirmed one must not fall
    # into a blind window: while the new segment is short the walk judges
    # against its median with the pre-change spread as the noise proxy.
    verdicts = evaluate_series(
        _history(_STEADY + [120.0, 120.5, 121.0, 160.0, 160.5]), series=_TIME,
    )
    assert _severities(verdicts[-5:]) == [
        Severity.WATCH, Severity.CONFIRMED,  # first episode (~120 s)
        Severity.OK,                          # new level holds
        Severity.WATCH, Severity.CONFIRMED,   # second episode (~160 s), no gap
    ]
    # The confirming night is judged against the level it departed FROM (the
    # first episode's ~120 s segment, spike included in its median window).
    assert verdicts[-1].baseline_median == pytest.approx(120.75, abs=1.0)


def test_unreliable_run_neither_confirms_nor_resets_pending():
    values = _STEADY + [120.0, 130.0, 120.5]
    reliable = [True] * len(_STEADY) + [True, False, True]
    verdicts = evaluate_series(_history(values, reliable), series=_TIME)
    # The unreliable night emits no verdict at all …
    assert len(verdicts) == len(values) - 1
    assert "2026-01-12" not in {v.run_id for v in verdicts}
    # … and the WATCH pending across it still confirms on the next reliable night.
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]


def test_unreliable_runs_excluded_from_baseline():
    # A wildly contaminated (unreliable) night must not shift the baseline at
    # all — it is excluded outright, not merely down-weighted.
    values = _STEADY + [500.0, 100.1]
    reliable = [True] * len(_STEADY) + [False, True]
    verdicts = evaluate_series(_history(values, reliable), series=_TIME)
    assert verdicts[-1].severity is Severity.OK
    assert verdicts[-1].baseline_median == pytest.approx(100.0, abs=0.5)


def test_cpu_efficiency_uses_absolute_floor():
    eff = SeriesId(
        detector="DET", platform="PLAT", sample="single_e",
        label="baseline", metric_family="cpu_efficiency_pp", metric="cpu_efficiency",
    )
    base = [0.98] * 10
    # −2 pp is under the 3 pp absolute floor even though z is infinite …
    ok = evaluate_series(_history(base + [0.96]), series=eff)
    assert ok[-1].severity is Severity.OK
    # … while −8 pp trips it.
    watch = evaluate_series(_history(base + [0.90]), series=eff)
    assert watch[-1].severity is Severity.WATCH
    assert watch[-1].direction is Direction.DOWN


def test_tiny_region_wobble_blocked_by_absolute_delta_floor():
    # A 50 µs region jumping +50% is timer noise (Δ ≪ 10 ms), even though both
    # the z-gate and the relative floor trip …
    region = SeriesId(
        detector="DET", platform="PLAT", sample="single_e",
        label="baseline", metric_family="region_time", metric="median_time_s",
        sub_detector="BeamPipe",
    )
    tiny = [5.0e-5] * 10
    ok = evaluate_series(_history(tiny + [7.5e-5]), series=region)
    assert ok[-1].severity is Severity.OK
    # … while the same tiny region genuinely blowing up (Δ ≫ 10 ms) still flags:
    # the floor is on the change, not on the baseline size.
    watch = evaluate_series(_history(tiny + [0.5]), series=region)
    assert watch[-1].severity is Severity.WATCH
    assert watch[-1].direction is Direction.UP


def test_nan_values_are_skipped():
    values = _STEADY + [float("nan"), 100.2]
    verdicts = evaluate_series(_history(values), series=_TIME)
    assert len(verdicts) == len(values) - 1
    assert verdicts[-1].severity is Severity.OK


def _window(verdict) -> tuple:
    return (verdict.last_accepted_run_id, verdict.onset_run_id)


def test_confirmed_verdict_windows_from_last_ok_to_the_watch_night():
    # Confirmation trails onset by one reliable night, so the cause landed in
    # (last OK, WATCH] — never on the confirming night itself.
    verdicts = evaluate_series(_history(_STEADY + [120.0, 120.5]), series=_TIME)
    confirmed = verdicts[-1]
    assert confirmed.severity is Severity.CONFIRMED
    assert confirmed.run_id == "2026-01-12"          # reported here …
    assert _window(confirmed) == ("2026-01-10", "2026-01-11")  # … caused here
    assert verdicts[-2].run_id == "2026-01-11"       # the onset is the WATCH night
    assert verdicts[-2].severity is Severity.WATCH


def test_only_confirmed_verdicts_carry_a_window():
    verdicts = evaluate_series(_history(_STEADY + [120.0, 120.5]), series=_TIME)
    for v in verdicts:
        if v.severity is Severity.CONFIRMED:
            continue
        assert _window(v) == (None, None), f"{v.severity} carried a window"


def test_unreliable_night_inside_the_window_does_not_narrow_it():
    # The skipped night is spanned by the window, not a bound on it: there is
    # no evidence the metric was at the accepted level that night.
    values = _STEADY + [130.0, 120.0, 120.5]
    reliable = [True] * len(_STEADY) + [False, True, True]
    verdicts = evaluate_series(_history(values, reliable), series=_TIME)
    assert "2026-01-11" not in {v.run_id for v in verdicts}  # emitted no verdict
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    assert _window(verdicts[-1]) == ("2026-01-10", "2026-01-12")


def test_window_is_open_ended_when_the_series_never_settled():
    # UNKNOWN is "no evidence", not "at the accepted level" — it must not
    # become a lower bound. Without one the window stays open rather than
    # inventing a night the metric was never observed good on.
    verdicts = evaluate_series(_history([100.0] * 7 + [120.0, 120.5]), series=_TIME)
    assert _severities(verdicts[:7]) == [Severity.UNKNOWN] * 7
    assert _severities(verdicts[-2:]) == [Severity.WATCH, Severity.CONFIRMED]
    assert _window(verdicts[-1]) == (None, "2026-01-08")


def test_second_step_with_no_ok_night_since_a_reanchor_bounds_on_the_reanchor():
    # A re-anchor redefines the accepted level, so the confirmed night is the
    # newest night at it. The second episode is bounded there rather than
    # falling back to an unbounded window.
    verdicts = evaluate_series(
        _history(_STEADY + [120.0, 120.5, 160.0, 160.5]), series=_TIME,
    )
    assert _severities(verdicts[-4:]) == [
        Severity.WATCH, Severity.CONFIRMED,   # first episode (~120 s), re-anchors
        Severity.WATCH, Severity.CONFIRMED,   # second episode (~160 s), no OK between
    ]
    first, second = verdicts[-3], verdicts[-1]
    assert _window(first) == ("2026-01-10", "2026-01-11")
    # Bounded on the first episode's confirmed night, not on the pre-change
    # level it already accounted for, and not on nothing at all.
    assert _window(second) == ("2026-01-12", "2026-01-13")


def test_window_reports_release_dates_alongside_run_ids():
    # run_id is the run directory; run_date is the Key4hep release measured.
    # The nightly build does not publish daily, so several runs can share one
    # release — and a window whose ends share a release proves the stack did
    # not move, whatever the run dates suggest.
    values = _STEADY + [120.0, 120.5]
    run_ids = [f"2026-02-{i + 1:02d}" for i in range(len(values))]
    releases = [f"2026-01-{i + 1:02d}" for i in range(len(values) - 3)] + ["2026-01-20"] * 3
    history = pd.DataFrame({
        "run_id":   run_ids,
        "run_date": pd.to_datetime(releases),
        "value":    values,
        "reliable": [True] * len(values),
    })
    confirmed = evaluate_series(history, series=_TIME)[-1]
    assert confirmed.severity is Severity.CONFIRMED
    assert _window(confirmed) == ("2026-02-10", "2026-02-11")
    assert confirmed.last_accepted_run_date == confirmed.onset_run_date == "2026-01-20"
