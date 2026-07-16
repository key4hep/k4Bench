"""Statistical step-change detector for nightly benchmark metrics.

Detection is deliberately conservative — the goal is a report developers trust,
so every gate errs toward *not* flagging:

1. **Baseline**: the trailing :data:`BASELINE_WINDOW_RUNS` *reliable* runs
   strictly before the night under test. Runs failing the conservative host
   reliability check (:mod:`k4bench.results.reliability`) never enter the
   baseline and are never themselves evaluated — contention is not a
   regression. Below :data:`MIN_BASELINE_RUNS` reliable points the verdict is
   ``UNKNOWN``, never a flag (no evidence ⇒ no verdict).
2. **Robust statistics**: the baseline center/spread are the median and the
   normal-consistent MAD (:data:`MAD_NORMAL_CONSISTENCY` × MAD), not
   mean/stddev, so one contaminated night that slipped past the reliability
   filter cannot distort the threshold.
3. **Robust z-gate**: flag only beyond :data:`Z_THRESHOLD` (the
   Iglewicz–Hoaglin robust-outlier threshold).
4. **Practical-effect floor** (:data:`EFFECT_FLOOR`), ANDed with the z-gate: a
   metric that is normally rock-steady has a tiny MAD, so the z-gate alone
   would trip on practically irrelevant wobbles.
5. **Two-strike confirmation**: the first night crossing both gates is
   ``WATCH``; only when the *next* reliable night repeats it in the same
   direction does it become ``CONFIRMED``. This is the single
   highest-leverage lever against false positives (the pattern used by
   Chromium's and Firefox's perf-CI bots) — more effective than tightening
   the z-gate, which just trades false positives for missed regressions.
6. **Change-point re-anchoring**: a ``CONFIRMED`` change is treated as the
   new accepted level — the baseline window is re-anchored on the post-change
   values, so an *expected* regression (say, a deliberate physics change)
   alerts exactly once instead of being re-judged against the pre-change
   median for weeks. While the new segment is short, judging continues
   against its median with the pre-change spread as the noise proxy, so a
   second change arriving right away is still caught. The state is recomputed
   from the history on every walk, so there is no state file to manage.
7. **No good/bad judgment**: a confirmed change is reported as ``UP`` or
   ``DOWN`` — a plain sign, not an evaluation. Faster is not "improved" any
   more than slower is "regressed" in the colloquial sense: either can be a
   deliberate change, an optimization, or a bug, and the report leaves that
   call to a human instead of asserting one.

Known v1 limitations (deliberate):

- Slow drift (a creeping regression too small to trip step detection) is out
  of scope — "Phase 5 (not built)". Gathering a track record on the
  step detector comes first; Mann-Kendall/EWMA can be layered on later.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

from k4bench.regression.models import Direction, MetricVerdict, SeriesId, Severity

#: Trailing window of reliable runs forming the baseline. Two weeks of
#: nightlies: long enough for a stable median/MAD, short enough that a
#: confirmed step ages into the accepted baseline within ~a week.
BASELINE_WINDOW_RUNS = 14

#: Minimum reliable baseline points before any verdict is issued. Below this
#: the MAD is too unstable to trust and the verdict stays ``UNKNOWN``.
MIN_BASELINE_RUNS = 7

#: Robust z-score gate. 3.5 is the Iglewicz–Hoaglin recommended threshold for
#: MAD-based outlier detection on modest sample sizes.
Z_THRESHOLD = 3.5

#: Scale factor making the MAD a consistent estimator of the standard
#: deviation under normality (1/Φ⁻¹(3/4)).
MAD_NORMAL_CONSISTENCY = 1.4826

#: Minimum practical effect per metric family, ANDed with the z-gate. Relative
#: fractions of the baseline median, except ``cpu_efficiency_pp`` which is an
#: absolute difference in efficiency (percentage points of a 0–1 ratio, where
#: a relative change would be unintuitive). Region metrics get a wider floor —
#: smaller absolute numbers, noisier.
EFFECT_FLOOR: dict[str, float] = {
    "time":              0.05,
    "memory":            0.05,
    "cpu_efficiency_pp": 0.03,
    "region_time":       0.08,
}

#: Families whose :data:`EFFECT_FLOOR` is an absolute difference, not a
#: fraction of the baseline median.
_ABSOLUTE_FLOOR_FAMILIES = frozenset({"cpu_efficiency_pp"})

#: Additional *absolute* change floor per family, ANDed with the relative
#: floor. Motivated by the retrospective run over the real EOS history
#: (2026-05-23 → 2026-07-10): sub-detector regions with median times of tens
#: of microseconds routinely wobble ±30–50 % from timer granularity alone,
#: and dominated the false flags. Requiring the *change itself* to exceed
#: 10 ms/event keeps those quiet while still catching a tiny region that
#: genuinely blows up (unlike a gate on the baseline median, which would
#: blind the detector to exactly that case). The time/memory entries are
#: prophylactic no-ops at current scales (run/event times ≫ 10 ms, RSS
#: baselines ≈ 6 GB), guarding hypothetical micro-configs.
ABS_DELTA_FLOOR: dict[str, float] = {
    "region_time": 0.010,   # seconds/event
    "time":        0.010,   # seconds
    "memory":      16.0,    # MB
}


def robust_baseline(values: np.ndarray) -> tuple[float, float]:
    """Return ``(median, scaled_mad)`` of *values*.

    The MAD is scaled by :data:`MAD_NORMAL_CONSISTENCY` so the z-scores built
    from it are comparable to classical standard scores under normality.
    """
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, MAD_NORMAL_CONSISTENCY * mad


def robust_change(x: float, med: float, mad: float) -> tuple[float | None, float]:
    """Return ``(pct_change, z_score)`` of *x* against baseline ``(med, mad)``.

    Shared by the confirmation walk below and by
    :mod:`k4bench.regression.report_builder`'s one-shot region-contributor
    lookup, so both use exactly the same robust-statistics math.
    """
    delta = x - med
    pct_change = delta / med if med != 0 else None
    if mad > 0:
        z = delta / mad
    else:
        # A perfectly flat baseline: any deviation is infinitely surprising
        # statistically; the practical-effect floor alone decides.
        z = 0.0 if delta == 0 else math.copysign(math.inf, delta)
    return pct_change, z


def _fmt_date(value) -> str:
    """Render a run date as ``YYYY-MM-DD`` (empty string when unknown)."""
    ts = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def _identity(row) -> tuple[str, str]:
    """``(run_id, run_date)`` of *row* — the run directory and the release it
    measured. Kept as one value so the pair can never drift apart."""
    return str(row.run_id), _fmt_date(row.run_date)


def evaluate_series(
    history: pd.DataFrame,
    *,
    series: SeriesId,
) -> list[MetricVerdict]:
    """Chronologically evaluate one metric history and return its verdict series.

    *history* holds the full ordered history of one series (one
    :class:`SeriesId`), with columns:

    - ``run_id`` — run identifier (the nightly date directory name),
    - ``run_date`` — datetime-like, defines the evaluation order,
    - ``value`` — the metric value (NaN/None rows are skipped),
    - ``reliable`` — the per-run reliability tri-state
      (``True``/``False``/``None``). Only ``False`` excludes a run: an
      *unknown* verdict (no machine info) is not evidence of contention, and
      excluding it would starve baselines on histories without machine info —
      the same policy as the dashboard's reliability filter.

    Each reliable run is judged against the trailing baseline window preceding
    it, carrying one bit of pending-WATCH state forward for the two-strike
    rule. Unreliable runs are skipped entirely — they neither confirm nor
    reset a pending WATCH, since there is no evidence either way for that
    night. Returns the full verdict series (the dashboard drill-down shades
    from it); callers wanting "tonight's" verdict take the last element.

    A ``CONFIRMED`` night is a **change-point**: the baseline window is
    re-anchored there (cleared and re-seeded with the two post-change values),
    so an expected/accepted change alerts exactly once instead of being
    re-judged against the pre-change median for weeks. While the new segment
    is still short (< :data:`MIN_BASELINE_RUNS` points) the walk keeps
    judging — against the growing segment's median, with the *pre-change*
    scaled MAD as the spread proxy (the noise level changes far less than the
    level itself) — so a second change arriving right after a confirmed one
    is still caught rather than falling into a blind window.

    Each ``CONFIRMED`` verdict also carries the window the change entered in:
    ``onset_*`` (the WATCH night — where it first appeared, one reliable night
    before it was confirmed) and ``last_accepted_*`` (the newest night before
    that observed at the then-accepted level). Because confirmation trails
    onset, the confirmed night is the wrong place to look for a cause; the
    change landed in ``(last_accepted, onset]``. An unreliable night inside
    the window is skipped, not judged, so it never narrows the window — it is
    spanned by it.
    """
    floor = EFFECT_FLOOR[series.metric_family]
    abs_delta_floor = ABS_DELTA_FLOOR.get(series.metric_family, 0.0)
    absolute_floor = series.metric_family in _ABSOLUTE_FLOOR_FAMILIES

    df = history.sort_values(["run_date", "run_id"], kind="stable")
    baseline: deque[float] = deque(maxlen=BASELINE_WINDOW_RUNS)
    pending: Direction | None = None
    pending_value: float | None = None  # the WATCH night's value (seeds the re-anchor)
    pending_run: tuple[str, str] | None = None   # the WATCH night's identity (the onset)
    last_accepted: tuple[str, str] | None = None  # newest night seen at the accepted level
    anchor_date: str | None = None      # date of the last confirmed change-point
    anchor_mad: float = 0.0             # pre-change spread, proxy while re-anchoring
    verdicts: list[MetricVerdict] = []

    def _verdict(row, **kw) -> MetricVerdict:
        run_id, run_date = _identity(row)
        return MetricVerdict(
            detector=series.detector,
            platform=series.platform,
            sample=series.sample,
            label=series.label,
            metric_family=series.metric_family,
            metric=series.metric,
            sub_detector=series.sub_detector,
            run_id=run_id,
            run_date=run_date,
            **kw,
        )

    for row in df.itertuples(index=False):
        if row.reliable is False:
            continue  # no evidence for this night: skip, don't touch `pending`
        x = row.value
        if x is None or (isinstance(x, float) and math.isnan(x)):
            continue
        x = float(x)

        reanchoring = anchor_date is not None and len(baseline) < MIN_BASELINE_RUNS
        if len(baseline) < MIN_BASELINE_RUNS and not reanchoring:
            verdicts.append(_verdict(
                row,
                value=x, baseline_median=None, baseline_mad=None,
                pct_change=None, z_score=None,
                severity=Severity.UNKNOWN, direction=Direction.NONE,
                reason=f"only {len(baseline)} reliable baseline runs "
                       f"(<{MIN_BASELINE_RUNS}) — not judged",
            ))
            baseline.append(x)
            continue

        if reanchoring:
            # Short post-change segment: its median is already the best center,
            # but its MAD is too unstable to trust — inherit the pre-change
            # spread instead, so a second change right after a confirmed one
            # is still detectable (no blind window).
            med = float(np.median(np.asarray(baseline)))
            mad = anchor_mad
        else:
            med, mad = robust_baseline(np.asarray(baseline))
        delta = x - med
        pct_change, z = robust_change(x, med, mad)

        effect = abs(delta) if absolute_floor else (abs(pct_change) if pct_change is not None else 0.0)
        tripped = abs(z) > Z_THRESHOLD and effect > floor and abs(delta) >= abs_delta_floor

        confirmed_now = False
        if not tripped:
            severity, direction = Severity.OK, Direction.NONE
            pending = pending_value = pending_run = None  # a clean night clears an unconfirmed WATCH
            last_accepted = _identity(row)
            reason = "within baseline variation"
            if reanchoring:
                reason += (f" (re-anchoring after confirmed change on {anchor_date}, "
                           f"{len(baseline)}/{MIN_BASELINE_RUNS} runs at the new level)")
        else:
            direction = Direction.UP if delta > 0 else Direction.DOWN
            if pending is direction:
                severity = Severity.CONFIRMED
                confirmed_now = True
            else:
                severity = Severity.WATCH
                pending, pending_value, pending_run = direction, x, _identity(row)
            change = (
                f"{delta:+.3f} (abs)" if absolute_floor or pct_change is None
                else f"{pct_change:+.1%}"
            )
            z_txt = "inf" if math.isinf(z) else f"{z:.1f}"
            reason = f"{change} vs baseline median {med:.4g} (robust z={z_txt})"

        window: dict[str, str | None] = {}
        if confirmed_now:
            # The change appeared on the WATCH night, one reliable night before
            # this one, and was last absent on `last_accepted` — so it entered
            # in `(last_accepted, onset]`. Stamp the pair the searchable window
            # is built from; `last_accepted` stays None if the series never
            # settled, leaving the window open-ended rather than falsely tight.
            onset = pending_run if pending_run is not None else _identity(row)
            window = {
                "onset_run_id": onset[0],
                "onset_run_date": onset[1],
                "last_accepted_run_id": last_accepted[0] if last_accepted else None,
                "last_accepted_run_date": last_accepted[1] if last_accepted else None,
            }

        verdicts.append(_verdict(
            row,
            value=x, baseline_median=med, baseline_mad=mad,
            pct_change=pct_change, z_score=z,
            severity=severity, direction=direction, reason=reason,
            **window,
        ))
        if confirmed_now:
            # Change-point: the confirmed level is the new normal. Re-anchor
            # the window on the two post-change values so the pre-change
            # median stops being the yardstick from tomorrow on; keep the
            # pre-change spread as the interim noise estimate.
            watch_value = pending_value if pending_value is not None else x
            baseline.clear()
            baseline.append(watch_value)
            baseline.append(x)
            anchor_date = _fmt_date(row.run_date) or str(row.run_id)
            anchor_mad = mad
            pending = pending_value = pending_run = None
            # Re-anchoring redefines the accepted level as the post-change one,
            # and this night is the newest sitting at it. Carrying the
            # pre-change night forward would blame an already-accepted change;
            # clearing it would leave a second step that confirms before any
            # OK night — the case the re-anchor exists to keep catching — with
            # no lower bound at all.
            last_accepted = _identity(row)
        else:
            baseline.append(x)

    return verdicts
