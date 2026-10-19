"""Robust estimators shared by the trend builders and the run readers.

A leaf module: it imports nothing of k4bench, so a reader such as
:mod:`k4bench.analysis.heap` can use it without importing the trend builders,
which import the readers.
"""

from __future__ import annotations

import numpy as np

#: Valid post-warm-up samples below which :func:`theil_sen_slope` gives no
#: slope: with fewer points one noisy sample decides a large share of the pairs.
MIN_SLOPE_EVENTS = 5

#: Samples above which :func:`theil_sen_slope` thins its input first. All
#: pairwise slopes of this many points are ~1.1M values; every nightly
#: benchmark stays below it and is estimated exactly.
MAX_SLOPE_EVENTS = 1500


def theil_sen_slope(x, y, *, max_points: int = MAX_SLOPE_EVENTS) -> float | None:
    """Median of the pairwise slopes ``(y_j - y_i) / (x_j - x_i)`` over ``i < j``.

    The Theil–Sen estimator: robust to roughly 29% outlying points, where
    least squares or ``last - first`` follow a single noisy sample. Pairs with
    equal ``x`` carry no slope and are skipped; non-finite points are dropped.

    Above *max_points* points (:data:`MAX_SLOPE_EVENTS` by default), the
    estimate is taken over *max_points* evenly spaced points (first and last
    included) in ``x`` order, each keeping its own ``x``. The cost is then
    bounded for any input size and the result stays deterministic.

    Returns ``None`` below :data:`MIN_SLOPE_EVENTS` finite points or when no
    pair has distinct ``x``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = len(x)
    if n < MIN_SLOPE_EVENTS:
        return None
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if n > max_points:
        keep = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
        x, y = x[keep], y[keep]
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    distinct = dx != 0
    if not distinct.any():
        return None
    return float(np.median((y[j] - y[i])[distinct] / dx[distinct]))
