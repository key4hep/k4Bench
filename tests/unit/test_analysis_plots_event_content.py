"""The Correlations plot: each event's cost against what it holds, with the
robust trend line."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from k4bench.analysis.event_drivers import Trend  # noqa: E402
from k4bench.analysis.plots.event_content import plot_event_content  # noqa: E402


def _line(fig):
    return next(t for t in fig.data if t.name == "Trend")


@pytest.mark.parametrize("log_y", [False, True])
def test_the_trend_line_is_the_fitted_line_between_the_ends(log_y):
    pairs = pd.DataFrame({"x": [1.0, 100.0], "y": [1.0, 100.0]})
    line = _line(plot_event_content(pairs, x_label="x", y_label="y",
                                    line=Trend(1.0, 0.0), log_y=log_y))
    x, y = np.asarray(line.x), np.asarray(line.y)
    # On a log axis two end points would draw y = 10 at x = 50.5.
    assert len(x) > 2
    assert y == pytest.approx(x)


def test_on_a_log_axis_the_line_stops_where_it_is_not_positive():
    pairs = pd.DataFrame({"x": [0.0, 10.0], "y": [1.0, 2.0]})
    line = _line(plot_event_content(pairs, x_label="x", y_label="y",
                                    line=Trend(1.0, -5.0), log_y=True))
    x, y = np.asarray(line.x), np.asarray(line.y, dtype=float)
    assert np.isnan(y[x <= 5.0]).all()
    assert (y[x > 5.0] > 0).all()
