"""The Correlations plot: each event's cost against what it holds, with the
robust trend line."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from k4bench.analysis.event_drivers import Trend  # noqa: E402
from k4bench.analysis.plots._theme import _BLUE, _RED  # noqa: E402
from k4bench.analysis.plots.event_content import plot_event_content, plot_event_costs  # noqa: E402


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


def test_only_the_points_carry_their_event_whatever_their_order():
    # Shuffled, with events missing: a point's place is not its event.
    pairs = pd.DataFrame({"x": [4.0, 1.0, 3.0, 2.0], "y": [8.0, 1.0, 9.0, 4.0]},
                         index=pd.Index([9, 3, 7, 4], name="event"))
    fig = plot_event_content(pairs, x_label="x", y_label="y", line=Trend(2.0, 0.0),
                             above=pairs.loc[[7]], selected=4)
    assert [t.name for t in fig.data] == ["Events", "Trend", "Above the trend", "Selected"]
    assert [int(row[0]) for row in fig.data[0].customdata] == [9, 3, 7, 4]
    assert all(t.customdata is None for t in fig.data[1:])
    assert (fig.data[3].x, fig.data[3].y) == ((2.0,), (4.0,))


def test_an_event_not_plotted_is_not_marked():
    pairs = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]}, index=pd.Index([1, 2]))
    fig = plot_event_content(pairs, x_label="x", y_label="y", selected=5)
    assert [t.name for t in fig.data] == ["Events"]


def test_an_events_costs_are_drawn_largest_first_the_rest_as_other():
    costs = pd.Series({"a": 1.0, "b": 5.0, "c": float("nan"), "d": 3.0, "e": 0.5})
    typical = pd.Series({"a": 1.0, "b": 2.0, "c": 4.0, "d": 3.0, "e": 0.5})
    fig = plot_event_costs(costs, typical, title="Time (ms)", highlight="d", top_n=3)
    bars, other, median = fig.data
    # The algorithm without a cost in the event is no bar, not a zero one.
    assert list(bars.y) == ["b", "d", "a"] and list(bars.x) == [5.0, 3.0, 1.0]
    assert [row[0] for row in bars.customdata] == ["b", "d", "a"]
    assert list(bars.marker.color) == [_BLUE, _RED, _BLUE]
    assert other.y == ("Other (1)",) and other.x == (0.5,) and other.customdata is None
    assert list(median.y) == ["b", "d", "a"] and list(median.x) == [2.0, 3.0, 1.0]
    assert median.customdata is None
    assert list(fig.layout.yaxis.categoryarray) == ["b", "d", "a", "Other (1)"]


def test_the_highlighted_algorithm_is_named_however_small_its_cost():
    costs = pd.Series({f"alg{i}": float(20 - i) for i in range(20)})
    fig = plot_event_costs(costs, costs, title="t", highlight="alg19")
    bars, other, median = fig.data
    assert list(bars.y) == [f"alg{i}" for i in range(14)] + ["alg19"]
    assert list(bars.marker.color) == [_BLUE] * 14 + [_RED]
    assert bars.customdata[-1][0] == "alg19" and median.y[-1] == "alg19"
    # alg14 to alg18.
    assert other.y == ("Other (5)",) and other.x == (sum(range(2, 7)),)


def test_peaks_left_out_are_their_largest_not_their_sum():
    costs = pd.Series({"a": 9.0, "b": 4.0, "c": 3.0, "d": 2.0})
    fig = plot_event_costs(costs, costs, title="t", top_n=1, additive=False)
    other = fig.data[1]
    assert other.y == ("Other (3), largest",) and other.x == (4.0,)


def test_an_event_no_algorithm_has_a_cost_in_draws_nothing():
    costs = pd.Series({"a": float("nan")})
    assert plot_event_costs(costs, pd.Series({"a": 1.0}), title="t") is None
