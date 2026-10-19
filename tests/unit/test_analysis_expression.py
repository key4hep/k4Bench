"""Per-event expressions (:mod:`k4bench.analysis.expression`): what they
compute, and everything they refuse to."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from k4bench.analysis.expression import (
    MAX_DEPTH,
    MAX_LENGTH,
    MAX_NODES,
    ExpressionError,
    evaluate,
    parse,
)

#: Three events: two particles, none, three.
EVENT_INDEX = np.array([0, 0, 2, 2, 2])
PARTICLES = {
    "pt": np.array([1.0, 3.0, 0.5, np.nan, 2.0]),
    "charge": np.array([1.0, 0.0, -1.0, 1.0, 0.0]),
    "final": np.array([True, True, True, False, True]),
    "charged": np.array([True, False, True, True, False]),
}
EVENTS = pd.DataFrame(
    {"n_hits": [10.0, 0.0, 30.0], "E_calo": [1.5, 0.0, 2.5]},
    index=pd.RangeIndex(3, name="entry"),
)


def _eval(expression: str) -> list[float]:
    return evaluate(expression, EVENTS, PARTICLES, EVENT_INDEX).tolist()


def _nan_equal(values, expected) -> bool:
    return np.array_equal(np.array(values), np.array(expected), equal_nan=True)


@pytest.mark.parametrize(("expression", "expected"), [
    ("count(final)", [2, 0, 2]),
    ("count(final & charged)", [1, 0, 1]),
    ("count(final & ~charged)", [1, 0, 1]),
    ("count(charged | (pt > 2.5))", [2, 0, 2]),
    ("count(0.5 < pt < 2.5)", [1, 0, 1]),
    ("count(final & (charge != 0))", [1, 0, 1]),
    # NaN values are skipped, and an event with none selected sums to 0.
    ("sum(pt)", [4.0, 0.0, 2.5]),
    ("sum(pt, charged)", [1.0, 0.0, 0.5]),
    ("max(pt, final)", [3.0, np.nan, 2.0]),
    ("min(pt)", [1.0, np.nan, 0.5]),
    ("mean(pt)", [2.0, np.nan, 1.25]),
    ("sum(pt**2, final)", [10.0, 0.0, 4.25]),
    ("sqrt(sum(pt**2, final))", [np.sqrt(10.0), 0.0, np.sqrt(4.25)]),
    # A condition counts as 1 or 0.
    ("sum(charged)", [1, 0, 2]),
    ("n_hits + 2 * E_calo", [13.0, 0.0, 35.0]),
    ("n_hits / count(final)", [5.0, np.nan, 15.0]),
    ("-n_hits", [-10.0, -0.0, -30.0]),
    ("abs(-E_calo)", [1.5, 0.0, 2.5]),
    ("log10(n_hits)", [1.0, np.nan, np.log10(30.0)]),
    ("n_hits > 5", [1.0, 0.0, 1.0]),
    ("2", [2.0, 2.0, 2.0]),
    ("1e400", [np.nan] * 3),
])
def test_an_expression_is_one_number_per_event(expression, expected):
    assert _nan_equal(_eval(expression), expected)


def test_the_result_is_indexed_as_the_events_and_named_by_the_expression():
    result = evaluate(" count(final) ", EVENTS, PARTICLES, EVENT_INDEX)
    assert result.index.equals(EVENTS.index)
    assert result.name == "count(final)"


def test_parentheses_cost_no_depth():
    assert _eval("(" * (MAX_DEPTH + 5) + "n_hits" + ")" * (MAX_DEPTH + 5)) == [10.0, 0.0, 30.0]


def test_columns_alone_need_no_particles():
    assert evaluate("n_hits * 2", EVENTS).tolist() == [20.0, 0.0, 60.0]


def test_a_reduction_without_particles_says_so():
    with pytest.raises(ExpressionError, match="recorded none"):
        evaluate("count(final)", EVENTS)


@pytest.mark.parametrize(("expression", "match"), [
    ("", "empty"),
    ("count(", "not an expression"),
    ("x" * (MAX_LENGTH + 1), f"longer than {MAX_LENGTH}"),
    ("+".join(["1"] * 60), f"more than {MAX_NODES}"),
    ("-" * (MAX_DEPTH + 2) + "n_hits", f"deeper than {MAX_DEPTH}"),
    ("pt", "particle field: fold it"),
    ("count(n_hits > 1)", "one number per event"),
    ("count(final & count(charged))", "cannot be inside another"),
    ("count(pt)", "count needs a condition"),
    ("sum(pt, pt)", "sum's mask needs a condition"),
    ("n_hits & E_calo", "& needs a condition"),
    ("final and charged", "use & and |"),
    ("count(not final)", "use ~ rather than not"),
    ("n_hits ** E_calo", "exponent must be a number"),
    ("n_hits ** 11", "at most 10"),
    ("9 ** 9 ** 9", "exponent must be a number"),
    ("n_hits % 3", "operator % is not allowed"),
    ("n_hits in E_calo", "comparison in is not allowed"),
    ("nhits", "unknown name nhits; did you mean n_hits"),
    ("count(fnal)", "unknown particle field fnal; did you mean final"),
    ("median(pt)", "unknown function median"),
    ("count()", "use count\\(mask\\)"),
    ("sum(pt, final, charged)", "sum\\(x\\) or sum\\(x, mask\\)"),
    ("abs(pt, pt)", "takes one argument"),
    ("sum(pt, mask=final)", "no named arguments"),
    ("True", "is not a number"),
    ("'text'", "is not a number"),
    ("__import__('os').system('true')", "can be called"),
    ("n_hits.real", "attribute access"),
    ("EVENTS['n_hits']", "indexing"),
    ("(lambda: 1)()", "can be called"),
    ("lambda: 1", "lambda"),
    ("[n_hits]", "lists"),
    ("n_hits if E_calo else 1", "conditional"),
    ("(x := 1)", "assignment"),
    ("f'{n_hits}'", "strings"),
    ("sum(*pt)", "starred"),
])
def test_what_cannot_be_computed_is_refused_in_words(expression, match):
    with pytest.raises(ExpressionError, match=match):
        _eval(expression)


def test_no_expression_is_dear_to_compute():
    started = time.perf_counter()
    for expression in ("9**9**9**9", "10**10**10", "n_hits**10**10", "-" * 400):
        with pytest.raises(ExpressionError):
            _eval(expression)
    assert time.perf_counter() - started < 1.0


def test_parse_checks_without_computing():
    assert parse("count(final)").body is not None
    with pytest.raises(ExpressionError):
        parse("a.b.c.d" * 100)


def test_particles_must_match_the_event_index():
    with pytest.raises(ValueError, match="one value per particle"):
        evaluate("count(final)", EVENTS, {"final": np.array([True])}, EVENT_INDEX)
