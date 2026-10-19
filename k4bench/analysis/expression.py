"""Per-event quantities written as expressions.

An expression is one number per event, computed from the event's generator
particles (:meth:`k4bench.analysis.event_source.EventSource.particles`) and
its per-event columns (what each collection holds, the canned quantities of
:mod:`k4bench.analysis.event_drivers`)::

    count(final & charged & (pt > 1))
    sum(E, final & (pdg == 22)) / visible_E
    n_VertexBarrelCollection + n_InnerTrackerBarrelCollection

Outside a reduction, a name is a per-event column. Inside one, a name is a
particle field, and the reduction folds the particles of each event into one
number:

- ``count(mask)``: the particles the mask selects;
- ``sum(x)``, ``mean(x)``, ``max(x)``, ``min(x)``, or with a mask as their
  second argument, ``sum(x, mask)``: over the particles selected, NaN values
  skipped. An event with none selected has a ``count`` and ``sum`` of 0 and a
  ``mean``, ``max`` and ``min`` of NaN.

Also allowed: numbers, ``+ - * / **`` (a power only to a constant of
magnitude at most :data:`MAX_EXPONENT`), comparisons, which may chain
(``0.5 < vr < 10``), ``&``, ``|`` and ``~`` between conditions, unary minus,
and ``abs``, ``sqrt`` and ``log10``. ``and``, ``or`` and ``not`` are not: a
condition is an array, so it combines with ``&``, ``|`` and ``~``, the
comparisons in parentheses. A condition used as a number is 1 or 0.

:func:`evaluate` never hands the text to Python's ``eval``. It parses it
(:mod:`ast`) and computes only the node types above, in floating point under
``np.errstate(all="ignore")``: an expression is at most :data:`MAX_LENGTH`
characters, :data:`MAX_NODES` nodes and :data:`MAX_DEPTH` deep, so none is
dear to compute, and an invalid operation gives NaN rather than an error. A
result that is not finite is NaN.
"""

from __future__ import annotations

import ast
import difflib
from collections.abc import Mapping

import numpy as np
import pandas as pd

#: The longest expression accepted, in characters.
MAX_LENGTH = 300
#: The most syntax-tree nodes an expression may have.
MAX_NODES = 100
#: The deepest an expression's syntax tree may nest.
MAX_DEPTH = 20
#: The largest magnitude a power's (constant) exponent may have.
MAX_EXPONENT = 10

#: The reductions, by how many arguments each takes.
REDUCTIONS = {"count": (1, 1), "sum": (1, 2), "mean": (1, 2), "max": (1, 2), "min": (1, 2)}
#: The functions of one number.
FUNCTIONS = {"abs": np.abs, "sqrt": np.sqrt, "log10": np.log10}

_ARITHMETIC = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
}
_COMPARISONS = {
    ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
    ast.Eq: np.equal, ast.NotEq: np.not_equal,
}
_CONDITIONS = {ast.BitAnd: np.logical_and, ast.BitOr: np.logical_or}

#: How a construct that is not allowed is named in the error.
_REFUSED = {
    ast.Attribute: "attribute access (a.b)",
    ast.Subscript: "indexing (a[b])",
    ast.Lambda: "lambda",
    ast.IfExp: "conditional expressions (a if b else c)",
    ast.Dict: "dictionaries",
    ast.Set: "sets",
    ast.List: "lists",
    ast.Tuple: "tuples",
    ast.ListComp: "comprehensions",
    ast.SetComp: "comprehensions",
    ast.DictComp: "comprehensions",
    ast.GeneratorExp: "comprehensions",
    ast.NamedExpr: "assignment (:=)",
    ast.Starred: "starred arguments",
    ast.JoinedStr: "strings",
}


class ExpressionError(ValueError):
    """An expression that cannot be computed, with why in words."""


def evaluate(
    expression: str,
    events: pd.DataFrame,
    particles: Mapping[str, np.ndarray] | None = None,
    event_index: np.ndarray | None = None,
) -> pd.Series:
    """*expression*'s value in each event: one number per row of *events*,
    whose columns are the per-event names, indexed as *events*.

    *particles* holds one array per particle field, and *event_index* the
    position in *events* of each particle's event; without them, an
    expression with a reduction cannot be computed. Raises
    :class:`ExpressionError`, saying why, for one that cannot be.
    """
    tree = parse(expression)
    if particles is not None and (event_index is None or any(
        len(values) != len(event_index) for values in particles.values()
    )):
        raise ValueError("every particle field needs one value per particle of event_index")
    scope = _Scope(events, particles, event_index)
    with np.errstate(all="ignore"):
        value = scope.event(tree.body)
        values = np.broadcast_to(np.asarray(value, dtype=float), (len(events),))
        values = np.where(np.isfinite(values), values, np.nan)
    return pd.Series(values, index=events.index, name=expression.strip())


def parse(expression: str) -> ast.Expression:
    """*expression* parsed and checked against the size limits, without
    being computed; :class:`ExpressionError` when it is no expression or is
    too large."""
    text = expression.strip()
    if not text:
        raise ExpressionError("the expression is empty")
    if len(text) > MAX_LENGTH:
        raise ExpressionError(f"the expression is longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"not an expression: {exc.msg}") from None
    nodes = sum(1 for _ in ast.walk(tree))
    if nodes > MAX_NODES:
        raise ExpressionError(f"the expression has {nodes} parts, more than {MAX_NODES}")
    if _depth(tree) > MAX_DEPTH:
        raise ExpressionError(f"the expression nests deeper than {MAX_DEPTH}")
    return tree


def _depth(node: ast.AST) -> int:
    return 1 + max((_depth(child) for child in ast.iter_child_nodes(node)), default=0)


class _Scope:
    """The names an expression reads, and how each node is computed."""

    def __init__(self, events: pd.DataFrame, particles, event_index):
        self.events = events
        self.particles = particles
        self.event_index = event_index

    def event(self, node: ast.AST):
        """*node*'s value per event: an array of one per event, or a number."""
        return self._value(node, particle=False)

    def particle(self, node: ast.AST):
        """*node*'s value per particle, inside a reduction."""
        return self._value(node, particle=True)

    def _value(self, node: ast.AST, *, particle: bool):
        match node:
            case ast.Constant(value=value):
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ExpressionError(f"{value!r} is not a number")
                return np.float64(value)
            case ast.Name(id=name):
                return self._name(name, particle=particle)
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return -_number(self._value(operand, particle=particle))
            case ast.UnaryOp(op=ast.UAdd(), operand=operand):
                return _number(self._value(operand, particle=particle))
            case ast.UnaryOp(op=ast.Invert(), operand=operand):
                return np.logical_not(_condition(self._value(operand, particle=particle), "~"))
            case ast.UnaryOp(op=ast.Not()):
                raise ExpressionError("use ~ rather than not: a condition is an array")
            case ast.BinOp(op=ast.Pow(), left=left, right=right):
                return np.power(_number(self._value(left, particle=particle)), _exponent(right))
            case ast.BinOp(op=op, left=left, right=right) if type(op) in _ARITHMETIC:
                return _ARITHMETIC[type(op)](
                    _number(self._value(left, particle=particle)),
                    _number(self._value(right, particle=particle)),
                )
            case ast.BinOp(op=op, left=left, right=right) if type(op) in _CONDITIONS:
                symbol = "&" if isinstance(op, ast.BitAnd) else "|"
                return _CONDITIONS[type(op)](
                    _condition(self._value(left, particle=particle), symbol),
                    _condition(self._value(right, particle=particle), symbol),
                )
            case ast.BinOp(op=op):
                raise ExpressionError(f"the operator {_operator(op)} is not allowed")
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                return self._compare(left, ops, comparators, particle=particle)
            case ast.BoolOp():
                raise ExpressionError(
                    "use & and | rather than and and or, with each comparison in "
                    "parentheses: (pt > 1) & charged"
                )
            case ast.Call():
                return self._call(node, particle=particle)
        raise ExpressionError(f"{_REFUSED.get(type(node), type(node).__name__)} is not allowed")

    def _name(self, name: str, *, particle: bool):
        if particle:
            if name in self.particles:
                return self.particles[name]
            if name in self.events.columns:
                raise ExpressionError(
                    f"{name} is one number per event, not a particle field: use it "
                    "outside count(...), sum(...)"
                )
            raise ExpressionError(_unknown(name, "particle field", self.particles))
        if name in self.events.columns:
            values = pd.to_numeric(self.events[name], errors="coerce")
            return values.to_numpy(dtype=float)
        if self.particles is not None and name in self.particles:
            raise ExpressionError(
                f"{name} is a particle field: fold it into one number per event, "
                f"e.g. sum({name}, final) or max({name})"
            )
        known = list(self.events.columns) + (list(self.particles) if self.particles else [])
        raise ExpressionError(_unknown(name, "name", known))

    def _compare(self, left, ops, comparators, *, particle: bool):
        result = None
        current = _number(self._value(left, particle=particle))
        for op, right in zip(ops, comparators, strict=True):
            if type(op) not in _COMPARISONS:
                raise ExpressionError(f"the comparison {_operator(op)} is not allowed")
            following = _number(self._value(right, particle=particle))
            step = _COMPARISONS[type(op)](current, following)
            result = step if result is None else np.logical_and(result, step)
            current = following
        return result

    def _call(self, node: ast.Call, *, particle: bool):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only count, sum, mean, max, min, abs, sqrt and log10 can be called")
        name = node.func.id
        if node.keywords:
            raise ExpressionError(f"{name}(...) takes no named arguments")
        args = node.args
        if name in FUNCTIONS:
            if len(args) != 1:
                raise ExpressionError(f"{name}(...) takes one argument")
            return FUNCTIONS[name](_number(self._value(args[0], particle=particle)))
        if name not in REDUCTIONS:
            raise ExpressionError(_unknown(name, "function", [*REDUCTIONS, *FUNCTIONS]))
        if particle:
            raise ExpressionError(f"{name}(...) cannot be inside another count(...), sum(...)")
        if self.particles is None:
            raise ExpressionError(
                f"{name}(...) folds generator particles, and this run recorded none"
            )
        least, most = REDUCTIONS[name]
        if not least <= len(args) <= most:
            usage = "count(mask)" if name == "count" else f"{name}(x) or {name}(x, mask)"
            raise ExpressionError(f"use {usage}")
        n_particles = len(self.event_index)
        if name == "count":
            values = None
            mask = _particles(_condition(self.particle(args[0]), "count"), n_particles)
        else:
            values = _particles(_number(self.particle(args[0])), n_particles)
            mask = (
                _particles(_condition(self.particle(args[1]), f"{name}'s mask"), n_particles)
                if len(args) == 2 else np.ones(n_particles, dtype=bool)
            )
        return _reduce(name, values, mask, self.event_index, len(self.events))


def _number(value):
    """*value* as numbers: a condition counts as 1 or 0."""
    array = np.asarray(value)
    if array.dtype == bool:
        return array.astype(float)
    if array.dtype.kind not in "iuf":
        raise ExpressionError("only numbers and conditions can be computed with")
    return value


def _condition(value, where: str):
    array = np.asarray(value)
    if array.dtype != bool:
        raise ExpressionError(
            f"{where} needs a condition, e.g. (pt > 1) or charged, not a number"
        )
    return array


def _particles(value, n: int) -> np.ndarray:
    """*value* as one per particle, a constant repeated."""
    return np.broadcast_to(value, (n,))


def _exponent(node: ast.AST) -> np.float64:
    """A power's exponent: a constant number of magnitude at most
    :data:`MAX_EXPONENT`."""
    sign = 1.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        node = node.operand
    if not (isinstance(node, ast.Constant) and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)):
        raise ExpressionError("a power's exponent must be a number, e.g. pt**2")
    exponent = sign * float(node.value)
    if abs(exponent) > MAX_EXPONENT:
        raise ExpressionError(f"a power's exponent must be at most {MAX_EXPONENT} in magnitude")
    return np.float64(exponent)


def _reduce(name: str, values, mask: np.ndarray, event_index: np.ndarray, n: int) -> np.ndarray:
    """The particles *mask* selects folded per event by *name*."""
    if name == "count":
        return np.bincount(event_index[mask], minlength=n).astype(float)
    mask = mask & ~np.isnan(values)
    index, selected = event_index[mask], np.asarray(values, dtype=float)[mask]
    counts = np.bincount(index, minlength=n)
    if name == "sum":
        return np.bincount(index, weights=selected, minlength=n)
    if name == "mean":
        sums = np.bincount(index, weights=selected, minlength=n)
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    out = np.full(n, -np.inf if name == "max" else np.inf)
    (np.maximum if name == "max" else np.minimum).at(out, index, selected)
    return np.where(counts > 0, out, np.nan)


def _operator(op: ast.AST) -> str:
    symbols = {
        ast.Mod: "%", ast.FloorDiv: "//", ast.MatMult: "@", ast.LShift: "<<",
        ast.RShift: ">>", ast.BitXor: "^", ast.Is: "is", ast.IsNot: "is not",
        ast.In: "in", ast.NotIn: "not in",
    }
    return symbols.get(type(op), type(op).__name__)


def _unknown(name: str, kind: str, known) -> str:
    close = difflib.get_close_matches(name, list(known), n=3)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    return f"unknown {kind} {name}{hint}"
