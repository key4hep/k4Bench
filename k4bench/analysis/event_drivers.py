"""What an event's cost goes with: per-event costs set against what each event
holds.

An event's cost (its time, an algorithm's time, instructions or allocations,
the time and steps simulation spent in a detector) is a :class:`Measure`, one
value per event. What the event holds is its row of an :class:`EventTable`:
the canned quantities of :data:`FEATURES`, computed from its generator
particles (:mod:`k4bench.analysis.event_source`), and what each collection
holds in it (:mod:`k4bench.analysis.event_size`), one column each under a name
an expression (:mod:`k4bench.analysis.expression`) can use:

- ``n_<collection>``, ``E_<collection>``: the elements and calorimeter energy
  of the event itself, as the job read it: simulation's output, or
  reconstruction's input;
- ``out_n_<collection>``, ``out_E_<collection>``: reconstruction's outputs.

A name that would collide with another gets its file's name appended, then a
number; :attr:`Column.collection` and :attr:`Column.file` say what it is.

Matching events
---------------
A table's rows are the entries of the files it was read from; a measure's
values are keyed by the event number the job's profiling counted. The two
match when the profiling counted exactly the entries ``0 .. n-1``, which is
what :func:`align` checks; when they do not (a filtered output, an event range
not from the start), it says so and nothing is matched. The files' rows are
matched entry by entry: :func:`event_table` refuses files whose EventHeaders
name different events at the same entry (a reordered output), as far as they
have EventHeaders. The warm-up event 0 is left out of every measure, as in the
other per-event views.

What the numbers do and do not say
----------------------------------
:func:`associations` ranks the columns by Spearman's rank correlation with a
measure, so a cost that rises with a quantity in any monotonic way ranks high
and one with a symmetric or periodic dependence (on ``phi``, or on ``theta``
for a forward-backward symmetric cost) ranks low; the latter take a scatter,
or ``abs_cos_theta``. The top of a ranking picked from dozens of columns is
an association to look at, not a significance: no threshold is drawn.
:func:`trend` fits one straight line, robustly, and the events well above it
are above that line, not necessarily expensive for their content as a whole.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePath

import numpy as np
import pandas as pd

from k4bench.analysis.components import ComponentTiming
from k4bench.analysis.event_size import OutputFile
from k4bench.analysis.event_source import EventSource
from k4bench.analysis.expression import evaluate
from k4bench.analysis.robust import MIN_SLOPE_EVENTS, theil_sen_slope

#: The event that warms the job up, left out of every measure.
WARM_UP_EVENT = 0

#: Fewest paired events a correlation is computed from.
MIN_PAIRS = MIN_SLOPE_EVENTS

#: How far above the trend an event must lie to be listed: this many robust
#: standard deviations (1.4826 × the median absolute deviation).
OUTLIER_SIGMAS = 3.0
#: The median absolute deviation of a normal distribution, in standard deviations.
_MAD_SIGMA = 1.4826
#: Relative to the measure's typical size, the residual below which an event
#: lies on the trend line when most events lie exactly on it.
_ON_LINE = 1e-9
#: The rank correlation above which two quantities count as moving together
#: from event to event, so the association matrix shows one for both.
SIMILAR_RHO = 0.9
#: A ratio to the trend is shown only where the trend is at least this share of
#: the median measured value; near zero it would be noise.
MIN_RATIO_SHARE = 0.1

_NEUTRINOS = "(abs(pdg) != 12) & (abs(pdg) != 14) & (abs(pdg) != 16)"


@dataclass(frozen=True)
class Feature:
    """A canned per-event quantity of the generator particles."""

    name: str
    label: str
    expression: str
    #: Whether it is ranked: a periodic quantity is only plotted.
    ranked: bool = True


#: The canned quantities, in the expression grammar so each is also an example.
FEATURES: tuple[Feature, ...] = (
    Feature("n_final", "Final-state particles", "count(final)"),
    Feature("n_charged", "Charged final-state particles", "count(final & charged)"),
    Feature("n_photons", "Final-state photons", "count(final & (pdg == 22))"),
    Feature(
        "n_neutral_hadrons", "Final-state neutral hadrons",
        f"count(final & ~charged & (pdg != 22) & {_NEUTRINOS})",
    ),
    Feature(
        "n_displaced", "Charged particles born > 0.5 mm off the beam axis",
        "count(final & charged & (vr > 0.5))",
    ),
    Feature(
        "mean_abs_cos_theta", "Mean |cos θ| of charged particles",
        "mean(abs_cos_theta, final & charged)",
    ),
    Feature("max_pt", "Highest final-state pT (GeV)", "max(pt, final)"),
    Feature("sum_pt_charged", "Summed charged pT (GeV)", "sum(pt, final & charged)"),
    Feature("visible_E", "Visible energy (GeV)", f"sum(E, final & {_NEUTRINOS})"),
)

#: The quantities of a sample with one final-state particle per event (a gun).
PRIMARY_FEATURES: tuple[Feature, ...] = (
    Feature("primary_abs_cos_theta", "Primary |cos θ|", "max(abs_cos_theta, final)"),
    Feature("primary_theta", "Primary θ (rad)", "max(theta, final)"),
    Feature("primary_phi", "Primary φ (rad)", "max(phi, final)", ranked=False),
    Feature("primary_E", "Primary energy (GeV)", "max(E, final)"),
)


@dataclass(frozen=True)
class Column:
    """One column of an :class:`EventTable`."""

    label: str
    #: ``truth`` (a canned quantity), ``content`` (the event as the job read
    #: it) or ``output`` (reconstruction's outputs).
    kind: str
    ranked: bool = True
    #: For a content column, the collection it counts and its file.
    collection: str | None = None
    file: str | None = None
    expression: str | None = None


@dataclass(frozen=True)
class EventTable:
    """What each event holds: one row per entry, one column per quantity."""

    frame: pd.DataFrame
    columns: Mapping[str, Column]
    #: The generator particles, for expressions; ``None`` without a source.
    source: EventSource | None = None
    #: The output files left out, each with why: they hold other events
    #: than the job read, or nothing shows they hold them in its order.
    left_out: Mapping[str, str] = field(default_factory=dict)
    _particles: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def events(self) -> int:
        return len(self.frame)

    @property
    def has_truth(self) -> bool:
        return self.source is not None

    def particles(self) -> dict[str, np.ndarray] | None:
        if self.source is None:
            return None
        if not self._particles:
            self._particles.update(self.source.particles())
        return self._particles

    def evaluate(self, expression: str) -> pd.Series:
        """*expression* in each event (:func:`k4bench.analysis.expression.evaluate`)."""
        particles = self.particles()
        return evaluate(
            expression, self.frame, particles,
            self.source.event_index if particles is not None else None,
        )

    def ranked(self) -> list[str]:
        return [name for name, column in self.columns.items() if column.ranked]


def event_table(
    source: EventSource | None,
    content: Sequence[OutputFile] = (),
    outputs: Sequence[OutputFile] = (),
) -> EventTable | None:
    """The table of a run's events: :data:`FEATURES` from *source*, and each
    collection's elements and energy from *content* (simulation's output
    file) or the source's input content, and from *outputs*
    (reconstruction's files), each measured file for as many events as the
    others; ``None`` when there is nothing to tabulate.

    Each output is checked against *source* on its own and left out, with
    why (:attr:`EventTable.left_out`), when it holds other events: another
    number of them, other EventHeaders, or an entry without one, its or the
    input's, which leaves its order unchecked.

    Raises ``ValueError`` for *outputs* without *source*, since only the
    input's EventHeaders show that the outputs hold the events in the order
    the job read them, and when the source and *content* disagree on the
    number of events or their EventHeaders on which event an entry is.
    """
    if source is None and any(_measured(f) for f in outputs):
        raise ValueError(
            "the job's input was not recorded, so nothing shows that its outputs "
            "hold the events in the order it read them"
        )
    left_out = {
        f.name: why for f in outputs if _measured(f) and (why := _unmatched(f, source))
    }
    outputs = [f for f in outputs if f.name not in left_out]
    measured = [f for f in (*content, *outputs) if _measured(f)]
    sizes = {
        n for n in (
            source.events if source is not None else None,
            *(f.events for f in measured),
        ) if n is not None
    }
    if not sizes:
        return None
    if len(sizes) > 1:
        raise ValueError(f"the event files disagree on the number of events: {sorted(sizes)}")
    _check_same_events([
        *([(source.file, source.event_numbers, source.run_numbers)] if source is not None else []),
        *((f.name, f.event_numbers, f.run_numbers) for f in measured),
    ])
    n = sizes.pop()
    frame = pd.DataFrame(index=pd.RangeIndex(n, name="entry"))
    columns: dict[str, Column] = {}
    taken = {f.name for f in (*FEATURES, *PRIMARY_FEATURES)}

    def add(name: str, values, column: Column) -> None:
        frame[name] = np.asarray(values, dtype=float)
        columns[name] = column
        taken.add(name)

    if source is not None:
        _add_content(add, taken, "", source.input_counts, source.input_energies, source.file,
                     "content")
    for output in content:
        if _measured(output):
            _add_content(add, taken, "", output.event_counts(), output.event_energies(),
                         output.name, "content")
    for output in outputs:
        if _measured(output):
            _add_content(add, taken, "out_", output.event_counts(), output.event_energies(),
                         output.name, "output")

    table = EventTable(frame, columns, source, left_out)
    if source is not None:
        features = list(FEATURES)
        if _one_primary(table):
            features += PRIMARY_FEATURES
        truth = {f.name: table.evaluate(f.expression) for f in features}
        for f in features:
            frame[f.name] = truth[f.name].to_numpy()
        columns = {
            **{f.name: Column(f.label, "truth", f.ranked, expression=f.expression) for f in features},
            **columns,
        }
        table = EventTable(frame, columns, source, left_out)
    return table


def _numbered(event_numbers: Sequence[int | None] | None) -> bool:
    return event_numbers is not None and None not in event_numbers


def _unmatched(output: OutputFile, source: EventSource) -> str | None:
    """Why *output* cannot be matched to *source* entry by entry, or
    ``None`` when it can."""
    if output.events != source.events:
        return f"it holds {output.events} events, the job read {source.events}"
    if not (_numbered(source.event_numbers) and _numbered(output.event_numbers)):
        return "an event without an EventHeader, its or the input's, leaves its order unchecked"
    try:
        _check_same_events([
            (source.file, source.event_numbers, source.run_numbers),
            (output.name, output.event_numbers, output.run_numbers),
        ])
    except ValueError as exc:
        return str(exc)
    return None


def _measured(output: OutputFile) -> bool:
    return output.events is not None and any(
        c.entries is not None or c.energy_gev is not None for c in output.collections
    )


def _check_same_events(headers: Sequence[tuple[str, Sequence | None, Sequence | None]]) -> None:
    """Raises ``ValueError`` when two of *headers* (each file's name and its
    entries' EventHeader event and run numbers) name different events at the
    same entry; an entry or a number a file lacks is not compared. Each
    entry's numbers are set against the first file to have them, so a file
    without them between two others hides no disagreement."""
    keyed = [
        (name, events, runs if runs is not None else [None] * len(events))
        for name, events, runs in headers if events is not None
    ]
    # Per entry, the first file naming its event, and the first naming its run.
    event_of: dict[int, tuple[str, int]] = {}
    run_of: dict[int, tuple[str, int]] = {}
    for name, events, runs in keyed:
        for entry, (event, run) in enumerate(zip(events, runs)):
            for number, first, what in ((event, event_of, "event"), (run, run_of, "run")):
                if number is None:
                    continue
                other, known = first.setdefault(entry, (name, number))
                if known != number:
                    raise ValueError(
                        f"{other} and {name} hold different events at entry {entry}: "
                        f"{what} {known} and {what} {number}"
                    )


def _one_primary(table: EventTable) -> bool:
    counts = table.evaluate("count(final)")
    return len(counts) > 0 and bool((counts == 1).all())


def _add_content(add, taken: set[str], prefix: str, counts: pd.DataFrame,
                 energies: pd.DataFrame, file: str, kind: str) -> None:
    for quantity, frame in (("n_", counts), ("E_", energies)):
        unit = " (GeV)" if quantity == "E_" else ""
        what = "energy" if quantity == "E_" else "elements"
        for collection in frame.columns:
            name = alias(f"{prefix}{quantity}", collection, file, taken)
            label = f"{collection} {what}{unit}" + (" (output)" if kind == "output" else "")
            add(name, frame[collection].to_numpy(), Column(label, kind, True, collection, file))


def alias(prefix: str, collection: str, file: str, taken: set[str]) -> str:
    """The expression name of *collection* of *file* under *prefix*: its
    name with every character that cannot be in a name made ``_``, then, if
    that is *taken*, the file's name appended, then a number."""
    name = _identifier(prefix + collection)
    if name in taken:
        name = _identifier(f"{name}_{PurePath(file).name.split('.')[0]}")
    base, number = name, 2
    while name in taken:
        name, number = f"{base}_{number}", number + 1
    return name


def _identifier(text: str) -> str:
    name = re.sub(r"\W", "_", text)
    return f"_{name}" if name[:1].isdigit() else name


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Measure:
    """One per-event cost of one configuration."""

    #: What it is the cost of: the whole event, an algorithm, a detector.
    name: str
    #: Which cost: ``time``, ``steps``, or a component metric.
    metric: str
    unit: str
    #: One value per event, indexed by the event number profiling counted,
    #: the warm-up left out.
    values: pd.Series
    #: The event numbers the job's profiling counted, the warm-up included.
    profiled: tuple[int, ...]


#: The component metrics offered, with their unit and how each is read.
MEASURE_METRICS = {
    "wall_s": ("time", "s"),
    "cpu_s": ("CPU time", "s"),
    "instructions": ("instructions", ""),
    "allocations": ("allocations", ""),
    "allocated_bytes": ("allocated bytes", "B"),
    # Includes the calls it made: a sequencer's is its children's largest.
    "peak_heap_bytes": ("peak heap (incl. nested calls)", "B"),
}

WHOLE_EVENT = "Whole event"


def _without_warm_up(values: pd.Series) -> pd.Series:
    return values[values.index != WARM_UP_EVENT].dropna()


def _costliest_first(table: pd.DataFrame) -> list:
    """*table*'s columns, the largest mean past the warm-up first."""
    return list(table.loc[table.index != WARM_UP_EVENT].mean().sort_values(ascending=False).index)


def simulation_measures(event_data: pd.DataFrame | None, regions: Mapping | None) -> list[Measure]:
    """The per-event costs of one simulation configuration: its event time
    (*event_data*, :func:`~k4bench.analysis.loader.load_event_timing`'s frame)
    and the time and steps spent in each detector (*regions*, its
    :func:`~k4bench.analysis.loader.load_region_timing` entry), the
    costliest detectors of each first."""
    measures: list[Measure] = []
    if event_data is not None and not event_data.empty:
        times = event_data.set_index("event_number")["event_time_s"]
        measures.append(Measure(WHOLE_EVENT, "time", "s", _without_warm_up(times),
                                tuple(int(e) for e in times.index)))
    if regions:
        at_location = regions.get("at_location")
        steps = regions.get("steps")
        if at_location is not None and not at_location.empty:
            profiled = tuple(int(e) for e in at_location.index)
            for detector in _costliest_first(at_location):
                measures.append(Measure(str(detector), "time", "s",
                                        _without_warm_up(at_location[detector]), profiled))
            if steps is not None:
                for detector in _costliest_first(steps):
                    measures.append(Measure(str(detector), "steps", "",
                                            _without_warm_up(steps[detector]),
                                            tuple(int(e) for e in steps.index)))
    return measures


def reconstruction_measures(
    event_data: pd.DataFrame | None, timing: ComponentTiming | None,
) -> list[Measure]:
    """The per-event costs of one reconstruction configuration: its event
    time (*event_data*) and, per algorithm, each of
    :data:`MEASURE_METRICS` its auditor file (*timing*) recorded, the
    costliest algorithms of each first."""
    measures: list[Measure] = []
    if event_data is not None and not event_data.empty:
        times = event_data.set_index("event_number")["event_time_s"]
        measures.append(Measure(WHOLE_EVENT, "time", "s", _without_warm_up(times),
                                tuple(int(e) for e in times.index)))
    if timing is None:
        return measures
    wall = timing.execute["wall_s"]
    algorithms = [a for a in timing.algorithms() if a in wall.columns]
    profiled = tuple(int(e) for e in wall.index)
    for metric, (what, unit) in MEASURE_METRICS.items():
        table = timing.execute.get(metric)
        if table is None:
            continue
        for algorithm in _costliest_first(table[[a for a in algorithms if a in table.columns]]):
            values = _without_warm_up(table[algorithm])
            if len(values):
                measures.append(Measure(str(algorithm), what, unit, values, profiled))
    return measures


# ---------------------------------------------------------------------------
# Matching, ranking and the trend
# ---------------------------------------------------------------------------


def align(measure: Measure, table: EventTable) -> str | None:
    """Why *measure*'s events cannot be matched to *table*'s rows, or
    ``None`` when they can: the job's profiling must have counted exactly
    the entries ``0 .. n-1`` of the table's files."""
    profiled = measure.profiled
    if len(set(profiled)) != len(profiled):
        return "the job's profiling counted some events twice"
    if sorted(profiled) != list(range(table.events)):
        return (
            f"the job's profiling counted {len(profiled)} events, numbered "
            f"{min(profiled, default='–')}–{max(profiled, default='–')}, but its event "
            f"files hold the entries 0–{table.events - 1}"
        )
    return None


def paired(measure: Measure, x: pd.Series) -> pd.DataFrame:
    """*measure*'s events beside *x* (a table column, indexed by entry), the
    events either lacks dropped: ``event``, ``x`` and ``y``."""
    y = measure.values
    joined = pd.DataFrame({"x": x.reindex(y.index).to_numpy(), "y": y.to_numpy()}, index=y.index)
    joined.index.name = "event"
    return joined.dropna()


def spearman(x: pd.Series, y: pd.Series) -> float | None:
    """Spearman's rank correlation of *x* and *y* over the events both have,
    or ``None`` with fewer than :data:`MIN_PAIRS` of them or either constant."""
    both = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(both) < MIN_PAIRS or both["x"].nunique() < 2 or both["y"].nunique() < 2:
        return None
    return float(both["x"].rank().corr(both["y"].rank()))


def associations(measure: Measure, table: EventTable,
                 columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Every ranked column of *table* (or *columns*) by the size of its rank
    correlation with *measure*, largest first: ``rho`` with its sign, and
    ``n``, the events paired. Columns without a correlation (constant,
    too few events) are left out.

    The columns with a value in every event of *measure* are ranked and
    correlated together; each of the others over the events it has
    (:func:`spearman`)."""
    names = list(columns if columns is not None else table.ranked())
    y = measure.values.dropna()
    x = table.frame.reindex(y.index)[names]
    complete = x.columns[x.notna().all()]
    rows = []
    if len(y) >= MIN_PAIRS and len(complete):
        rho = _rank_correlations(x[complete], y)
        rows += [(name, r, len(y)) for name, r in rho.items() if np.isfinite(r)]
    for name in x.columns.difference(complete, sort=False):
        pairs = paired(measure, table.frame[name])
        if (r := spearman(pairs["x"], pairs["y"])) is not None:
            rows.append((name, r, len(pairs)))
    frame = pd.DataFrame(rows, columns=["column", "rho", "n"]).set_index("column")
    return frame.reindex(frame["rho"].abs().sort_values(ascending=False, kind="stable").index)


def _rank_correlations(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Spearman's rank correlation of each column of *x* with *y*, all over
    the same events: NaN for a constant column or a constant *y*."""
    ranks = x.rank().to_numpy(dtype=float, copy=True)
    y_ranks = y.rank().to_numpy(dtype=float, copy=True)
    ranks -= ranks.mean(axis=0)
    y_ranks -= y_ranks.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (ranks * y_ranks[:, None]).sum(axis=0) / np.sqrt(
            (ranks**2).sum(axis=0) * (y_ranks**2).sum()
        )
    return pd.Series(rho, index=x.columns)


@dataclass(frozen=True)
class Trend:
    """A straight line through a measure against one quantity: Theil–Sen's
    slope, and the median intercept."""

    slope: float
    intercept: float

    def at(self, x) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(x, dtype=float)


def trend(x: pd.Series, y: pd.Series) -> Trend | None:
    """The robust line of *y* against *x*; ``None`` with fewer than
    :data:`MIN_SLOPE_EVENTS` events or no two distinct *x*."""
    both = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    slope = theil_sen_slope(both["x"], both["y"])
    if slope is None:
        return None
    return Trend(slope, float(np.median(both["y"] - slope * both["x"])))


def above_trend(pairs: pd.DataFrame, line: Trend) -> pd.DataFrame:
    """The events of *pairs* (:func:`paired`) more than
    :data:`OUTLIER_SIGMAS` robust standard deviations above *line*, furthest
    first: ``x``, ``y``, ``expected``, ``residual`` and ``ratio`` (NaN where
    the line is too near zero, :data:`MIN_RATIO_SHARE`). When most events lie
    exactly on the line, as a count that rarely changes does, every event
    above it beyond rounding (:data:`_ON_LINE`)."""
    expected = line.at(pairs["x"])
    residual = pairs["y"].to_numpy() - expected
    spread = _MAD_SIGMA * float(np.median(np.abs(residual - np.median(residual))))
    if not spread > 0:
        spread = _ON_LINE * max(float(np.median(np.abs(pairs["y"]))), 1.0) / OUTLIER_SIGMAS
    floor = MIN_RATIO_SHARE * abs(float(np.median(pairs["y"])))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(expected > floor, pairs["y"].to_numpy() / expected, np.nan)
    out = pairs.assign(expected=expected, residual=residual, ratio=ratio)
    return out[out["residual"] > OUTLIER_SIGMAS * spread].sort_values("residual", ascending=False)



def association_matrix(
    measures: Sequence[Measure], table: EventTable, *,
    rows: int = 15, columns: int = 12, per_row: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """The rank correlations of the first *rows* of *measures* that match
    *table* with the columns most associated with them: the *per_row*
    strongest of each, the strongest overall first, at most *columns* of
    them, a column that moves with one already chosen (:data:`SIMILAR_RHO`)
    folded into it. ``(rho, n, similar)``: the first two indexed by measure
    name with one column per table column, NaN where a pair has no
    correlation; the last, each chosen column's folded ones."""
    ranked = {
        m.name: associations(m, table)
        for m in [m for m in measures if align(m, table) is None][:rows]
    }
    strength: dict[str, float] = {}
    for frame in ranked.values():
        for name, rho in frame["rho"].head(per_row).items():
            strength[name] = max(strength.get(name, 0.0), abs(rho))
    candidates = sorted(strength, key=lambda name: -strength[name])
    alike = table.frame[candidates].rank().corr().abs()
    similar: dict[str, list[str]] = {}
    for name in candidates:
        twin = next((c for c in similar if alike.loc[name, c] > SIMILAR_RHO), None)
        if twin is not None:
            similar[twin].append(name)
        elif len(similar) < columns:
            similar[name] = []
    chosen = list(similar)
    rho = pd.DataFrame(
        {name: frame["rho"].reindex(chosen) for name, frame in ranked.items()},
    ).T.reindex(columns=chosen)
    n = pd.DataFrame(
        {name: frame["n"].reindex(chosen) for name, frame in ranked.items()},
    ).T.reindex(columns=chosen)
    return rho, n, similar
