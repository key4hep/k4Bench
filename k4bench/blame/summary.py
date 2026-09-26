"""The evidence summary both blame prompts open with.

A model given the evidence in the order it was computed — per-metric bullets,
history tables, a list of configurations that did not move, then twenty diffs —
has to reassemble the decisive comparisons itself, and on a wide night it does
not. The facts that decide an attribution are few: what moved and in what
shape, where in the removal sweep it did and did not, whether the machines
explain it, and which candidates reach the geometry — including the detectors
they reach that did *not* move. This module states them first, deterministically
and in the same words for both passes, from :mod:`k4bench.blame.sweep`,
:mod:`k4bench.blame.geometry` and :mod:`k4bench.blame.evidence`.

Every line either states a measured fact or says that it is unknown. A family
nobody judged, a run on an unreliable host, a configuration without event
records is named as such, never rendered as flat.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from k4bench.blame.evidence import MetricHistory
from k4bench.blame.geometry import DetectorTouch, FileChange, geometry_tree
from k4bench.blame.prompt import host_sentences, pct_phrase
from k4bench.blame.sweep import (
    ELSEWHERE,
    FAILED,
    OK,
    SETTLING,
    STEPPED,
    SWEEP_METRICS,
    UNJUDGED,
    WATCH,
    FamilyReading,
    ScopeSweep,
    SweepCell,
    SweepRow,
    TimeShape,
)
from k4bench.labels import pretty_sample
from k4bench.regression.models import RegionDelta

#: How each metric is named to the model.
METRIC_NAMES = {
    "mean_time_s": "mean event time",
    "median_time_s": "median event time",
    "trimmed_mean_time_s": "trimmed mean event time",
    "wall_time_s": "wall time",
    "peak_vmem_mb": "VmPeak",
    "mean_rss_anon_mb": "anonymous RSS",
}

#: Column headings in a sweep table.
_COLUMN = {
    "mean_time_s": "mean",
    "median_time_s": "median",
    "trimmed_mean_time_s": "trimmed",
    "wall_time_s": "wall",
    "peak_vmem_mb": "VmPeak",
    "mean_rss_anon_mb": "anonRSS",
}

_MARK = {
    STEPPED: "S", WATCH: "w", OK: "·", ELSEWHERE: "x", SETTLING: "~",
    UNJUDGED: "?", FAILED: "F",
}

SWEEP_LEGEND = (
    "S stepped in this window · w moved past the gates once, not confirmed · "
    "· within its normal variation · x stepped in another window · ~ still "
    "settling after an earlier step of its own · ? not judged · F failed; "
    "each percentage is the move against that configuration's own baseline"
)

#: Configurations a sweep table lists before it summarises the rest. Enough for
#: every sweep this suite runs today; a wider one keeps every configuration the
#: pattern is read from and summarises the quiet remainder, so the cap can
#: shorten the table but never hide the pattern.
MAX_SWEEP_ROWS = 60

#: Names listed in one reading line before the rest are counted.
_MAX_NAMED = 12

#: Configurations whose event records are spelled out per scope.
MAX_EVENT_RECORDS = 4

#: Files named per detector in the geometry map.
_MAX_TOUCH_FILES = 4

#: Constant changes named per file.
_MAX_CONSTANTS = 6


def scope_name(sweep: ScopeSweep) -> str:
    """``ILD_FCCee_v02 · single_e-_10GeV · x86_64-el9-gcc16-opt`` — the order
    the other evidence lines use."""
    return f"{sweep.detector} · {sweep.sample} · {sweep.platform}"


def _sample_phrase(sample: str) -> str:
    pretty = pretty_sample(sample)
    return f"{sample} ({pretty})" if pretty != sample else sample


def _pct(value: float | None) -> str:
    return pct_phrase(value)


def _spread(value: float) -> str:
    """An unsigned spread, to two significant figures below 0.1% — a memory
    series that varies by 0.002% must not read as one that does not vary."""
    if value == 0 or abs(value) >= 0.001:
        return pct_phrase(abs(value), signed=False)
    return f"{abs(value) * 100:.2g}%"


def _named(items: Sequence[str], limit: int = _MAX_NAMED) -> str:
    shown = list(items[:limit])
    rest = len(items) - len(shown)
    return ", ".join(shown) + (f", … (+{rest} more)" if rest > 0 else "")


# ── The sweep table ───────────────────────────────────────────────────────────

def _families_shown(sweep: ScopeSweep) -> list[str]:
    """The stepped families, then any other family judged somewhere."""
    stepped = list(sweep.stepped_families)
    rest = [
        family for family in SWEEP_METRICS
        if family not in stepped and any(
            (c := row.cell(m)) is not None and c.comparable
            for row in sweep.rows for m in SWEEP_METRICS[family]
        )
    ]
    return stepped + rest


def _cell_text(cell: SweepCell | None) -> str:
    if cell is None:
        return "—"
    mark = _MARK.get(cell.status, "?")
    if cell.pct is None or cell.status in (UNJUDGED, FAILED):
        text = mark
    else:
        text = f"{cell.pct * 100:+.1f}%{mark}"
    return text + (f"[{cell.fact_id}]" if cell.fact_id else "")


def _row_move(row: SweepRow, metric: str) -> float:
    cell = row.cell(metric)
    if cell is None or cell.pct is None or not math.isfinite(cell.pct):
        return 0.0
    return abs(cell.pct)


def _ordered_rows(sweep: ScopeSweep, families: Sequence[str]) -> list[SweepRow]:
    """The baseline first, then by the first family's lead move, largest first."""
    lead = sweep.lead_metric(families[0]) if families else ""
    rows = sorted(
        sweep.rows,
        key=lambda r: (r.label != "baseline", -_row_move(r, lead), r.label),
    )
    return rows


def sweep_table_lines(
    sweep: ScopeSweep, *, indent: str = "  ", max_rows: int = MAX_SWEEP_ROWS,
) -> list[str]:
    """Every configuration of *sweep* as one aligned row per configuration.

    Past *max_rows*, the configurations any reading names — the baseline, every
    stepped or watched one, the opposite-sign and larger ones — are kept and the
    quiet remainder is summarised with the range of its moves."""
    families = _families_shown(sweep)
    if not sweep.rows or not families:
        return []
    metrics = [m for family in families for m in SWEEP_METRICS[family]]
    rows = _ordered_rows(sweep, families)
    keep = set()
    for family in sweep.stepped_families:
        reading = sweep.reading(family)
        keep |= set(reading.stepped) | set(reading.opposite)
        keep |= {label for label, _r in reading.larger}
    keep |= {
        r.label for r in rows
        if r.label == "baseline" or any(c.status in (STEPPED, WATCH) for c in r.cells)
    }
    shown = rows
    hidden: list[SweepRow] = []
    if len(rows) > max_rows:
        shown = [r for r in rows if r.label in keep]
        for r in rows:
            if len(shown) >= max_rows:
                break
            if r not in shown:
                shown.append(r)
        hidden = [r for r in rows if r not in shown]
        shown = [r for r in rows if r in shown]
    width = min(32, max(len("configuration"), *(len(r.label) for r in shown)))
    texts = [[_cell_text(r.cell(m)) for m in metrics] for r in shown]
    widths = [
        max(len(_COLUMN[m]), *(len(t[i]) for t in texts)) for i, m in enumerate(metrics)
    ]

    def line(label: str, cells: list[str]) -> str:
        parts, at = [], 0
        for family in families:
            n = len(SWEEP_METRICS[family])
            parts.append("  ".join(
                cells[i].ljust(widths[i]) for i in range(at, at + n)
            ))
            at += n
        return f"{indent}  {label[:width].ljust(width)}  " + "  |  ".join(parts).rstrip()

    lines = [
        f"{indent}All {len(sweep.rows)} configurations of {scope_name(sweep)} in "
        f"this window ({SWEEP_LEGEND}):",
        line("configuration", [_COLUMN[m] for m in metrics]),
    ]
    lines += [line(r.label, t) for r, t in zip(shown, texts)]
    if hidden:
        lead = sweep.lead_metric(families[0])
        moves = [
            c.pct for r in hidden
            if (c := r.cell(lead)) is not None and c.comparable
        ]
        spread = (
            f": {METRIC_NAMES[lead]} {_pct(min(moves))} to {_pct(max(moves))}"
            if moves else ""
        )
        lines.append(
            f"{indent}  … {len(hidden)} further configuration(s), none stepped "
            f"or watched in this window{spread}"
        )
    return lines


# ── What the sweep's pattern says ─────────────────────────────────────────────

def _move_of(sweep: ScopeSweep, label: str, metric: str) -> str:
    row = sweep.row(label)
    cell = row.cell(metric) if row is not None else None
    return f"{label} {_pct(cell.pct)}" if cell is not None and cell.pct is not None else label


def _baseline_phrase(sweep: ScopeSweep, reading: FamilyReading) -> str:
    """What the baseline did in the lead metric, and in any other metric of
    the family that stepped there — each named as itself."""
    name = METRIC_NAMES[reading.lead]
    row = sweep.row("baseline")
    if row is None:
        return "There is no baseline configuration in this sweep."
    cell = reading.baseline
    if cell is not None and cell.status == STEPPED:
        return f"The baseline stepped: {name} {_pct(cell.pct)}."
    others = [
        f"{METRIC_NAMES[c.metric]} {_pct(c.pct)}" for c in row.cells
        if c.metric in SWEEP_METRICS[reading.family] and c.metric != reading.lead
        and c.status == STEPPED
    ]
    if cell is None:
        lead = f"The baseline has no {name} measurement"
    elif not cell.comparable:
        lead = f"The baseline's {name} was not judged ({cell.cause or cell.status})"
    else:
        word = "moved once without confirming" if cell.status == WATCH else "did not step"
        if not others:
            return f"The baseline {word}: {name} {_pct(cell.pct)}."
        lead = f"The baseline's {name} {word} ({_pct(cell.pct)})"
    if others:
        return f"{lead}, but it stepped in {', '.join(others)}."
    return f"{lead}."


def _short(metric: str) -> str:
    """``"mean"``, ``"median"``, ``"trimmed mean"``, ``"wall time"``, ``"VmPeak"``."""
    return METRIC_NAMES[metric].replace(" event time", "")


def _shape_item(label: str, shape: TimeShape) -> str:
    """One configuration's shape as its numbers — ``"no_TPC (mean +10.8%;
    median -0.7%, trimmed mean -0.5%)"``."""
    views = ", ".join(
        f"{_short(m)} {_pct(p)}" for m, p in shape.typical if m != shape.lead
    )
    return (
        f"{label} ({_short(shape.lead)} {_pct(shape.lead_pct)}"
        + (f"; {views}" if views else "") + ")"
    )


def _amount(cell: SweepCell, *, in_unit: bool) -> str:
    """A cell's move in the metric's own unit when *in_unit*, else as a
    percentage — the form the size ratios beside it were taken in."""
    if in_unit and cell.delta is not None:
        unit = "MB" if cell.metric.endswith("_mb") else "s"
        return f"{cell.delta:+.3g} {unit}"
    return _pct(cell.pct)


def _relative_lines(
    sweep: ScopeSweep, reading: FamilyReading, bullet: str,
) -> list[str]:
    """The configurations that moved less or more than the baseline, each with
    its own move and the baseline's in the unit the ratio was taken in."""
    base = reading.baseline
    lines = []
    for word, items in (("less", reading.smaller), ("more", reading.larger)):
        if not items or base is None:
            continue
        parts = []
        for label, ratio in items[:_MAX_NAMED]:
            cell = sweep.row(label).cell(reading.lead)
            in_unit = cell.delta is not None and bool(base.delta)
            parts.append(f"{label} {_amount(cell, in_unit=in_unit)} ({ratio:.0%} of it)")
        rest = len(items) - len(parts)
        in_unit = base.delta is not None and all(
            sweep.row(label).cell(reading.lead).delta is not None for label, _r in items
        )
        lines.append(
            f"{bullet}Moved the same way but {word} than the baseline's "
            f"{_amount(base, in_unit=in_unit)} in {METRIC_NAMES[reading.lead]}: "
            + "; ".join(parts) + (f"; … (+{rest} more)" if rest > 0 else "") + "."
        )
    return lines


def _shape_phrase(reading: FamilyReading) -> list[str]:
    """How the stepped time configurations are shaped, counted."""
    if not reading.shapes:
        return []
    by_kind: dict[str, list[str]] = {}
    for label, shape in reading.shapes:
        by_kind.setdefault(shape.kind, []).append(label)
    total = len(reading.stepped)
    shapes = dict(reading.shapes)

    def items(labels: list[str]) -> str:
        return _named([_shape_item(label, shapes[label]) for label in labels])

    lines = []
    tail = by_kind.get("tail", [])
    if tail:
        example = next(
            (s for label, s in reading.shapes if label == "baseline" and s.kind == "tail"),
            next(s for _l, s in reading.shapes if s.kind == "tail"),
        )
        label = next(name for name, s in reading.shapes if s is example)
        views = ", ".join(f"{_short(m)} {_pct(p)}" for m, p in example.typical)
        rest = [name for name in tail if name != label]
        lines.append(
            f"On {len(tail)} of {total} stepped configurations the typical event "
            f"did not follow — the median and trimmed mean moved less than a third "
            f"of the step, so a few long events carry it (e.g. {label}: "
            f"{METRIC_NAMES[example.lead]} {_pct(example.lead_pct)}, {views})"
            + (f"; the others: {items(rest)}." if rest else ".")
        )
    typical = by_kind.get("typical", [])
    if typical:
        lines.append(
            f"On {len(typical)} of {total} the typical event moved with the step "
            f"(median and trimmed mean followed it): {items(typical)}."
        )
    outside = by_kind.get("outside", [])
    if outside:
        lines.append(
            f"On {len(outside)} of {total} only the wall time stepped and the mean "
            f"event time did not follow — the step happened outside the event "
            f"loop (initialisation or teardown): {items(outside)}."
        )
    mixed = by_kind.get("mixed", [])
    if mixed:
        lines.append(
            f"On {len(mixed)} of {total} the typical event moved partly: "
            f"{items(mixed)}."
        )
    unread = total - len(reading.shapes)
    if unread:
        lines.append(
            f"On {unread} of {total} no typical view was judged, so their shape "
            f"is unknown."
        )
    return lines


def _also_lines(sweep: ScopeSweep, reading: FamilyReading) -> list[str]:
    """Each other metric of the family that stepped where the lead did not,
    with its own move and the lead's on the same configuration."""
    name = METRIC_NAMES[reading.lead]
    lines = []
    for metric, labels in reading.also:
        parts = []
        for label in labels:
            row = sweep.row(label)
            cell, lead = row.cell(metric), row.cell(reading.lead)
            beside = (
                f"{_short(reading.lead)} {_pct(lead.pct)}"
                if lead is not None and lead.comparable
                else f"{_short(reading.lead)} not judged"
            )
            parts.append(f"{label} {_pct(cell.pct)} ({beside})")
        lines.append(
            f"{METRIC_NAMES[metric]} independently stepped on {len(labels)} "
            f"additional configuration(s), where {name} did not step: "
            f"{_named(parts)}."
        )
    return lines


def family_reading_lines(
    sweep: ScopeSweep, family: str, *, indent: str = "  ",
) -> list[str]:
    """What *sweep*'s pattern says for one family, one fact per line."""
    reading = sweep.reading(family)
    name = METRIC_NAMES[reading.lead]
    bullet = f"{indent}- "
    if not reading.stepped:
        quiet = _quiet_family(sweep, reading)
        if not quiet:
            causes = sorted({cause for _label, cause in reading.unjudged})
            return [f"{bullet}{family}: {name} was not judged on any configuration ({', '.join(causes)})."]
        return [f"{bullet}{family}: {quiet}."]
    moves = [_move_of(sweep, label, reading.lead) for label in reading.stepped]
    directions = []
    if reading.down:
        directions.append(f"{reading.down} down")
    if reading.up:
        directions.append(f"{reading.up} up")
    unsized = len(reading.stepped) - reading.down - reading.up
    if unsized:
        directions.append(f"{unsized} of unknown size")
    lines = [
        f"{bullet}{family}: {name} stepped on {len(reading.stepped)} of "
        f"{reading.judged} judged configurations ({', '.join(directions)}): "
        f"{_named(moves)}."
    ]
    lines += [f"{bullet}{line}" for line in _also_lines(sweep, reading)]
    lines.append(f"{bullet}{_baseline_phrase(sweep, reading)}")
    if reading.opposite:
        if reading.up and reading.down and len(reading.opposite) == len(reading.stepped):
            lines.append(f"{bullet}They stepped in opposite directions.")
        else:
            lines.append(
                f"{bullet}Against the direction of the rest: "
                f"{_named([_move_of(sweep, label, reading.lead) for label in reading.opposite])}."
            )
    if reading.isolated:
        lines.append(
            f"{bullet}These are isolated configurations on a baseline that did "
            f"not move."
        )
    if reading.absent:
        lines.append(
            f"{bullet}Without the step — judged and moved less than a third of "
            f"it in {name} ({len(reading.absent)}): "
            f"{_named([_move_of(sweep, label, reading.lead) for label in reading.absent])}."
        )
    if reading.unconfirmed:
        lines.append(
            f"{bullet}Moved as far as a stepped configuration in {name} without "
            f"confirming ({len(reading.unconfirmed)}): "
            f"{_named([_move_of(sweep, label, reading.lead) for label in reading.unconfirmed])}."
        )
    lines += _relative_lines(sweep, reading, bullet)
    if reading.unjudged:
        causes: dict[str, int] = {}
        for _label, cause in reading.unjudged:
            causes[cause] = causes.get(cause, 0) + 1
        lines.append(
            f"{bullet}{name} could not be read on {len(reading.unjudged)} "
            f"configuration(s) ("
            + ", ".join(f"{n} {cause}" for cause, n in sorted(causes.items()))
            + ") — unknown, not flat."
        )
    lines += [f"{bullet}{line}" for line in _shape_phrase(reading)]
    if family == "memory":
        lines.append(
            f"{bullet}Memory is determined primarily by what the job loads rather "
            f"than by a few long events. Any machine dependence should be "
            f"assessed from the measured host evidence."
        )
    return lines


# ── The event records ─────────────────────────────────────────────────────────

def _long_event(event) -> str:
    where = ""
    if event.region and event.region_seconds is not None:
        where = f", {event.region_seconds:.1f} s of it in {event.region}"
    return f"{event.seconds:.1f} s (event {event.event}{where})"


#: An event number changed between the two ends when its relative move differs
#: from the typical event's (the median's) by more than this, and by more than
#: :data:`_CHANGED_EVENT_SECONDS` in absolute terms; otherwise it moved with the
#: typical event.
_CHANGED_EVENT_FRACTION = 0.25
_CHANGED_EVENT_SECONDS = 0.5


def _matched_phrase(profile) -> str:
    """Which of the longest events changed between the two ends beyond what
    the typical event did, and which moved with it — the same event number
    simulated at both ends."""
    if not profile.base.median:
        # Without the typical event's own move there is nothing to tell a
        # changed event from one that moved with it.
        return ""
    typical = (profile.onset.median - profile.base.median) / profile.base.median
    changed, held = [], []
    for m in profile.matched:
        if m.base is None or m.onset is None or m.base <= 0:
            continue
        beyond = m.onset - m.base * (1 + typical)
        if (
            abs(m.onset / m.base - 1 - typical) > _CHANGED_EVENT_FRACTION
            and abs(beyond) > _CHANGED_EVENT_SECONDS
        ):
            changed.append(f"event {m.event} went {m.base:.1f} s → {m.onset:.1f} s")
        else:
            held.append(f"event {m.event} ({m.base:.1f} s → {m.onset:.1f} s)")
    if not changed:
        return ""
    text = "the same event numbers at both ends: " + ", ".join(changed)
    if held:
        text += ", while " + ", ".join(held) + " moved with the typical event"
    return text


def event_record_line(label: str, shape: TimeShape, *, matched: bool = False) -> str | None:
    """One configuration's event records, or ``None`` without a profile;
    *matched* adds which of the longest events changed between the ends."""
    profile = shape.profile
    if profile is None or not profile.base.longest or not profile.onset.longest:
        return None
    bits = [
        f"{label}: longest event {_long_event(profile.base.longest[0])} at the "
        f"base, {_long_event(profile.onset.longest[0])} at the onset"
    ]
    if matched and (phrase := _matched_phrase(profile)):
        bits.append(phrase)
    moves = []
    if shape.mean_move is not None:
        moves.append(f"the mean moved {_pct(shape.mean_move)}")
    if shape.move_without_longest is not None:
        moves.append(
            f"{_pct(shape.move_without_longest)} without each end's longest event"
        )
    if shape.median_move is not None:
        moves.append(f"the median {_pct(shape.median_move)}")
    if moves:
        bits.append(", ".join(moves))
    share = shape.stepping_share
    if share is not None:
        bits.append(f"Geant4 stepping carries {share:.0%} of the mean's move")
    return "; ".join(bits) + "."


def event_record_lines(
    sweep: ScopeSweep, *, indent: str = "  ", limit: int = MAX_EVENT_RECORDS,
) -> list[str]:
    """The per-event records behind the stepped time configurations — the
    baseline first, then the largest moves — or nothing when the report
    recorded none."""
    if "time" not in sweep.stepped_families:
        return []
    reading = sweep.reading("time")
    shapes = sorted(
        (
            (label, shape) for label, shape in reading.shapes
            if shape.profile is not None
        ),
        key=lambda ls: (ls[0] != "baseline", -abs(ls[1].lead_pct), ls[0]),
    )
    lines = [
        line for index, (label, shape) in enumerate(shapes[:limit])
        if (line := event_record_line(label, shape, matched=index == 0)) is not None
    ]
    if not lines:
        return []
    out = [
        f"{indent}- Per-event records at both ends of the window (warm-up "
        f"excluded; a release measured on several nights is their median):"
    ]
    out += [f"{indent}    {line}" for line in lines]
    rest = len(shapes) - min(len(shapes), limit)
    if rest > 0:
        out.append(f"{indent}    (+{rest} more stepped configuration(s) with event records)")
    return out


#: Regions named per configuration in the summary; the history tables below
#: carry the full list.
_MAX_SUMMARY_REGIONS = 3


def _region_move(delta: RegionDelta) -> str:
    """``"HCAL +0.142"``; a region measured at one end only says so in words
    rather than as a move against zero."""
    if delta.base is None:
        return f"{delta.region} newly present ({delta.onset:.3g})"
    if delta.onset is None:
        return f"{delta.region} no longer measured (was {delta.base:.3g})"
    return f"{delta.region} {delta.delta:+.3g}"


def region_summary_lines(
    sweep: ScopeSweep, *, indent: str = "  ", limit: int = MAX_EVENT_RECORDS,
) -> list[str]:
    """The typical event's largest per-region movements on the configurations
    that stepped in time — the baseline first, then the largest moves — or
    nothing when no region timing was recorded for them.

    The movements are of per-event medians, so each says how the typical
    event's time in one region moved. They are never added up or turned into a
    region's share of the step: a sum of medians is not the median of a sum."""
    if "time" not in sweep.stepped_families:
        return []
    reading = sweep.reading("time")
    labels = list(reading.stepped) + [
        label for _metric, labels in reading.also for label in labels
    ]
    labels = sorted(dict.fromkeys(labels), key=lambda label: label != "baseline")
    rows = [
        row for label in labels
        if (row := sweep.row(label)) is not None and row.regions
    ]
    if not rows:
        return []
    out = [
        f"{indent}- Largest typical-event region movements (per-event medians, "
        f"s/event, base → onset):"
    ]
    out += [
        f"{indent}    {row.label}: "
        + "; ".join(_region_move(d) for d in row.regions[:_MAX_SUMMARY_REGIONS])
        + "."
        for row in rows[:limit]
    ]
    rest = len(rows) - min(len(rows), limit)
    if rest > 0:
        out.append(f"{indent}    (+{rest} more stepped configuration(s) with region timing)")
    return out


# ── One line per scope ────────────────────────────────────────────────────────

def _shape_summary(reading: FamilyReading) -> str:
    """The stepped configurations' shapes, counted per kind, each kind with one
    configuration's numbers — the baseline's where it has that shape."""
    kinds: dict[str, list[tuple[str, TimeShape]]] = {}
    for label, shape in reading.shapes:
        kinds.setdefault(shape.kind, []).append((label, shape))
    words = {
        "tail": "carried by a few long events (typical event unmoved)",
        "typical": "in the typical event",
        "outside": "outside the event loop",
        "mixed": "partly in the typical event",
    }
    parts = []
    for kind, items in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        label, shape = next(
            ((label, s) for label, s in items if label == "baseline"), items[0],
        )
        parts.append(f"{len(items)} {words[kind]}, e.g. {_shape_item(label, shape)}")
    return "; ".join(parts)


def _quiet_family(sweep: ScopeSweep, reading: FamilyReading) -> str:
    """A family nothing stepped in: the range it moved over, and every
    configuration that moved past the gates once without confirming — a first
    strike in this very window is not a configuration that held."""
    name = METRIC_NAMES[reading.lead]
    cells = [
        (row.label, c) for row in sweep.rows
        if (c := row.cell(reading.lead)) is not None and c.comparable
    ]
    if not cells:
        return ""
    moves = [c.pct for _label, c in cells]
    watched = sorted(
        ((label, c.pct) for label, c in cells if c.status == WATCH),
        key=lambda lp: (-abs(lp[1]), lp[0]),
    )
    text = (
        f"no {reading.family} step ({name} {_pct(min(moves))} to "
        f"{_pct(max(moves))} on {len(cells)} configurations"
    )
    if reading.unjudged:
        text += f", not judged on {len(reading.unjudged)}"
    base = reading.baseline
    if base is not None and base.comparable:
        text += f", baseline {_pct(base.pct)}"
    if watched:
        text += (
            f"; {len(watched)} moved past the gates once without confirming — "
            + _named([f"{label} {_pct(pct)}" for label, pct in watched], 4)
        )
    return text + ")"


def scope_state(sweep: ScopeSweep) -> str:
    """What *sweep* measured in this window, in one clause per family."""
    if sweep.unread:
        return f"no reading — {sweep.unread}; unknown, not flat"
    parts = []
    for family in _families_shown(sweep):
        reading = sweep.reading(family)
        name = METRIC_NAMES[reading.lead]
        if not reading.stepped:
            if quiet := _quiet_family(sweep, reading):
                parts.append(quiet)
            continue
        base = reading.baseline
        base_text = (
            f"baseline {_pct(base.pct)}{'' if 'baseline' in reading.stepped else ' (not stepped)'}"
            if base is not None and base.comparable else "baseline not judged"
        )
        text = (
            f"{name} stepped on {len(reading.stepped)} of {reading.judged} "
            f"({reading.down} down, {reading.up} up; {base_text}"
            + (f"; not judged on {len(reading.unjudged)}" if reading.unjudged else "")
            + ")"
        )
        if reading.isolated:
            text += ", isolated configurations on a flat baseline"
        shapes = _shape_summary(reading)
        if shapes:
            text += f" — {shapes}"
        for metric, labels in reading.also:
            text += (
                f"; {METRIC_NAMES[metric]} independently stepped on "
                f"{len(labels)} more where {name} did not"
            )
        parts.append(text)
    unjudged = [
        family for family in SWEEP_METRICS
        if family not in _families_shown(sweep)
    ]
    if unjudged:
        parts.append(f"{' and '.join(unjudged)} not judged")
    if sweep.failures:
        parts.append(
            f"{len(sweep.failures)} job failure(s) — {sweep.failures[0]}"
        )
    return "; ".join(parts) if parts else "nothing judged"


def scope_line(sweep: ScopeSweep) -> str:
    """*sweep* in one line: what stepped, how, and against which baseline."""
    where = f"{sweep.detector} · {_sample_phrase(sweep.sample)} · {sweep.platform}"
    return f"{where}: {scope_state(sweep)}."


# ── The metric's own history, compressed ──────────────────────────────────────

def history_summary(
    history: MetricHistory | None, metric: str, label: str = "baseline",
) -> list[str]:
    """What a representative metric's own history says: its noise, its record,
    its persistence and its machines. Empty without a history."""
    if history is None or not history.points:
        return []
    name = METRIC_NAMES.get(metric, metric)
    bits = []
    band = history.noise_band
    if band is not None:
        bits.append(f"this series normally varies by ±{_spread(band)}")
    quiet = history.quiet_boundary_move
    if quiet is not None:
        bits.append(
            f"it moved up to {pct_phrase(quiet, signed=False)} across "
            f"{history.quiet_boundaries} release boundary(ies) where no tracked "
            f"package changed"
        )
    before = history.before_window
    if before:
        bits.append(f"{history.prior_flags} of {len(before)} earlier releases were flagged")
    persistence = history.persistence
    after = len(history.after_onset)
    if persistence == "persisted":
        bits.append(f"the new level held across {after} later release(s)")
    elif persistence == "returned":
        bits.append("later releases came back towards the old level")
    elif not history.measured_after_onset:
        bits.append("no release after the onset has been measured yet")
    else:
        bits.append(
            f"whether the new level held across the {after} later release(s) "
            f"is unknown"
        )
    lines = [f"History of {label}'s {name}: " + "; ".join(bits) + "."]
    lines += host_sentences(history)
    spread = history.host_spread
    if spread is not None:
        value, releases = spread
        lines.append(
            f"Machines measuring the same release differed by up to "
            f"{_spread(value)} on this series ({releases} release(s) measured "
            f"on several machines)."
        )
    return lines


# ── What a candidate changes in benchmarked geometry ─────────────────────────

_VERSION_TOKEN = re.compile(r"_[ov]\d+")


def _stem(ref: str) -> str:
    base = ref.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return _VERSION_TOKEN.sub("", base)


def _include_phrases(change: FileChange) -> list[str]:
    removed, added = list(change.includes_removed), list(change.includes_added)
    phrases = []
    for old in list(removed):
        new = next((a for a in added if _stem(a) == _stem(old)), None)
        if new is not None:
            phrases.append(f"include switched {old} → {new}")
            removed.remove(old)
            added.remove(new)
    if added:
        phrases.append("include added " + ", ".join(added))
    if removed:
        phrases.append("include removed " + ", ".join(removed))
    return phrases


def file_change_phrase(change: FileChange) -> str:
    """One file's structural change in a clause."""
    name = change.path.rsplit("/", 1)[-1]
    if change.added:
        return f"{name}: new file"
    bits = []
    if change.constants_changed:
        shown = [
            f"{n} {old} → {new}"
            for n, old, new in change.constants_changed[:_MAX_CONSTANTS]
        ]
        rest = len(change.constants_changed) - len(shown)
        bits.append(
            "constants changed " + ", ".join(shown)
            + (f" (+{rest} more)" if rest > 0 else "")
        )
    if change.constants_added:
        names = [n for n, _v in change.constants_added]
        bits.append(f"{len(names)} constant(s) added ({_named(names, 4)})")
    if change.constants_removed:
        names = [n for n, _v in change.constants_removed]
        bits.append(f"{len(names)} constant(s) removed ({_named(names, 4)})")
    bits += _include_phrases(change)
    if change.other_lines:
        elements = [
            f"<{element}> ×{n}" if element else f"{n} continuing an element opened on an unchanged line"
            for element, n in change.other_elements
        ]
        bits.append(
            f"{change.other_lines} other changed line(s)"
            + (f" ({', '.join(elements)})" if elements else "")
        )
    if change.display_lines:
        bits.append(
            f"display settings only ({change.display_lines} lines)" if not bits
            else f"{change.display_lines} display-setting lines"
        )
    if not bits:
        bits.append("comments or whitespace only")
    text = f"{name}: " + "; ".join(bits)
    if change.clipped:
        text += " (its hunk was clipped, so this is a lower bound)"
    return text


def touch_lines(
    touches: Sequence[DetectorTouch],
    sweeps_by_detector: Mapping[str, Sequence[ScopeSweep]],
    *,
    this_detector: str = "",
    indent: str = "  ",
) -> list[str]:
    """The benchmarked detectors a candidate's diff reaches, what it changes in
    each, and what each measured in this window — the detector this run loads
    first, then those whose compact directory it changes, then those it reaches
    only elsewhere in their geometry tree."""
    if not touches:
        return []

    def order(touch: DetectorTouch) -> tuple:
        return (touch.detector != this_detector, not touch.own_files, touch.detector)

    lines = [
        f"{indent}benchmarked geometry it reaches (k4Bench's structural reading "
        f"of the diff; names, values and paths are quoted from it):"
    ]
    tree_only: list[DetectorTouch] = []
    for touch in sorted(touches, key=order):
        if not touch.own_files:
            tree_only.append(touch)
            continue
        where = " — this run" if touch.detector == this_detector else ""
        parts = [file_change_phrase(c) for c in touch.changes]
        if touch.unread:
            parts.append(f"{touch.unread} file(s) with no readable hunk")
        if touch.tree_files:
            parts.append(
                f"{len(touch.tree_files)} more file(s) elsewhere under "
                f"{geometry_tree(touch.geometry_path)}"
            )
        same = (
            f" The same change as {', '.join(touch.same_as)}." if touch.same_as else ""
        )
        lines.append(
            f"{indent}  - {touch.detector}{where}: its compact directory "
            f"{touch.own_dir} — " + "; ".join(parts) + "." + same
        )
        lines += _measured_lines(sweeps_by_detector.get(touch.detector, ()), indent)
    for touch in tree_only:
        where = " — this run" if touch.detector == this_detector else ""
        lines.append(
            f"{indent}  - {touch.detector}{where}: nothing in its compact "
            f"directory; {len(touch.tree_files)} file(s) elsewhere under "
            f"{geometry_tree(touch.geometry_path)} ("
            + _named([f.removeprefix(geometry_tree(touch.geometry_path)) for f in touch.tree_files], _MAX_TOUCH_FILES)
            + "), which it loads only if its compact files include them."
        )
        lines += _measured_lines(sweeps_by_detector.get(touch.detector, ()), indent)
    return lines


def _measured_lines(sweeps: Sequence[ScopeSweep], indent: str) -> list[str]:
    """What each of a detector's scopes measured in this window, one line each."""
    if not sweeps:
        return [f"{indent}      this window: none of its runs measured this window."]
    platforms = {sweep.platform for sweep in sweeps}
    return [
        f"{indent}      this window, {_sample_phrase(sweep.sample)}"
        + (f" · {sweep.platform}" if len(platforms) > 1 else "")
        + f": {scope_state(sweep)}."
        for sweep in sweeps
    ]



# ── Composing the summary ─────────────────────────────────────────────────────

EVIDENCE_HEADER = (
    "EVIDENCE SUMMARY — computed by k4Bench from the measurements; read it "
    "first. Each line states a measured fact or says that it is unknown; the "
    "details further down support it."
)


def scope_evidence_lines(
    sweep: ScopeSweep,
    *,
    history: MetricHistory | None = None,
    history_metric: str = "",
    history_label: str = "baseline",
    indent: str = "  ",
) -> list[str]:
    """Everything the summary says about one scope under judgement: each
    family's reading, the event records and region movements behind a time
    step, and the history and machines of its representative metric."""
    if sweep.unread:
        return [f"{indent}- No reading: {sweep.unread}; unknown, not flat."]
    lines = []
    for family in _families_shown(sweep):
        lines += family_reading_lines(sweep, family, indent=indent)
    for family in SWEEP_METRICS:
        if family not in _families_shown(sweep):
            lines.append(f"{indent}- {family}: not judged on any configuration.")
    for failure in sweep.failures:
        lines.append(f"{indent}- Job failure: {failure} — no reading for it, not flat.")
    lines += event_record_lines(sweep, indent=indent)
    lines += region_summary_lines(sweep, indent=indent)
    lines += [
        f"{indent}- {line}"
        for line in history_summary(history, history_metric, history_label)
    ]
    return lines


def other_scope_lines(
    sweeps: Sequence[ScopeSweep],
    *,
    exclude: set[tuple[str, str, str]] = frozenset(),
    indent: str = "",
) -> list[str]:
    """One line for every other scope that measured this window — including
    those whose run could not be read, which say so."""
    others = [sweep for sweep in sweeps if sweep.scope not in exclude]
    if not others:
        return []
    return [
        f"{indent}Every other benchmark scope that measured this window (same "
        f"releases, same night):",
        *(f"{indent}- {scope_line(sweep)}" for sweep in others),
    ]


def representative(
    sweep: ScopeSweep | None, candidates: Sequence[tuple[str, str, object]],
) -> tuple[str, str, object] | None:
    """``(label, metric, item)`` of the history a scope is summarised by: the
    baseline's lead metric in the most stepped family when it stepped, else
    another metric of that family the baseline stepped in, else the first of
    *candidates* — ``(label, metric, item)`` triples ordered largest move
    first."""
    if not candidates:
        return None
    if sweep is not None:
        for family in sweep.stepped_families:
            lead = sweep.lead_metric(family)
            for wanted in (
                lambda metric: metric == lead,
                lambda metric: metric in SWEEP_METRICS[family],
            ):
                for label, metric, item in candidates:
                    if label == "baseline" and wanted(metric):
                        return label, metric, item
    return candidates[0]
