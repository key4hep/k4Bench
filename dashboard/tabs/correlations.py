"""Correlations: what an event's cost goes with.

Shared by the Simulation and the Reconstruction page. It sets the per-event
costs of one configuration against what each of its events holds
(:mod:`k4bench.analysis.event_drivers`): the generator particles of the run's
``event_source.json.gz`` and what each collection held in each event, from the
configuration's ``_event_size.json`` and, in reconstruction, the input's
collections. Reconstruction offers each configuration that succeeded, the
baseline first; every one reads the same input, so they share the source,
and each its own outputs. A simulation benchmark records what its events
hold for its first run only, and its other configurations take a detector
out, so it draws the baseline (its single configuration without one) alone.

Top to bottom: the configuration, in reconstruction; the metric (time,
steps, instructions, ...); a matrix of the
costliest costs of that metric against the quantities most associated with
them, each square coloured and sized by their rank correlation, a click
plotting it; one row choosing the cost to plot (Of), the expression it is
plotted against and a log y axis; what an expression can use, with examples
that work for the run; the scatter with its robust trend line; the events
well above that line; and, in reconstruction, one event of the scatter, chosen
by a click on its point or by its number: which entry of the input it is, its
values plotted, and what each algorithm cost in it of the metric drawn, a
click on one opening it in Components. The warm-up event is left out, as in
the other per-event views.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data import cached_load_event_size, cached_load_event_source
from k4bench.analysis.components import ComponentTiming
from k4bench.analysis.event_drivers import (
    WARM_UP_EVENT,
    WHOLE_EVENT,
    EventTable,
    Measure,
    above_trend,
    align,
    association_matrix,
    associations,
    event_table,
    paired,
    reconstruction_measures,
    simulation_measures,
    trend,
)
from k4bench.analysis.event_source import PARTICLE_FIELDS
from k4bench.analysis.expression import MAX_LENGTH, ExpressionError
from k4bench.analysis.loader import failed_config_mask, judgeable_config_data
from k4bench.analysis.plots.event_content import (
    plot_association_matrix,
    plot_event_content,
    plot_event_costs,
)
from k4bench.labels import BASELINE_LABEL
from reco._controls import select_config
from reco._inspect import clickable_chart
from sections import SectionScope
from ui_chrome import drop_stale_selection

_KEY = "corr"
_CONFIG_KEY = f"{_KEY}_config"
#: A configuration asked for that is not on offer, and the one shown
#: instead, said so for as long as that one is shown.
_MISSING_CONFIG_KEY = f"{_KEY}_config_missing"
#: The configuration drawn on the previous run, to tell when it changes.
_SHOWN_KEY = f"{_KEY}_config_shown"
_METRIC_KEY = f"{_KEY}_metric"
_MEASURE_KEY = f"{_KEY}_of"
_EXPRESSION_KEY = f"{_KEY}_x"
_EXAMPLE_KEY = f"{_KEY}_example"
_LOG_KEY = f"{_KEY}_log_y"
_MATRIX_KEY = f"{_KEY}_matrix"
#: A cost and expression a click on the matrix chose, applied on the rerun it
#: starts, before the selectors they set are drawn.
_PENDING_KEY = f"{_KEY}_pending"
_CHART_KEY = f"{_KEY}_chart"
#: The event shown on its own, and the chart of its algorithms.
_EVENT_KEY = f"{_KEY}_event"
_EVENT_CHART_KEY = f"{_KEY}_event_chart"
#: The Reconstruction page's section this is drawn in, which the Components
#: section, opened from it, offers to go back to.
_SECTION = "Correlations"

#: The costs of the chosen kind the matrix draws, costliest first, and the
#: quantities it sets them against.
_MATRIX_ROWS = 15
_MATRIX_COLUMNS = 12

#: How each unit is shown: its scale and its name on screen.
_UNITS = {"s": (1e3, "ms"), "B": (1 / 1024**2, "MB"), "": (1.0, "")}

_ORIGINS = {
    "backfill-verified": "Generator particles added later, from the very file this run read.",
    "backfill-inferred": (
        "Generator particles added later, from another detector's simulation of this "
        "sample: treat them as inferred."
    ),
}

_MATRIX_HELP = (
    "Each square sets a cost (down, costliest first) against a quantity of the "
    "events (across: those most associated with the costs, by the names an "
    "expression uses). Its colour is Spearman's rank correlation ρ over the "
    "events: red, the cost rises with the quantity; blue, it falls; its size, how "
    "consistently. Quantities that move together from event to event share a "
    "column, the others named when you hover it. A cost that depends on a quantity symmetrically (on θ, forward "
    "and backward alike) or periodically (on φ) shows little however strongly it "
    "depends on it: plot it. Among this many pairs a few strong ones turn up by "
    "chance: a square is a lead, not proof of a cause."
)

_EXPRESSION_LINE = (
    "Any number per event: a quantity's name, like `n_charged`, or the generator "
    "particles folded into one number, like `count(final & charged & (pt > 1))`, "
    "combined with `+ - * /`."
)

#: What each reduction and operator does, for the reference.
_GRAMMAR = [
    ("count(mask)", "the particles of the event the condition selects"),
    ("sum(x), sum(x, mask)", "x summed over the event's particles, or those selected; 0 for none"),
    ("mean(x), mean(x, mask)", "x averaged; empty when none are selected"),
    ("max(x), max(x, mask)", "the largest x; empty when none are selected"),
    ("min(x), min(x, mask)", "the smallest x; empty when none are selected"),
    ("& | ~", "and, or, not, between conditions: (pt > 1) & charged"),
    ("< <= > >= == !=", "comparisons, which give conditions; they chain: 0.5 < vr < 10"),
    ("+ - * / **", "arithmetic; a power only to a number up to 10: pt**2"),
    ("abs(x) sqrt(x) log10(x)", "functions of one number"),
]

#: Where each kind of quantity comes from, as the reference says it.
_KINDS = {
    "truth": "generator particles",
    "content": "the event's collections",
    "output": "the job's outputs",
}


def render(
    data_dir: str,
    results: pd.DataFrame | None,
    event_data: dict | None,
    *,
    region_data: dict | None = None,
    components: dict[str, ComponentTiming] | None = None,
) -> SectionScope | None:
    """The section for the run in *data_dir*: a simulation run's
    *event_data* and *region_data*, or a reconstruction run's *event_data*
    and *components*, each per configuration."""
    reco = components is not None
    source, problem = cached_load_event_source(data_dir)
    if problem:
        st.warning(f"The run's generator particles could not be read: {problem}")
    files, unreadable = cached_load_event_size(data_dir)
    for label, reason in unreadable.items():
        st.warning(f"What **{label}**'s events hold could not be read: {reason}")
    files = judgeable_config_data(files, results) or {}
    event_data = judgeable_config_data(event_data, results) or {}
    costs = (
        judgeable_config_data(components, results) or {} if reco
        else judgeable_config_data(region_data, results) or {}
    )
    configs = sorted(set(event_data) | set(costs), key=lambda c: (c != BASELINE_LABEL, c))
    if not configs:
        st.info("No per-event costs in this run.")
        return None
    if source is None and not any(_has_content(f) for f in files.values()):
        st.info(
            "This run recorded nothing of what its events hold: its benchmark predates "
            "it, or the run's output and input could not be read."
        )
        return None

    # A click on the matrix chose these on the previous run.
    pending = st.session_state.pop(_PENDING_KEY, None)

    choosing = reco and len(configs) > 1
    if choosing:
        config_col, metric_col, note_col = st.columns(
            [1, 1, 2], gap="medium", vertical_alignment="bottom",
        )
    else:
        metric_col, note_col = st.columns([1, 3], gap="medium", vertical_alignment="bottom")
    if reco:
        config = _select_config(configs, results, config_col if choosing else None)
    else:
        # The full detector as configured: a simulation benchmark records
        # what each event holds for its first run only, and its other
        # configurations take a detector out, which is not what this is for.
        config = configs[0]
    # What was chosen among another configuration's costs and outputs is
    # checked against this one's before it is drawn.
    previous = st.session_state.get(_SHOWN_KEY)
    st.session_state[_SHOWN_KEY] = config
    changed = reco and previous is not None and previous != config
    if changed:
        # A new figure: a selection on the old one names other squares.
        st.session_state[f"{_MATRIX_KEY}_nonce"] = (
            st.session_state.get(f"{_MATRIX_KEY}_nonce", 0) + 1
        )
    elif pending:
        st.session_state[_MEASURE_KEY], st.session_state[_EXPRESSION_KEY] = pending

    try:
        table = (
            event_table(source, outputs=files.get(config, ())) if reco
            else event_table(source, content=files.get(config, ()))
        )
    except ValueError as exc:
        st.warning(f"What **{config}**'s events hold cannot be tabulated: {exc}")
        return None
    measures = (
        reconstruction_measures(event_data.get(config), costs.get(config)) if reco
        else simulation_measures(event_data.get(config), costs.get(config))
    )
    if table is None or not measures:
        st.info(f"Nothing of what **{config}**'s events hold, or what they cost, was recorded.")
        return None
    metrics = list(dict.fromkeys(m.metric for m in measures))
    notes = [] if config == BASELINE_LABEL or choosing else [f"Configuration **{config}**."]
    with metric_col:
        asked = st.session_state.get(_METRIC_KEY) or st.query_params.get(_METRIC_KEY)
        drop_stale_selection(_METRIC_KEY, metrics)
        metric = st.selectbox(
            "Metric", metrics, key=_METRIC_KEY, bind="query-params",
            help="What is measured of each event: its time, or, as recorded, the steps "
                 "simulation took, an algorithm's instructions or allocations.",
        )
    if changed and asked and asked != metric:
        notes.append(f"**{config}** recorded no {asked}: its {metric} is shown.")
    if source is not None and source.origin in _ORIGINS:
        notes.append(_ORIGINS[source.origin])
    notes += [f"`{name}` is left out: {why}." for name, why in table.left_out.items()]

    of_metric = [m for m in measures if m.metric == metric]
    names = [m.name for m in of_metric]
    # The cost the plot row will hold: the matrix, drawn first, rings it.
    chosen = st.session_state.get(_MEASURE_KEY) or st.query_params.get(_MEASURE_KEY)
    drop_stale_selection(_MEASURE_KEY, names)
    measure = of_metric[names.index(chosen) if chosen in names else 0]
    if changed and chosen and chosen not in names:
        notes.append(f"**{chosen}** has no {metric} in **{config}**: **{measure.name}** is shown.")
    default = _default_column(measure, table)
    if _EXPRESSION_KEY not in st.session_state and _EXPRESSION_KEY not in st.query_params:
        st.session_state[_EXPRESSION_KEY] = _expression_of(table, default)
    expression = st.session_state.get(_EXPRESSION_KEY) or st.query_params.get(_EXPRESSION_KEY, "")
    if changed and expression.strip():
        try:
            table.evaluate(expression)
        except ExpressionError as exc:
            before = expression
            expression = st.session_state[_EXPRESSION_KEY] = _expression_of(table, default)
            notes.append(f"`{before.strip()}` cannot be computed for **{config}**'s events "
                         f"({exc}): `{expression}` is plotted instead.")
    if notes:
        with note_col:
            st.caption(" ".join(notes))

    _matrix(of_metric, table, (measure.name, _column_of(table, expression)))
    _plot(of_metric, table, config if reco else None)
    return None


def _select_config(configs: list[str], results: pd.DataFrame | None, column) -> str:
    """The reconstruction configuration drawn: chosen among *configs* in
    *column*, or the only one without a dropdown when *column* is ``None``.
    One asked for, by a link or kept from another run, that is not among them
    is said so, with why, for as long as the one shown instead is."""
    asked = drop_stale_selection(_CONFIG_KEY, configs)
    if asked is None and _CONFIG_KEY not in st.session_state:
        asked = st.query_params.get(_CONFIG_KEY)
    if column is None:
        config = configs[0]
    else:
        with column:
            config = select_config(configs, _CONFIG_KEY)
    if asked is not None and asked not in configs:
        st.session_state[_MISSING_CONFIG_KEY] = (asked, config)
    missing, instead = st.session_state.get(_MISSING_CONFIG_KEY, (None, None))
    if missing is None or missing in configs or instead != config:
        st.session_state.pop(_MISSING_CONFIG_KEY, None)
    else:
        st.warning(f"**{missing}** {_why_not_offered(missing, results)}, so **{config}** "
                   "is shown instead.")
    return config


def _why_not_offered(label: str, results: pd.DataFrame | None) -> str:
    """Why configuration *label* has no per-event costs to draw."""
    if results is None or "label" not in results.columns:
        return "is not in this run"
    ran = results["label"].astype(str) == label
    if not ran.any():
        return "is not in this run"
    if failed_config_mask(results)[ran].any():
        return "failed in this run"
    return "recorded no per-event costs in this run"


def _has_content(outputs) -> bool:
    return any(
        c.entries is not None or c.energy_gev is not None for f in outputs for c in f.collections
    )


def _scale(measure: Measure) -> tuple[float, str]:
    return _UNITS.get(measure.unit, (1.0, measure.unit))


def _named(measure: Measure) -> str:
    """What *measure* is: ``Event time``, ``Refit instructions``."""
    if measure.name == WHOLE_EVENT:
        return f"Event {measure.metric}"
    return f"{measure.name} {measure.metric}"


def _axis_title(measure: Measure) -> str:
    _, unit = _scale(measure)
    title = _named(measure)
    return f"{title} ({unit})" if unit else title


def _per_event(measure: Measure) -> str:
    """*measure*'s mean per event as a reader takes it in."""
    return _written(float(measure.values.mean()), measure)


def _written(value: float, measure: Measure) -> str:
    """*value*, one of *measure*'s, as a reader takes it in."""
    scale, unit = _scale(measure)
    value *= scale
    if unit == "ms" and value >= 1e3:
        return f"{value / 1e3:.3g} s"
    return f"{value:.3g} {unit}".strip() if unit else f"{value:.3g}"


def _expression_of(table: EventTable, column: str) -> str:
    """How *column* is written: a canned quantity's expression, else its name."""
    return table.columns[column].expression or column


def _column_of(table: EventTable, expression: str) -> str | None:
    """The column *expression* writes out, by its name or its expression."""
    text = expression.strip()
    if text in table.columns:
        return text
    return next((n for n, c in table.columns.items() if c.expression == text), None)


def _matrix(measures: list[Measure], table: EventTable, selected: tuple[str, str | None]) -> None:
    """The costliest of *measures* against the quantities most associated
    with them; a click plots the pair."""
    rho, n, similar = association_matrix(
        measures, table, rows=_MATRIX_ROWS, columns=_MATRIX_COLUMNS,
    )
    if rho.empty:
        return
    by_name = {m.name: m for m in measures}
    fig = plot_association_matrix(
        rho, n,
        row_labels={r: f"{r} · {_per_event(by_name[r])}" for r in rho.index},
        column_labels={c: c for c in rho.columns},
        column_titles={c: _column_title(table, c, similar.get(c, ())) for c in rho.columns},
        selected=selected if selected[1] is not None else None,
    )
    if fig is None:
        return
    nonce = st.session_state.setdefault(f"{_MATRIX_KEY}_nonce", 0)
    event = st.plotly_chart(
        fig, key=f"{_MATRIX_KEY}_{nonce}", on_select="rerun", selection_mode="points",
        width="stretch", config={"displayModeBar": False},
    )
    st.caption("Rank correlation of each cost with each quantity. Click a square to plot it.",
               help=_MATRIX_HELP)
    for point in (getattr(event, "selection", None) or {}).get("points", []):
        if point.get("curve_number") == 0 and point.get("point_index") is not None:
            cost, column = fig.data[0].customdata[point["point_index"]][:2]
            st.session_state[_PENDING_KEY] = (cost, _expression_of(table, column))
            st.session_state[f"{_MATRIX_KEY}_nonce"] = nonce + 1
            st.rerun()


def _column_title(table: EventTable, column: str, alike) -> str:
    """What a matrix column is, and the quantities folded into it."""
    title = table.columns[column].label
    return f"{title}<br>Moves with it: {', '.join(alike)}" if alike else title


def _default_column(measure: Measure, table: EventTable) -> str:
    """What the box starts with: the primary's |cos θ| of a one-primary
    sample, else the quantity most associated with *measure*."""
    if "primary_abs_cos_theta" in table.columns:
        return "primary_abs_cos_theta"
    ranked = associations(measure, table).index
    return ranked[0] if len(ranked) else next(iter(table.columns))


def _try() -> None:
    """Write the example tried into the box."""
    if example := st.session_state.get(_EXAMPLE_KEY):
        st.session_state[_EXPRESSION_KEY] = example
        st.session_state[_EXAMPLE_KEY] = None


def _examples(table: EventTable) -> dict[str, str]:
    """Expressions that work for *table*, each with what it gives."""
    examples = {}
    if table.has_truth:
        examples |= {
            "count(final & charged & (pt > 1))": "Charged particles above 1 GeV",
            "mean(abs_cos_theta, final & charged)": "How far forward the event goes",
            "sum(E, final & (pdg == 22)) / visible_E": "The photons' share of the energy",
            "max(vr, final & charged)": "The farthest a charged particle is born from the axis",
        }
    counts = [n for n, c in table.columns.items() if c.kind != "truth" and n.startswith("n_")]
    if len(counts) >= 2:
        examples[f"{counts[0]} + {counts[1]}"] = "Two collections' elements together"
    return examples


def _plot(measures: list[Measure], table: EventTable, config: str | None = None) -> None:
    """The plot row (the cost, the expression, the log y axis), the
    reference, the scatter and the events above its trend; with *config*, a
    reconstruction configuration, also one event of the scatter on its own."""
    names = [m.name for m in measures]
    of_col, x_col, log_col = st.columns(
        [3, 8, 1.4], gap="small", vertical_alignment="bottom",
    )
    with of_col:
        name = st.selectbox(
            "Of", names, key=_MEASURE_KEY, bind="query-params",
            help="The cost plotted: the whole event, a detector or an algorithm, "
                 "the costliest first.",
        )
    with x_col:
        text = st.text_input(
            "Against", key=_EXPRESSION_KEY, bind="query-params", max_chars=MAX_LENGTH,
            placeholder="count(final & charged & (pt > 1))",
        )
    with log_col:
        log_y = st.toggle("Log y", value=False, key=_LOG_KEY, bind="query-params")
    st.caption(_EXPRESSION_LINE)
    _reference(table)

    measure = measures[names.index(name)]
    if why := align(measure, table):
        st.warning(f"The events of **{measure.name}** cannot be matched to what they hold: {why}.")
        return
    if not text.strip():
        st.info("Type an expression, or try an example under What can I write?")
        return
    try:
        x = table.evaluate(text)
    except ExpressionError as exc:
        st.error(f"The expression cannot be computed: {exc}.")
        return
    column = _column_of(table, text)
    x_label = table.columns[column].label if column else text.strip()

    scale, _ = _scale(measure)
    pairs = paired(measure, x)
    pairs["y"] *= scale
    line = trend(pairs["x"], pairs["y"])
    above = above_trend(pairs, line) if line is not None else None
    if pairs.empty:
        st.info("No event has both values.")
        return
    # Filled last, once the event shown on its own, which it marks, is known.
    chart = st.container()
    notes = []
    if missing := len(measure.values) - len(pairs):
        notes.append(f"{missing} of {len(measure.values)} events have no value of {x_label}.")
    if line is None:
        notes.append("Too few distinct events for a trend line.")
    if notes:
        st.caption(" ".join(notes))
    if above is not None and not above.empty:
        _above_table(above, x_label, _axis_title(measure))
    event = None if config is None else _event_detail(measures, measure, table, x, x_label, config)
    fig = plot_event_content(
        pairs, x_label=x_label, y_label=_axis_title(measure),
        line=line, above=above, log_y=log_y, selected=event,
    )
    with chart:
        if config is None:
            st.plotly_chart(fig, width="stretch", key=_CHART_KEY)
        else:
            # A point carries its event's number, whatever its place.
            clickable_chart(fig, _CHART_KEY, config, _SECTION, on_click=_choose_event,
                            width="stretch")


def _choose_event(event: str) -> None:
    st.session_state[_EVENT_KEY] = int(event)


_EVENT_HELP = (
    "One event of the plot, chosen by a click on its point or by its number: the "
    "job's own count of the events it processed, the warm-up 0 left out, which is "
    "the entry of its input the job read; the input's EventHeader may number it "
    "otherwise. Each bar is an algorithm's own cost in this event, of the metric "
    "drawn: for time, its own time, without the algorithms it called, so the bars "
    "are no profile of the event and need not add up to its time. The tick across "
    "a bar is that algorithm's median over the events it ran in. Click a bar to "
    "open the algorithm in Components."
)


def _event_detail(
    measures: list[Measure], measure: Measure, table: EventTable, x: pd.Series, x_label: str,
    config: str,
) -> int | None:
    """One event of *config*'s plot of *measure* against *x*: which entry of
    the input it is, its values plotted and, of the metric of *measures*,
    each algorithm's cost in it, a click on one opening it in Components.
    The event shown, or ``None`` before one is chosen."""
    events = sorted(e for e in measure.profiled if e != WARM_UP_EVENT)
    drop_stale_selection(_EVENT_KEY, events)
    st.markdown("##### One event", help=_EVENT_HELP)
    pick_col, about_col = st.columns([1, 4], vertical_alignment="bottom")
    with pick_col:
        event = st.selectbox(
            "Event", events, index=None, key=_EVENT_KEY, bind="query-params",
            placeholder="Click a point",
        )
    with about_col:
        if event is None:
            st.caption("Click a point of the plot, or choose an event by its number, to "
                       "see what each algorithm cost in it.")
            return None
        st.caption(_identity(event, config, table, measures, measure, x, x_label))

    metric = measure.metric
    algorithms = [m for m in measures if m.name != WHOLE_EVENT and align(m, table) is None]
    if not algorithms:
        st.caption(
            f"No algorithm of **{config}** was audited, so what each cost in it is not known."
            if all(m.name == WHOLE_EVENT for m in measures) else
            f"The events of **{config}**'s algorithms cannot be matched to what they hold."
        )
        return event
    scale, unit = _scale(algorithms[0])
    costs = pd.Series({m.name: m.values.get(event, float("nan")) for m in algorithms}) * scale
    typical = pd.Series({m.name: float(m.values.median()) for m in algorithms}) * scale
    title = f"{metric[:1].upper()}{metric[1:]} in event {event}" + (f" ({unit})" if unit else "")
    fig = plot_event_costs(costs, typical, title=title, highlight=measure.name)
    if fig is None:
        st.caption(f"No algorithm of **{config}** has a {metric} in event {event}.")
        return event
    clickable_chart(fig, _EVENT_CHART_KEY, config, _SECTION, width="stretch",
                    config={"displayModeBar": False})
    if absent := int(costs.isna().sum()):
        st.caption(f"{absent} of {len(costs)} algorithms have no {metric} in event {event}: "
                   "they did not run in it, or recorded none.")
    return event


def _identity(
    event: int, config: str, table: EventTable, measures: list[Measure], measure: Measure,
    x: pd.Series, x_label: str,
) -> str:
    """Which event *event* is, and its values plotted."""
    text = f"Event **{event}** of **{config}**"
    if (source := table.source) is not None:
        text += f", entry {source.first_entry + event} of `{source.file}`"
        numbers, runs = source.event_numbers, source.run_numbers
        if numbers is not None and numbers[event] is not None:
            run = runs[event] if runs is not None else None
            text += f" (EventHeader event {numbers[event]}" + (
                f", run {run})" if run is not None else ")")
    # The event's own time beside an algorithm's, from the same job.
    whole = next((m for m in measures if m.name == WHOLE_EVENT), None)
    shown = [whole] if (
        whole is not None and whole is not measure and align(whole, table) is None
    ) else []
    parts = [
        f"{_named(m)}: "
        + (_written(float(m.values[event]), m) if event in m.values.index else "none")
        for m in (*shown, measure)
    ]
    value = x.get(event)
    parts.append(f"{x_label}: " + (f"{value:.4g}" if pd.notna(value) else "no value"))
    return f"{text}. " + " · ".join(parts)


def _reference(table: EventTable) -> None:
    """Examples that work for *table*, and every name an expression can use."""
    with st.expander("What can I write?"):
        if examples := _examples(table):
            st.pills(
                "Try an example", list(examples), key=_EXAMPLE_KEY, on_change=_try,
                format_func=examples.get,
            )
        names, fields, grammar = st.tabs([
            f"Quantities of this run ({len(table.columns)})",
            f"Particle fields ({len(PARTICLE_FIELDS)})" if table.has_truth else "Particle fields",
            "Reductions and operators",
        ])
        with names:
            st.caption("One number per event: use them as they are, outside count(...), sum(...).")
            frame = table.frame
            st.dataframe(
                pd.DataFrame({
                    "Name": list(table.columns),
                    "What it is": [c.label for c in table.columns.values()],
                    "Least": [frame[n].min() for n in table.columns],
                    "Most": [frame[n].max() for n in table.columns],
                    "From": [_KINDS[c.kind] for c in table.columns.values()],
                    "Written as": [c.expression or "" for c in table.columns.values()],
                }),
                hide_index=True, width="stretch", height=320,
                column_config={
                    "Least": st.column_config.NumberColumn("Least", format="%.4g", width="small"),
                    "Most": st.column_config.NumberColumn("Most", format="%.4g", width="small"),
                },
            )
        with fields:
            if not table.has_truth:
                st.info("This run recorded no generator particles, so none can be folded.")
            else:
                st.caption(
                    "One value per generator particle: use them inside count(...), "
                    "sum(...), mean(...), max(...) or min(...), which fold an event's "
                    "particles into one number."
                )
                st.dataframe(
                    pd.DataFrame(
                        [(name, unit, what) for name, (unit, what) in PARTICLE_FIELDS.items()],
                        columns=["Name", "Unit", "What it is"],
                    ),
                    hide_index=True, width="stretch",
                )
        with grammar:
            st.dataframe(
                pd.DataFrame(_GRAMMAR, columns=["Write", "What it gives"]),
                hide_index=True, width="stretch",
            )
            st.caption(
                f"At most {MAX_LENGTH} characters. An expression is parsed and computed "
                "piece by piece, never run as Python."
            )


def _above_table(above: pd.DataFrame, x_label: str, y_label: str) -> None:
    with st.expander(f"{len(above)} event{'s' if len(above) != 1 else ''} above the trend"):
        st.caption(
            "More than three robust standard deviations above the trend line: above "
            "that line, which is not to say expensive for all they hold. The ratio is "
            "left out where the line is near zero."
        )
        st.dataframe(
            above.reset_index().rename(columns={
                "event": "Event", "x": x_label, "y": y_label, "expected": "Trend",
                "residual": "Above by", "ratio": "× trend",
            }),
            hide_index=True, width="stretch",
            column_config={
                "Event": st.column_config.NumberColumn("Event", format="%d"),
                x_label: st.column_config.NumberColumn(x_label, format="%.4g"),
                y_label: st.column_config.NumberColumn(y_label, format="%.4g"),
                "Trend": st.column_config.NumberColumn("Trend", format="%.4g"),
                "Above by": st.column_config.NumberColumn("Above by", format="%.4g"),
                "× trend": st.column_config.NumberColumn("× trend", format="%.2f"),
            },
        )
