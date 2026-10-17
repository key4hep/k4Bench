"""How long a k4run job's phases take, and which components their time goes
to: in one run (:func:`plot_phase_component_pie`, :func:`plot_event_composition` for the
event loop event by event) and across nightly tags (:func:`plot_phase_history`)."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go

from ._theme import _OTHER_COLOR, _PALETTE, _TEMPLATE
from ._utils import bottom_legend

#: Phases in the order a job runs them, with their display names.
PHASE_NAMES = {
    "configure": "Configure",
    "initialize": "Initialize",
    "start": "Start",
    "execute": "Event loop",
    "stop": "Stop",
    "finalize": "Finalize",
}

def _seconds(value: float) -> str:
    return f"{value:.3g} s" if value < 100 else f"{value:,.0f} s"


#: Slices smaller than this share carry no percentage on the slice itself:
#: the legend and the hover name them.
_MIN_LABEL_SHARE = 0.04


def _total(value: float) -> str:
    """A phase's total, as the dashboard's phase cards write it."""
    return f"{value:.2f} s" if value < 100 else f"{value:,.0f} s"


def top_components(seconds: pd.Series, top_n: int) -> list[str]:
    """The *top_n* components of *seconds* that took longest, longest first:
    the slices :func:`plot_phase_component_pie` names."""
    seconds = seconds.dropna()
    return seconds[seconds > 0].sort_values(ascending=False).index[:top_n].astype(str).tolist()


def plot_phase_component_pie(
    seconds: pd.Series,
    *,
    top_n: int = 5,
    colors: Mapping[str, str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """A donut of one phase's time by component: the :func:`top_components`
    and the rest as one "Other (k)" slice, the phase's total in the middle.

    *seconds* is component → self seconds in the phase
    (:meth:`~k4bench.analysis.components.ComponentTiming.phase_component_seconds`),
    which add up to the phase. *colors* maps a component to its colour, so a
    component that several donuts show keeps one; the others take the
    palette in order.
    """
    seconds = seconds.dropna()
    seconds = seconds[seconds > 0].sort_values(ascending=False)
    total = float(seconds.sum())
    top = top_components(seconds, top_n)
    rest = seconds.drop(index=top)
    names = list(top)
    values = [float(seconds[name]) for name in top]
    palette = iter(_PALETTE * (len(top) // len(_PALETTE) + 1))
    slice_colors = [(colors or {}).get(name) or next(palette) for name in top]
    if len(rest):
        names.append(f"Other ({len(rest)})")
        values.append(float(rest.sum()))
        slice_colors.append(_OTHER_COLOR)
    shares = [v / total if total else 0.0 for v in values]

    fig = go.Figure(go.Pie(
        labels=names, values=values, hole=0.58, sort=False, direction="clockwise",
        marker=dict(colors=slice_colors, line=dict(color="white", width=2)),
        text=[f"{share:.0%}" if share >= _MIN_LABEL_SHARE else "" for share in shares],
        textinfo="text", textposition="inside", insidetextorientation="horizontal",
        customdata=[_seconds(v) for v in values],
        hovertemplate="<b>%{label}</b><br>%{customdata} · %{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{_total(total)}</b>", x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=18),
    )
    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text=title, x=0.5, xanchor="center") if title else None,
        height=440,
        margin=dict(l=10, r=10, t=50 if title else 10, b=10),
        legend=dict(orientation="v", x=0.5, xanchor="center", y=-0.05, yanchor="top"),
    )
    return fig


#: Phases shorter than this in every run of the window are not offered:
#: Gaudi's start and stop take milliseconds.
MIN_PHASE_S = 0.05

def plot_event_composition(
    times: pd.DataFrame,
    components: list[str],
    *,
    colors: Mapping[str, str] | None = None,
    sort: bool = False,
    warm_up: int | None = 0,
) -> go.Figure:
    """The event loop event by event: a column per event stacking each of
    *components*' self time in it, the rest as "Other".

    *times* is the job's ``execute["wall_s"]`` (events × components, ``NaN``
    where a component did not run). *colors* maps a component to its colour,
    as :func:`plot_phase_component_pie` takes it. *sort* orders the events by
    their time, longest first, instead of by event number. The *warm_up*
    event is drawn paler.
    """
    times = times.fillna(0.0)
    named = [c for c in components if c in times.columns]
    rest = [c for c in times.columns if c not in named]
    series = {c: times[c] for c in named}
    if rest and (times[rest].to_numpy() > 0).any():
        series[f"Other ({len(rest)})"] = times[rest].sum(axis=1)
    totals = times.sum(axis=1)
    order = totals.sort_values(ascending=False).index if sort else times.index
    labels = [str(event) for event in order]
    opacity = [0.45 if event == warm_up else 1.0 for event in order]

    palette = iter(_PALETTE * (len(named) // len(_PALETTE) + 1))
    fig = go.Figure()
    for name, seconds in series.items():
        color = (
            _OTHER_COLOR if name not in named
            else (colors or {}).get(name) or next(palette)
        )
        fig.add_trace(go.Bar(
            x=labels, y=seconds.reindex(order), name=name,
            marker=dict(color=color, opacity=opacity),
            hovertemplate=f"{name}: %{{y:.3g}} s<extra></extra>",
        ))
    fig.update_layout(
        template=_TEMPLATE,
        barmode="stack",
        bargap=0.15,
        height=460,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        legend=bottom_legend(),
    )
    fig.update_xaxes(
        type="category",
        title_text="Event, longest first" if sort else "Event number",
        showticklabels=not sort,
    )
    fig.update_yaxes(title_text="Self time (s)")
    return fig


#: Nightly tags up to which every column carries its total.
_MAX_TOTAL_LABELS = 20


#: The columns of :func:`phase_component_series`.
SERIES_COLUMNS = ["run_id", "run_date", "x_date", "k4h_release", "label", "seconds", "share_pct"]


def phase_component_series(rows: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """One phase's time per run as series to draw: the *top_n* components
    that took longest over the window (ranked by their median), and the rest
    summed as "Other (k)", in that order.

    *rows* are one config's and one phase's
    :func:`~k4bench.analysis.trend.build_phase_trend` rows: its components,
    or, for a phase timed only as a whole, its total with ``component`` set
    to the phase's name. One row per run and series, the series named in
    ``label``, with its ``seconds`` and its ``share_pct`` of the run's phase,
    the run's ``run_id``, ``run_date``, ``x_date`` and ``k4h_release`` kept.
    """
    if rows.empty:
        return pd.DataFrame(columns=SERIES_COLUMNS)
    run_keys = ["run_id", "run_date", "x_date", "k4h_release"]
    ranked = rows.groupby("component")["seconds"].median().sort_values(ascending=False)
    top = ranked.index[:top_n].tolist()
    rest = ranked.index[top_n:]
    labelled = rows.assign(label=rows["component"].where(
        rows["component"].isin(top), f"Other ({len(rest)})",
    ))
    series = labelled.groupby([*run_keys, "label"], dropna=False, as_index=False)["seconds"].sum()
    totals = series.groupby("run_id")["seconds"].transform("sum")
    series["share_pct"] = series["seconds"] / totals.where(totals > 0) * 100
    order = {name: i for i, name in enumerate([*top, f"Other ({len(rest)})"])}
    series = series.sort_values(
        ["x_date", "run_date"], kind="stable",
    ).sort_values("label", key=lambda names: names.map(order), kind="stable")
    return series[SERIES_COLUMNS].reset_index(drop=True)


def plot_phase_history(
    series: pd.DataFrame,
    *,
    palette: list[str] = _PALETTE,
    relative: bool = False,
) -> go.Figure | None:
    """One phase's time per nightly tag, a column per tag stacking
    :func:`phase_component_series`' series in their order, the last of them
    ("Other") grey: a column's height is the phase's duration, written above
    it. *relative* draws each series as its share of the column. A tag run
    more than once shows its latest run. ``None`` when there is nothing to
    draw.
    """
    if series.empty:
        return None
    # The series' order is the input's: longest first, "Other" last.
    names = list(dict.fromkeys(series["label"]))
    series = (
        series.dropna(subset=["run_date"]).sort_values("run_date")
        .drop_duplicates(["label", "x_date"], keep="last")
    )
    dates = sorted(pd.to_datetime(series["x_date"]).unique())
    table = series.pivot_table(
        index="x_date", columns="label", values="seconds", aggfunc="sum", fill_value=0.0,
    ).reindex(index=dates, columns=names, fill_value=0.0)
    totals = table.sum(axis=1)

    fig = go.Figure()
    for i, name in enumerate(names):
        seconds = table[name]
        share = (seconds / totals.where(totals > 0)).fillna(0.0)
        other = name.startswith("Other (")
        fig.add_trace(go.Bar(
            x=dates, y=share * 100 if relative else seconds, name=name,
            marker=dict(
                color=_OTHER_COLOR if other else palette[i % len(palette)],
                line=dict(color="white", width=1),
            ),
            customdata=list(zip(seconds, share)),
            hovertemplate=f"{name}: %{{customdata[0]:.3g}} s · %{{customdata[1]:.0%}}<extra></extra>",
        ))
    if not relative and len(dates) <= _MAX_TOTAL_LABELS:
        fig.add_trace(go.Scatter(
            x=dates, y=totals, mode="text", text=[_seconds(t) for t in totals],
            textposition="top center", showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        template=_TEMPLATE, barmode="stack", bargap=0.45, height=460,
        margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified",
        legend=bottom_legend(),
    )
    # The date axis the other trend views draw: one tick per nightly tag, and
    # a night without a run is a gap.
    fig.update_xaxes(
        type="date", tickmode="array", tickvals=dates,
        ticktext=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates], tickangle=-30,
        title_text="Key4hep Nightly Tag",
    )
    top_y = 100 if relative else float(totals.max()) * 1.12
    fig.update_yaxes(
        title_text="Share of the phase (%)" if relative else "Time (s)",
        range=[0, top_y or 1],
    )
    return fig
