"""Event Size: what each event of a configuration's output takes on disk,
after ROOT's compression, collection by collection.

Shared by the Simulation page (ddsim's output file) and the Reconstruction
page (the EDM4hep files the k4run job wrote). It reads the run's
``_event_size.json`` (:mod:`k4bench.analysis.event_size`), which the runners
measure right after each run. Current Run gives an output file's event size,
file size, compression and largest collection in four tiles, the event split
by collection and by type in two donuts, then every collection's compressed
and uncompressed bytes per event and its share of the event. Historical Trends draws the event size across the trend window, a
column per nightly tag stacking its largest collections, or a line per
collection.

Sizes do not depend on how loaded the host was, so the historical view offers
no run-quality filter: a run flagged unreliable for its timing wrote the same
bytes.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from data import cached_load_event_size
from k4bench.analysis.event_size import FRAME_PARAMETERS, KB, OutputFile, describe_compression
from k4bench.analysis.loader import judgeable_config_data
from k4bench.analysis.plots import plot_phase_component_pie, plot_phase_history
from k4bench.analysis.plots.heap import format_mb
from k4bench.analysis.plots.phases import phase_component_series
from k4bench.labels import BASELINE_LABEL
from sections import TREND_WINDOW_SCOPE, SectionScope
from tabs._historical_trends import render_historical_trends
from ui_chrome import drop_stale_selection
from ui_utils import (
    PALETTES,
    display_options,
    fit_legend_below,
    palette_control,
    top_n_control,
    view_control_row,
)

_KEY = "evt_size"
_VIEW_KEY = f"{_KEY}_view"
_HIST_KEY = f"{_KEY}_hist"

#: Sub-views, in dispatch order; the first is the one without history.
_VIEWS = ["Current Run", "Historical Trends"]

#: How the frame parameters are written on screen.
_FRAME_PARAMETERS_LABEL = "Frame parameters"

#: Collections the history names; the rest are one "Other" series.
_TOP_N = 8
_TOP_N_KEY = f"{_HIST_KEY}_topn"

_SCALES = {"kB per event": False, "Share of the event": True}

#: Rows the collections table shows before it scrolls.
_TABLE_ROWS = 20

#: Collections, or types, each donut names; the rest are one "Other" slice.
_PIE_TOP_N = 6

#: The donuts' height: two share the row, so each is as wide as half of it.
_PIE_HEIGHT = 620
_CHARTS = {"Bars": False, "Lines": True}


def _kb(value: float) -> str:
    """*value* kB as a reader takes it in: MB from 1024 kB."""
    return format_mb(value / KB) if value >= KB else f"{value:.3g} kB"


def _name(collection: str) -> str:
    return _FRAME_PARAMETERS_LABEL if collection == FRAME_PARAMETERS else collection


def _type(row: pd.Series) -> str:
    """A collection's type as the table writes it: ``edm4hep::MCParticle``,
    marked when the collection is a subset of another's objects."""
    name = row["type"].removesuffix("Collection") or "—"
    return f"{name} (subset)" if row["subset"] else name


def _configs(labels) -> list[str]:
    """*labels* in the order a selector offers them: the baseline first."""
    return sorted(labels, key=lambda label: (label != BASELINE_LABEL, label))


def _event_files(data_dir: str, results: pd.DataFrame | None) -> dict[str, list[OutputFile]]:
    """Each configuration that succeeded and wrote a podio file, with its
    podio files, largest first, naming any record that could not be read."""
    records, unreadable = cached_load_event_size(data_dir)
    for label, reason in unreadable.items():
        st.warning(f"The event sizes of **{label}** could not be read, so it is not offered: {reason}")
    records = judgeable_config_data(records, results) or {}
    files = {
        label: sorted((f for f in outputs if f.has_events), key=lambda f: -f.bytes)
        for label, outputs in records.items()
    }
    return {label: f for label, f in files.items() if f}


def _pick_file(names: list[str], key: str) -> str:
    """The output file chosen of *names*, the largest first; no selector for one."""
    if len(names) == 1:
        return names[0]
    drop_stale_selection(key, names)
    return st.selectbox(
        "Output file", names, key=key, bind="query-params",
        help="The job wrote several EDM4hep files: each holds its own events.",
    )


def _render_tiles(output: OutputFile, per_event: pd.DataFrame) -> None:
    """The file's event size, its size on disk, its compression and its
    largest collection."""
    compressed = per_event["compressed_kb"].sum()
    uncompressed = per_event["uncompressed_kb"].sum()
    largest = per_event.index[0]
    tiles = [
        (
            "Event size", _kb(compressed), "per event, compressed",
            "The bytes the file's events take on disk, every collection's branches "
            "and the frame parameters, divided by its events. The file's metadata "
            f"and run trees are not counted. Measured over this run's {output.events} "
            "events: a production file of many more events fills ROOT's baskets "
            "further, which can compress a little differently.",
        ),
        (
            "File size", format_mb(output.bytes / KB**2),
            f"{output.events:,} events, metadata included",
            "The file's size on disk, as the job wrote it.",
        ),
        (
            "Compression", f"{uncompressed / compressed:.2f}×" if compressed > 0 else "—",
            describe_compression(output.compression),
            "The events' uncompressed bytes over their compressed bytes, with the "
            "ROOT compression setting the job wrote the file with.",
        ),
        (
            "Largest collection", f"{per_event['share'].iloc[0]:.0%}",
            f"{_name(largest)}, {_kb(per_event['compressed_kb'].iloc[0])}",
            "The share of each event's bytes on disk the largest collection's "
            "branches take.",
        ),
    ]
    for column, (title, value, caption, help_text) in zip(st.columns(len(tiles)), tiles):
        with column, st.container(border=True, height="stretch"):
            st.metric(title, value, help=help_text)
            st.caption(caption)


def _render_donuts(per_event: pd.DataFrame) -> None:
    """The event's compressed bytes split by collection and by type."""
    names = pd.Index([_name(name) for name in per_event.index])
    kinds = per_event.apply(_type, axis=1).where(
        per_event.index != FRAME_PARAMETERS, _FRAME_PARAMETERS_LABEL,
    )
    splits = {
        "By collection": per_event["compressed_kb"].set_axis(names),
        "By type": per_event["compressed_kb"].groupby(kinds.to_numpy()).sum(),
    }
    for column, (title, parts) in zip(st.columns(len(splits), gap="medium"), splits.items()):
        fig = plot_phase_component_pie(parts, top_n=_PIE_TOP_N, title=title, unit="kB")
        fig.update_layout(height=_PIE_HEIGHT, margin=dict(l=0, r=0, t=40, b=0))
        with column:
            st.plotly_chart(
                fig, width="stretch", key=f"{_KEY}_{title.lower().replace(' ', '_')}_chart",
                config={"displayModeBar": False},
            )


def _render_table(per_event: pd.DataFrame) -> None:
    """Every collection, largest first, with its bytes per event."""
    table = pd.DataFrame({
        "Collection": [_name(name) for name in per_event.index],
        "Type": per_event.apply(_type, axis=1).to_list(),
        "Compressed": per_event["compressed_kb"].to_numpy(),
        "Share": per_event["share"].to_numpy() * 100,
        "Uncompressed": per_event["uncompressed_kb"].to_numpy(),
        "Compression": (
            per_event["uncompressed_kb"] / per_event["compressed_kb"].where(per_event["compressed_kb"] > 0)
        ).to_numpy(),
    })
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        # Up to :data:`_TABLE_ROWS` rows at once, the header's height included.
        height=35 * (min(len(table), _TABLE_ROWS) + 1) + 3,
        column_config={
            "Collection": st.column_config.TextColumn("Collection", width="medium"),
            "Type": st.column_config.TextColumn("Type", width="medium"),
            "Compressed": st.column_config.NumberColumn(
                "Compressed", format="%.2f kB",
                help="The bytes the collection's branches take on disk per event.",
            ),
            "Share": st.column_config.ProgressColumn(
                "Share", format="%.1f%%", min_value=0, max_value=100,
                help="Its share of the event's compressed bytes.",
            ),
            "Uncompressed": st.column_config.NumberColumn(
                "Uncompressed", format="%.2f kB",
                help="The bytes per event before ROOT's compression.",
            ),
            "Compression": st.column_config.NumberColumn(
                "Compression", format="%.2f×",
                help="Uncompressed over compressed bytes.",
            ),
        },
    )
    st.caption(
        "Per event of the file, in kB of 1024 bytes. A collection's bytes are those "
        "of all its branches: its data, its relations and vector members, each with "
        "its TKey headers. The frame parameters are podio's per-event parameters."
    )


def _render_current_run(data_dir: str, results: pd.DataFrame | None) -> None:
    files = _event_files(data_dir, results)
    if not files:
        st.info(
            "No event size in this run: no configuration that succeeded left an "
            "`_event_size.json` naming an EDM4hep output file. The runners record "
            "it from the runs after k4Bench started measuring their output files."
        )
        return
    config_col, file_col = st.columns([2, 3], vertical_alignment="bottom")
    with config_col:
        config = st.selectbox(
            "Configuration", _configs(files), key=f"{_KEY}_config", bind="query-params",
        )
    outputs = {f.name: f for f in files[config]}
    with file_col:
        output = outputs[_pick_file(list(outputs), f"{_KEY}_file")]
    per_event = output.per_event()
    _render_tiles(output, per_event)
    _render_donuts(per_event)
    _render_table(per_event)

    records, _ = cached_load_event_size(data_dir)
    others = [f for f in records.get(config, ()) if not f.has_events]
    if others:
        st.caption(
            "The job also wrote, without EDM4hep events: "
            + ", ".join(f"`{f.name}` ({format_mb(f.bytes / KB**2)})" for f in others)
        )


def _render_historical(
    trend: pd.DataFrame | None, display_options_slot, all_runs: bool = False,
) -> None:
    if trend is None or trend.empty:
        st.info(
            "No event sizes in the selected window. Widen the trend window in the "
            "sidebar: only runs from after k4Bench started measuring their output "
            "files have them."
        )
        return
    config_col, file_col, chart_col, scale_col = st.columns(
        [2, 3, 1.4, 2.2], vertical_alignment="bottom",
    )
    with config_col:
        config = st.selectbox(
            "Configuration", _configs(trend["label"].unique()), key=f"{_HIST_KEY}_config",
            bind="query-params",
        )
    rows = trend[trend["label"] == config]
    # The largest file the latest run wrote first, as Current Run offers them.
    sizes = rows[rows["collection"].isna()].sort_values("run_date").groupby("file")["file_mb"].last()
    with file_col:
        file = _pick_file(sizes.sort_values(ascending=False).index.tolist(), f"{_HIST_KEY}_file")
    with chart_col:
        lines = _CHARTS[st.segmented_control(
            "Chart", list(_CHARTS), default="Bars", key=f"{_HIST_KEY}_chart_type",
            bind="query-params",
        ) or "Bars"]
    with scale_col:
        relative = _SCALES[st.segmented_control(
            "Show", list(_SCALES), default="kB per event", key=f"{_HIST_KEY}_scale",
            bind="query-params",
        ) or "kB per event"]

    collections = rows[(rows["file"] == file) & rows["collection"].notna()].rename(
        columns={"collection": "component"},
    )
    # The Top N chooses the series before either chart draws its popover, in
    # which it then sits; Streamlit has already restored the slider's value.
    top_n = int(st.session_state.get(_TOP_N_KEY, _TOP_N))
    series = phase_component_series(
        collections, top_n, value="compressed_kb",
        apart={FRAME_PARAMETERS: _FRAME_PARAMETERS_LABEL}, all_runs=all_runs,
    )
    top_n_control_ = top_n_control(_TOP_N_KEY, "collections", "collections", default=_TOP_N)
    if lines:
        stat, title, unit = (
            ("share_pct", "Share of the event (%)", "%") if relative
            else ("compressed_kb", "Compressed (kB per event)", "kB")
        )
        render_historical_trends(
            series, list(dict.fromkeys(series["label"])), [(stat, title)],
            std_col="", n_col_candidates=[], unit=unit, error_sources={},
            key_prefix=_HIST_KEY, display_options_slot=display_options_slot,
            extra_controls=(top_n_control_,), all_runs=all_runs,
        )
        return
    display = display_options(
        top_n_control_,
        palette_control(f"{_HIST_KEY}_palette", top_n + 1),
        key_prefix=f"{_HIST_KEY}_display",
        slot=display_options_slot,
    )
    fig = plot_phase_history(
        series, palette=PALETTES[display["palette"]], relative=relative,
        value="compressed_kb", unit="kB", y_title="Event size (kB, compressed)", whole="event",
        all_runs=all_runs,
    )
    if fig is None:
        st.info(f"No event sizes for **{config}** in the selected window.")
        return
    fit_legend_below(fig, series["label"].nunique())
    st.plotly_chart(fig, width="stretch", key=f"{_HIST_KEY}_chart")


def render(
    data_dir: str,
    results: pd.DataFrame | None,
    load_trend: Callable[[], pd.DataFrame | None] | None = None,
    *,
    all_runs: bool = False,
) -> SectionScope | None:
    """The run in *data_dir*, its configurations' *results* deciding which
    succeeded, and, given *load_trend*, which loads the trend window's
    ``build_event_size_trend`` rows of configurations that succeeded, its
    history: every run of a release when *all_runs*."""
    views = _VIEWS if load_trend is not None else _VIEWS[:1]
    drop_stale_selection(_VIEW_KEY, views)
    view, _, display_options_slot = view_control_row(views, key=_VIEW_KEY)
    if view == "Historical Trends" and load_trend is not None:
        _render_historical(load_trend(), display_options_slot, all_runs)
        # The trends span the window's releases, not the sidebar's one.
        return TREND_WINDOW_SCOPE
    _render_current_run(data_dir, results)
    return None
