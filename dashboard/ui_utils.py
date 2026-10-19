"""Shared UI utilities for dashboard tabs.

All palette/style constants and reusable Plotly helpers live here so each tab
imports from one place rather than duplicating the same definitions.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import plotly.colors as _pc
import streamlit as st
from plotly.colors import qualitative as _ql

from k4bench.analysis.plots import BinCountOptions, event_bin_options
from k4bench.analysis.plots._theme import PALETTE
from k4bench.analysis.plots._utils import _default_baseline, bottom_legend
from stats import select_top_n_by_ratio


# ── Data-validation helpers ────────────────────────────────────────────────────

def is_valid_df(df: "pd.DataFrame | None") -> bool:
    """Return *True* iff *df* is a non-``None``, non-empty :class:`~pandas.DataFrame`."""
    return df is not None and not df.empty


#: :func:`reset_widget_on_scope`'s default: reset a widget by dropping its value.
_DROP = object()


def reset_widget_on_scope(
    key: str, scope: object, *, reset_unscoped: bool = False,
    query_param: str | None = None, default: object = _DROP,
) -> None:
    """Drop a keyed widget's value when its context-dependent default changes.

    Streamlit handles values that disappear from a widget's options, but it
    retains a value that is still valid even when it came from another report,
    release range, or automatic palette size. Call this before creating the
    widget. The first render preserves existing state by default so query-
    seeded/deep-linked values remain authoritative; ``reset_unscoped`` is for
    purely automatic controls such as palette sizing.

    *query_param* drops that ``?param=`` alongside the stored value. A widget
    that writes its selection back to the URL needs it: clearing session state
    alone leaves the old value in the query string, from where
    :func:`~ui_chrome.seed_query_param` seeds it straight back on the same run
    — so the reset would silently do nothing whenever the stale value is also
    valid in the new scope. The incoming deep link survives, since a first
    render (no scope recorded yet) never resets.

    A widget bound to the URL (``bind="query-params"``) is reset by setting it
    to its *default* instead: its dropped value would be seeded straight back
    from its parameter, which only its value can change.
    """
    scope_key = f"_{key}_scope"
    previous = st.session_state.get(scope_key)
    st.session_state[scope_key] = scope
    if (
        (previous is not None and previous != scope)
        or (previous is None and reset_unscoped and key in st.session_state)
    ):
        if default is _DROP:
            st.session_state.pop(key, None)
        else:
            st.session_state[key] = default
        if query_param is not None:
            st.query_params.pop(query_param, None)


# ── Colour helper ──────────────────────────────────────────────────────────────

def to_rgba(color: str, alpha: float) -> str:
    """Convert any Plotly colour string to ``rgba(…)`` with the given alpha."""
    color = color.strip()
    if color.startswith("rgba("):
        return color
    try:
        r, g, b = (
            _pc.hex_to_rgb(color)
            if color.startswith("#")
            else _pc.unlabel_rgb(color)
        )
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return color


# ── Style constants ────────────────────────────────────────────────────────────


# ── Matplotlib qualitative palettes (tab10 / tab20 / tab30 / tab40) ───────────
# tab20 = tab10 hues reordered so all 10 dark shades come first, then the 10
# lighter companions → ≤10 items look identical to plain "Matplotlib".
# tab30 adds 10 hand-picked distinct colours from matplotlib's tab20b map.
# tab40 extends further with 10 from tab20c.
# "Matplotlib (auto)" picks the smallest variant that covers n without cycling.
_TAB20_DARK  = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
_TAB20_LIGHT = ["#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
                "#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5"]
_TAB20  = _TAB20_DARK + _TAB20_LIGHT
# One medium-dark representative per hue group in tab20b
_TAB20B_10 = ["#5254a3","#8ca252","#bd9e39","#ad494a","#a55194",
              "#6b6ecf","#b5cf6b","#e7ba52","#d6616b","#ce6dbd"]
# One medium representative per hue group in tab20c
_TAB20C_10 = ["#3182bd","#e6550d","#31a354","#756bb1","#636363",
              "#6baed6","#fd8d3c","#74c476","#9e9ac8","#969696"]
_TAB30 = _TAB20 + _TAB20B_10
_TAB40 = _TAB30 + _TAB20C_10

PALETTES: dict[str, list[str]] = {
    "Matplotlib":       PALETTE,
    "Matplotlib tab20": _TAB20,
    "Matplotlib tab30": _TAB30,
    "Matplotlib tab40": _TAB40,
    "Plotly":           _ql.Plotly,
    "D3":               _ql.D3,
    "G10":              _ql.G10,
    "Dark24":           _ql.Dark24,
    "Light24":          _ql.Light24,
    "Alphabet":         _ql.Alphabet,
    "Safe":             _ql.Safe,
    "Bold":             _ql.Bold,
}
PALETTE_NAMES = list(PALETTES.keys())


def auto_palette_index(n: int) -> int:
    """Return the *PALETTES* index for the smallest Matplotlib tab-N that fits *n*
    items without colour cycling.  Used as the ``index`` default for palette
    selectboxes so the dropdown already shows the right entry on first render.
    The user can always switch back to plain "Matplotlib" (10-colour cycling).
    """
    if n <= len(PALETTE):
        name = "Matplotlib"
    elif n <= len(_TAB20):
        name = "Matplotlib tab20"
    elif n <= len(_TAB30):
        name = "Matplotlib tab30"
    else:
        name = "Matplotlib tab40"
    try:
        return PALETTE_NAMES.index(name)
    except ValueError:
        return 0


# ── Configuration selector ─────────────────────────────────────────────────────
# Shared by the Event Timing and Event Memory tabs so their "which runs get
# plotted" behaviour cannot drift apart.

#: How many configurations (including the baseline) the selector pre-checks.
_DEFAULT_N_CONFIGS = 2


def baseline_selector_control(key_prefix: str, current_labels: list[str]) -> str:
    """Render the "Baseline" selectbox and return the chosen label.

    Defaults to :func:`~k4bench.analysis.plots._utils._default_baseline`
    (alphabetically first — the same fallback the plotting layer itself uses
    when no explicit baseline is given), but any available run can be picked
    instead, for full control over what the ratio panel and Statistics table
    are measured against.
    """
    default_label = _default_baseline(current_labels)
    key = f"{key_prefix}_baseline"
    # *current_labels* is whatever the run currently offers. A configuration
    # appearing or disappearing there must not silently change a still-valid
    # reference and therefore reinterpret every ratio on the page. Only discard
    # the stored choice when that baseline itself is gone.
    if key in st.session_state and st.session_state[key] not in current_labels:
        st.session_state.pop(key, None)
    return st.selectbox(
        "Baseline",
        options=current_labels,
        index=current_labels.index(default_label),
        key=key,
        bind="query-params",
        help=(
            "The configuration used as the reference. The ratio panel and the "
            "Statistics table's ratio column measure every other configuration "
            "relative to this one."
        ),
    )


def config_selector_control(
    key_prefix: str,
    event_data: dict,
    current_labels: list[str],
    baseline_label: str,
    column: str,
    unit: str,
    compare_by_default: bool = True,
) -> list[str]:
    """Render the "Configurations" multiselect and return the labels to plot.

    Pre-checks the baseline plus the ``_DEFAULT_N_CONFIGS - 1`` runs with the
    largest ratio deviation from it (:func:`~stats.select_top_n_by_ratio`), or
    the baseline alone unless *compare_by_default*, but any of
    *current_labels* can be added or removed freely. The primary plot and
    Statistics table follow this selection so their colours and rows stay
    aligned; a table of every configuration in the run remains available in a
    collapsed expander below it.

    The baseline itself is not one of the selectable options: it is always
    included in what gets plotted, so the ratio panel it anchors can never be
    left without a reference by an incidental deselection. Changing the
    baseline (in :func:`baseline_selector_control`) resets this selection back
    to the fresh top-N-by-ratio default for the new reference, rather than
    keeping a manual pick that was made relative to the old one.
    """
    selectable = [lbl for lbl in current_labels if lbl != baseline_label]
    key = f"{key_prefix}_configs"
    default_n = min(_DEFAULT_N_CONFIGS, len(current_labels)) if compare_by_default else 1
    default_all = select_top_n_by_ratio(
        event_data, current_labels, column, unit, baseline_label, True, default_n,
    )
    default_extra = [lbl for lbl in default_all if lbl != baseline_label]
    # A baseline change deliberately refreshes the comparison default: the new
    # baseline moves out of the options and the old one moves in. A change in
    # which configurations the run offers does not reset a still-valid manual
    # pick; Streamlit drops values that disappear from ``options`` itself.
    reset_widget_on_scope(
        key, baseline_label, reset_unscoped=True, default=default_extra,
    )

    n_extra = max(default_n - 1, 0)
    noun = "configuration" if n_extra == 1 else "configurations"
    default_help = (
        f"defaults to the {n_extra} {noun} with the largest deviation from it"
        if compare_by_default else "none is added by default"
    )
    selected_extra = st.multiselect(
        "Configurations",
        options=selectable,
        default=default_extra,
        key=key,
        bind="query-params",
        help=(
            f"`{baseline_label}` (the baseline) is always plotted. Pick which "
            "other configurations to compare it against — "
            f"{default_help}. The "
            "primary Statistics table follows this selection; every "
            "configuration in this run remains available below it."
        ),
    )
    return [baseline_label, *selected_extra]


# ── Histogram binning ──────────────────────────────────────────────────────────
# The Event Timing and Event Memory tabs share these helpers verbatim, so the
# two views cannot drift apart: they differ only in *key_prefix* (widget state
# is per-tab, since a bin count for timing means nothing for memory) and the
# metric column.

#: Default bar opacity — low enough that overlapping fills stay out of the way
#: of the (always fully opaque) outlines, without the user having to reach for
#: the slider first.
_HIST_DEFAULT_ALPHA = 0.25


@st.cache_data(show_spinner=False)
def cached_event_bin_options(
    event_data: dict[str, pd.DataFrame],
    column: str,
    labels: tuple[str, ...],
    exclude_events: tuple[int, ...] = (0,),
) -> BinCountOptions:
    """Cache data-derived bin controls across appearance-only dashboard reruns."""
    return event_bin_options(
        event_data,
        column=column,
        labels=list(labels),
        exclude_events=list(exclude_events),
    )


def _validate_bin_count(
    key: str,
    auto_key: str,
    custom_key: str,
    warning_key: str,
    minimum: int,
    maximum: int,
) -> None:
    """Clamp an edited bin count, retain a warning, and track custom state."""
    raw = int(st.session_state[key])
    bins = min(max(raw, minimum), maximum)
    if raw < minimum:
        st.session_state[warning_key] = (
            f"{raw} is below the minimum of {minimum}; "
            f"the value was reset to {minimum}."
        )
    elif raw > maximum:
        st.session_state[warning_key] = (
            f"{raw} is above the maximum of {maximum} for the current data; "
            f"the value was reset to {maximum}."
        )
    else:
        st.session_state.pop(warning_key, None)
    st.session_state[key] = bins
    st.session_state[custom_key] = bins != int(st.session_state[auto_key])


def view_control_row(
    options: list[str], *, key: str,
) -> tuple[str, Any, Any]:
    """Render the dashboard's canonical tab-local view header.

    The view radio occupies the left; two deferred right-hand slots hold the
    run-quality scope and Display options respectively. Returning placeholders
    lets callers compute their data before deciding whether either control is
    needed without changing the row where it appears. The view is bound to
    ``?{key}=``, so a copied link reopens it.
    """
    view_col, options_col = st.columns(
        [5, 3], gap="medium", vertical_alignment="bottom",
    )
    with view_col:
        view = (
            st.radio("**View**", options=options, horizontal=True, key=key,
                     bind="query-params")
            if len(options) > 1
            else options[0]
        )
    with options_col:
        option_alignment = st.container(
            horizontal=True, horizontal_alignment="right",
            vertical_alignment="bottom", width="stretch",
        )
        actions = option_alignment.container(
            border=False, horizontal=True, vertical_alignment="bottom",
            width="content", gap="small",
        )
        reliability_slot = actions.container(width="content").empty()
        display_options_slot = actions.container(width="content").empty()
    return view, reliability_slot, display_options_slot


# ── Display options ────────────────────────────────────────────────────────────
# Every view whose figures can be restyled collects those knobs into one
# "Display options" popover. The page flow then carries only the controls that
# change *what* you are looking at — baseline, configuration, attribution,
# metric, view, report night — and the controls that change *how it is drawn*
# read identically everywhere instead of teaching a new layout per tab. The
# page-flow controls are bound to the URL (``bind="query-params"``), so a copied
# link reopens what was being looked at; the popover's are not, and a link
# draws it the default way.
#
# A control is declared once, as a :class:`DisplayControl` that carries its own
# default, and :func:`display_options` derives the "Reset to defaults" button
# from those declarations. A second, hand-maintained copy of the defaults would
# be free to drift from what the widgets actually do — and would have to be
# written out again for every view that adopts the popover.


@dataclass(frozen=True)
class DisplayControl:
    """One appearance control: where its state lives, and how to draw it.

    Attributes
    ----------
    name : str
        Key under which :func:`display_options` returns this control's value,
        so callers read results by name rather than by position and reordering
        a declaration list cannot silently rebind them.
    key : str
        The widget's ``session_state`` key.
    default : Any
        The value "Reset to defaults" writes back under *key*.
    render : callable
        Draws the widget inside the popover and returns its value. Anything the
        widget needs done to ``session_state`` first — a
        :func:`reset_widget_on_scope` call, seeding an automatic value —
        belongs in here rather than in the factory, so constructing a control
        stays free of side effects and the order a caller declares them in
        cannot matter.
    on_reset : callable or None
        Extra bookkeeping the reset click performs, for a control that keeps
        companion state beside its widget key.
    """

    name: str
    key: str
    default: Any
    render: Callable[[], Any]
    on_reset: Callable[[], None] | None = None


def _reset_display_controls(controls: tuple[DisplayControl, ...]) -> None:
    """Restore every control in *controls* to the default it declared."""
    for control in controls:
        st.session_state[control.key] = control.default
        if control.on_reset is not None:
            control.on_reset()


def display_options(
    *controls: DisplayControl,
    key_prefix: str,
    slot=None,
) -> dict[str, Any]:
    """Render the shared "Display options" popover; return ``{name: value}``.

    The controls are stacked vertically in the order they are declared, so their
    labels do not collapse at narrower desktop widths, with the reset button
    last.

    *slot* is where the trigger goes: a placeholder (:func:`streamlit.empty`) or
    a column reserved on the view's control row. Several views can only size
    their palette once the figure's series are known, which is well after that
    row has been laid out — reserving the slot first keeps the trigger on the
    row regardless. The trigger is wrapped in a right-aligned horizontal
    container so it sits at the end of the row in every view. Without a *slot*
    it renders wherever the cursor is.
    """
    names = [control.name for control in controls]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            "Display control names must be unique: " + ", ".join(duplicate_names)
        )
    keys = [control.key for control in controls]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError(
            "Display control widget keys must be unique: " + ", ".join(duplicate_keys)
        )

    placement = (
        slot.container(horizontal=True, horizontal_alignment="right")
        if slot is not None
        else nullcontext()
    )
    with placement, st.popover("Display options", icon="👁️", width="content"):
        values = {control.name: control.render() for control in controls}
        st.button(
            "Reset to defaults",
            key=f"{key_prefix}_reset",
            width="stretch",
            on_click=_reset_display_controls,
            args=(controls,),
        )
    return values


def palette_control(key: str, n_items: int | None = None) -> DisplayControl:
    """Declare the qualitative "Colour palette" selectbox.

    *n_items* is how many series the figure will colour: the default is the
    smallest Matplotlib tab-N that covers them without cycling
    (:func:`auto_palette_index`), and that automatic choice is re-applied
    whenever the series count crosses a palette boundary.

    Pass ``None`` when the view has no data to size the palette from — an empty
    configuration filter, a run missing the requested attribution. The widget is
    still registered, so a stored selection survives the empty render rather
    than being garbage-collected as stale, but **no scope is recorded**: an
    empty render would record a palette size of 0, which the next render with
    data reads as a size change and answers by discarding the palette the user
    picked.
    """
    index = 0 if n_items is None else auto_palette_index(n_items)

    def render() -> str:
        if n_items is not None:
            reset_widget_on_scope(key, index, reset_unscoped=True)
        return st.selectbox(
            "Colour palette", options=PALETTE_NAMES, index=index, key=key,
        )

    return DisplayControl(
        name="palette", key=key, default=PALETTE_NAMES[index], render=render,
    )


#: Style-cycling modes, ordered by how much visual separation each adds.
_STYLE_CYCLING_OPTIONS = [
    "Colour only", "Colour + Dash", "Colour + Marker", "Colour + Dash + Marker",
]

_STYLE_CYCLING_HELP = (
    "When the number of series exceeds the palette size, additional visual "
    "cues are layered on top of colour — dash pattern and/or marker shape — so "
    "every line stays distinguishable even with 20+ of them."
)


def style_cycling_control(key: str) -> DisplayControl:
    """Declare the "Style cycling" selectbox for multi-line trend figures."""

    def render() -> str:
        return st.selectbox(
            "Style cycling",
            options=_STYLE_CYCLING_OPTIONS,
            index=0,
            key=key,
            help=_STYLE_CYCLING_HELP,
        )

    return DisplayControl(
        name="style", key=key, default=_STYLE_CYCLING_OPTIONS[0], render=render,
    )


def style_cycling_flags(style_cycling: str) -> tuple[bool, bool]:
    """Split a "Style cycling" choice into ``(use_dash, use_marker)``."""
    return (
        style_cycling in ("Colour + Dash", "Colour + Dash + Marker"),
        style_cycling in ("Colour + Marker", "Colour + Dash + Marker"),
    )


def opacity_control(
    key: str,
    *,
    label: str = "Opacity",
    default: float = 0.85,
    minimum: float = 0.1,
    help: str | None = None,
) -> DisplayControl:
    """Declare an opacity slider.

    Bounds and default are per-view rather than shared: a histogram's fills
    start far more transparent than a trend line, and a trend line has a floor
    below which it stops being traceable at all.
    """

    def render() -> float:
        return st.slider(
            label,
            min_value=minimum,
            max_value=1.0,
            value=default,
            step=0.05,
            key=key,
            help=help,
        )

    return DisplayControl(name="alpha", key=key, default=default, render=render)


def smooth_lines_control(key: str) -> DisplayControl:
    """Declare the "Smooth lines" toggle (spline instead of straight segments)."""

    def render() -> bool:
        return st.toggle(
            "Smooth lines",
            value=False,
            key=key,
            help=(
                "Draw each series as a spline through its points. Purely "
                "cosmetic — the measured points themselves do not move."
            ),
        )

    return DisplayControl(name="smooth", key=key, default=False, render=render)


#: How many detector regions the current-run region timing chart ranks.
TOP_N_DEFAULT = 8


def top_n_control(
    key: str, noun: str = "detectors", ranked: str = "detector regions",
    default: int = TOP_N_DEFAULT,
) -> DisplayControl:
    """Declare the "Top N detectors" slider — a plot-density knob like *bins*.
    *noun* names what is counted, *ranked* what the chart ranks."""

    def render() -> int:
        return st.slider(
            f"Top N {noun}",
            min_value=3,
            max_value=15,
            value=default,
            key=key,
            help=(
                f"How many of the most expensive {ranked} the chart "
                "ranks; the remainder are left out of the figure."
            ),
        )

    return DisplayControl(name="top_n", key=key, default=default, render=render)


def _bin_count_control(key: str, options: BinCountOptions) -> DisplayControl:
    """Declare the bin-count field, seeded automatically until the user edits it.

    *options.automatic* is the count the plot would have picked on its own.
    The allowed maximum is derived from the current pooled sample rather than
    being a separate hard-coded UI policy.

    Until the user changes the field, its value follows NumPy's automatic count
    whenever the displayed configurations change. Once edited, the chosen count
    is preserved across those changes. Entering the current automatic count
    returns the field to auto-following behaviour, without adding a separate
    mode selector to the interface. That behaviour is tracked in companion
    ``session_state`` entries beside the widget key, which is why this is the
    one control with an *on_reset* hook: writing the default count back without
    also clearing the "custom" flag would leave the field pinned to a count the
    user is no longer choosing.
    """
    bins_default = options.automatic
    auto_key = f"_{key}_auto"
    custom_key = f"_{key}_custom"
    warning_key = f"_{key}_warning"

    def render() -> int:
        is_custom = bool(st.session_state.get(custom_key, False))
        st.session_state[auto_key] = bins_default
        if key not in st.session_state or not is_custom:
            st.session_state[key] = bins_default
            st.session_state.pop(warning_key, None)
        elif not options.minimum <= int(st.session_state[key]) <= options.maximum:
            _validate_bin_count(
                key, auto_key, custom_key, warning_key,
                options.minimum, options.maximum,
            )

        bins = st.number_input(
            "Bins",
            step=1,
            key=key,
            on_change=_validate_bin_count,
            args=(
                key, auto_key, custom_key, warning_key,
                options.minimum, options.maximum,
            ),
            help=(
                f"Number of histogram bins ({options.minimum}-{options.maximum} "
                "for the current data), "
                "shared by every configuration in the plot and by the ratio panel "
                "below it. Starts at the automatically determined count for the "
                f"current selection ({bins_default}); an edited value is preserved."
            ),
        )
        if warning := st.session_state.get(warning_key):
            st.warning(warning, icon="⚠️")
        return int(bins)

    def on_reset() -> None:
        st.session_state[auto_key] = bins_default
        st.session_state[custom_key] = False
        st.session_state.pop(warning_key, None)

    return DisplayControl(
        name="bins", key=key, default=bins_default,
        render=render, on_reset=on_reset,
    )


def _error_bars_control(key: str) -> DisplayControl:
    """Declare the "Histogram error bars" toggle."""

    def render() -> bool:
        return st.toggle(
            "Histogram error bars",
            value=False,
            key=key,
            help="Poisson √N error bars on the bin contents.",
        )

    return DisplayControl(name="errors", key=key, default=False, render=render)


def _mean_lines_control(key: str) -> DisplayControl:
    """Declare the "Histogram mean lines" toggle."""

    def render() -> bool:
        return st.toggle(
            "Histogram mean lines",
            value=True,
            key=key,
            help="Dashed vertical line at each configuration's mean.",
        )

    return DisplayControl(
        name="mean_lines", key=key, default=True, render=render,
    )


def histogram_display_controls(
    key_prefix: str,
    bin_options: BinCountOptions,
    n_configs: int,
    slot=None,
) -> tuple[int, str, float, bool, bool]:
    """Render the event-histogram Display options and return their values.

    Shared by the Event Timing and Event Memory current-run views so the two
    cannot drift apart; they differ only in *key_prefix*, since a bin count for
    timing means nothing for memory.
    """
    values = display_options(
        _bin_count_control(f"{key_prefix}_hist_bins", bin_options),
        opacity_control(
            f"{key_prefix}_hist_alpha",
            label="Bar opacity",
            default=_HIST_DEFAULT_ALPHA,
            minimum=0.0,
            help=(
                "Opacity of the filled bars. The step outline over each histogram "
                "stays fully opaque, so turning this down un-clutters overlapping "
                "configurations without hiding any of them — at 0 only the outlines "
                "remain."
            ),
        ),
        _error_bars_control(f"{key_prefix}_hist_errors"),
        _mean_lines_control(f"{key_prefix}_hist_mean_lines"),
        palette_control(f"{key_prefix}_palette", n_configs),
        key_prefix=f"{key_prefix}_hist",
        slot=slot,
    )
    return (
        values["bins"], values["palette"], values["alpha"],
        values["errors"], values["mean_lines"],
    )


DASHES  = ["solid", "dash", "dot", "dashdot"]
SYMBOLS = ["circle", "square", "diamond", "cross",
            "triangle-up", "star", "pentagon", "hexagon"]


# ── Legend helpers ─────────────────────────────────────────────────────────────

#: Bottom margin (px) reserved for tick labels + horizontal legend.
#: Kept as a floor for the dynamic sizing in :func:`legend_below`.
_LEGEND_B_MARGIN = 160

#: Breathing room (px) between the x-tick labels and the legend, on top of the
#: per-chart ``tick_clearance``.  ~75 px ≈ 2 cm at 96 DPI.
_LEGEND_GAP = 75

def legend_below(
    n_entries: int,
    *,
    tick_clearance: int = 60,
    gap: int = _LEGEND_GAP,
    entry_width: int = 220,
    font_size: int = 13,
    ref_width: int = 1100,
    side_margin: int = 20,
) -> tuple[dict, int]:
    """Build a horizontal legend below the plot and the bottom margin that fits it.

    Returns ``(legend, b_margin)``: the caller uses ``b_margin`` for
    ``margin=dict(b=...)`` and adds it to the figure's ``height``, so the plot
    keeps its own height wherever the legend wraps as estimated.

    Why this is built this way
    --------------------------
    The legend is :func:`~k4bench.analysis.plots._utils.bottom_legend`, which
    Plotly lays out at whatever width the figure is drawn, so it fits at any
    width. ``b_margin`` is only the room the figure starts from: the ticks,
    *gap* and the legend's rows at ``ref_width``. Where the legend wraps onto
    more rows, automargin takes the extra room out of the gap and then the plot;
    where onto fewer, the gap is wider.

    Parameters
    ----------
    n_entries : number of legend items (e.g. configs or detectors plotted).
    tick_clearance : px the x-tick labels / axis titles need below the plot
        (use ~70 for rotated date ticks).
    gap : extra breathing room between the ticks and the legend, on top of
        ``tick_clearance`` (default :data:`_LEGEND_GAP` ≈ 2 cm).
    ref_width : plot width (px) the starting margin's row count is estimated
        at; it sets the gap only, never whether the legend fits.
    """
    usable = max(1, ref_width - 2 * side_margin)
    per_row = max(1, usable // entry_width)
    rows = max(1, math.ceil(max(1, n_entries) / per_row))
    row_h = font_size + 8
    legend_h = rows * row_h + 12
    # Offset from the plot's bottom edge to the legend's top edge.
    offset = tick_clearance + gap
    b_margin = max(_LEGEND_B_MARGIN, offset + legend_h)
    return bottom_legend(entry_width=entry_width, font_size=font_size), b_margin
