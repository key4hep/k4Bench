"""Stack Changes tab — what moved in Key4hep between two nightlies.

The package diff is cross-detector: a Key4hep release is one stack, sourced
identically by every detector benchmarked against it, so only the platform
scopes it. Which detector supplied the answer is an implementation detail —
the tab reads whichever run it finds first. The *reverse* view below the diff
(the regressions whose onset falls in the range) is scoped to the sidebar's
detector and sample like every other judged view, with a toggle to widen it
back to the whole platform.

The comparison is between **releases**, never run dates. The nightly build
lags: on a day with no new nightly the benchmark sources the newest one
available, so consecutive run dates routinely share one identical stack — most
of the run dates on EOS are not release dates at all. Comparing run dates would
manufacture changes that never happened.
"""

from __future__ import annotations

from urllib.parse import urlencode

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from k4bench.analysis.plots._theme import PALETTE, _TEMPLATE
from k4bench.provenance.diff import ADDED, CHANGED, REMOVED, diff_packages, unchanged_packages
from k4bench.regression.engine import Z_THRESHOLD
from k4bench.regression.models import MetricVerdict
from k4bench.regression.render import _pretty_sample, from_json
from k4bench.regression.report_builder import EVENT_METRICS, RUN_METRICS
from remote_cache import (
    _cached_fetch_report,
    _cached_fetch_stack_packages,
    _cached_list_detectors,
    _cached_list_report_dates,
    _cached_list_stacks,
)
from tabs import _blame
from tabs._regression_flags import FLAG_MARKS, add_severity_markers, flag_table
from ui_chrome import _drop_stale_selection, seed_query_param
from ui_utils import _METRIC_LABELS, _METRIC_UNITS, _legend_below, _to_rgba

#: Releases are stored as ``key4hep-{YYYY-MM-DD}`` directories; the tab talks
#: in the bare nightly tag the rest of the dashboard shows on its axes.
_PREFIX = "key4hep-"

#: This tab's section name and the query-param names its two pickers seed from
#: (see :func:`_seed`). Kept here, beside the widgets that read them, so a tab
#: that deep-links *into* this view (the Regressions blame note, via
#: :func:`deep_link`) shares one source of truth: rename a param and its builder
#: moves with it, instead of silently breaking a literal in another module.
_TAB_NAME = "Stack Changes"
PARAM_FROM = "from"
PARAM_TO = "to"


def deep_link(
    *, detector: str, platform: str, head_release: str,
    base_release: str | None = None, sample: str | None = None,
) -> str:
    """A relative query string that opens this tab seeded to a release range.

    ``base_release`` → ``head_release`` become the two pickers. Omitting
    *base_release* (an open-ended blame window) leaves the older end at the
    tab's own default for the user to choose.

    *detector* is carried for two reasons: the sidebar resolves the platform
    list *from the selected detector* (seeding the platform without a detector
    that offers it would be rejected), and the regressions-in-range view below
    the diff is scoped to the sidebar's detector/sample. *sample* completes
    that scope when the caller has one (a blame link from a specific verdict).
    """
    params = {
        "tab": _TAB_NAME, "detector": detector, "platform": platform,
        PARAM_TO: head_release,
    }
    if sample:
        params["sample"] = sample
    if base_release:
        params[PARAM_FROM] = base_release
    return "?" + urlencode(params)

#: Link colour carries meaning here: it marks the one *action* in a row. The
#: package and its two commits are identifiers, and a grid column has no
#: styling of its own, so linking them would draw the whole table blue and
#: leave the eye nothing to land on. The compare view spans both commits, so
#: nothing is lost by being the only link.
_STATUS_BADGE = {CHANGED: "🔄 changed", ADDED: "➕ added", REMOVED: "➖ removed"}

#: Why the dates offered here are not the dates benchmarks ran on. Lives in the
#: pickers' tooltips: it explains a gap in the list ("where is 2026-07-14?")
#: rather than the diff, so it is worth a look on demand and not a standing line
#: of prose above the answer.
_TAG_HELP = (
    "These are Key4hep release tags, not benchmark run dates: the nightly build "
    "does not publish every day, and a run then re-uses the newest release "
    "available."
)

_DOCS_URL = (
    "https://key4hep.github.io/k4Bench/user-guide/features/dashboard/#stack-changes-tab"
)


def _release(stack: str) -> str:
    return stack.removeprefix(_PREFIX)


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _seed(key: str, param: str, options: list[str], default: str) -> None:
    """Seed a picker from ``?param=`` if present, else from *default*.

    Both halves must go through ``session_state``: Streamlit rejects a widget
    that sets ``session_state[key]`` *and* passes ``index=`` in the same run
    (see :func:`ui_chrome.seed_query_param`), so the default cannot be an
    ``index=`` argument on the selectbox.
    """
    seed_query_param(key, param, options)
    if key not in st.session_state:
        st.session_state[key] = default


def _from_default_for(releases: list[str]) -> str:
    """The "From release" default: normally the second-newest, but when a deep
    link seeds only ``?to=`` (an open-ended blame window), the release one older
    than that ``to`` — so the pickers open on an older→newer range rather than
    tripping the reversed-range warning. *releases* is newest-first."""
    to_seed = st.query_params.get(PARAM_TO)
    if PARAM_FROM not in st.query_params and to_seed in releases:
        i = releases.index(to_seed)
        return releases[i + 1] if i + 1 < len(releases) else releases[i]
    return releases[1]


def _stacks_for_platform(data_url: str, platform: str) -> list[str]:
    """Every release benchmarked on *platform*, newest first.

    Unioned across detectors rather than read from one: detectors join and
    leave the matrix, so no single detector's history is the full set of
    releases.
    """
    stacks: set[str] = set()
    for detector in _cached_list_detectors(data_url):
        try:
            stacks.update(_cached_list_stacks(data_url, detector, platform))
        except requests.RequestException:
            # A detector never benchmarked on this platform has no such
            # directory to list. Only that is expected here — a broader catch
            # would bury a bug in the listing itself as "no releases".
            continue
    return sorted(stacks, reverse=True)


def _packages(data_url: str, platform: str, stack: str) -> dict | None:
    """The release's package map, from whichever detector has it recorded.

    Detectors are tried in turn because any one of them may have skipped this
    release, or may have run it before provenance capture existed.
    """
    for detector in _cached_list_detectors(data_url):
        packages = _cached_fetch_stack_packages(data_url, detector, platform, stack)
        if packages:
            return packages
    return None


def packages_for_release(data_url: str, platform: str, release: str) -> dict | None:
    """The package map for a bare release tag (e.g. ``2026-07-10``) on *platform*,
    from whichever detector recorded it. The public provenance lookup other tabs
    use, so they need neither the ``key4hep-`` prefix nor this module's
    internals."""
    return _packages(data_url, platform, _PREFIX + release)




def render(data_url: str, platform: str, detector: str, sample: str) -> None:
    st.caption(
        "Which Key4hep packages moved between two nightly releases — the upstream "
        f"changes a regression between them could be attributed to. "
        f"[Learn more →]({_DOCS_URL})"
    )

    stacks = _stacks_for_platform(data_url, platform)
    if len(stacks) < 2:
        st.info(
            f"Need at least two Key4hep releases on `{platform}` to compare; "
            f"found {len(stacks)}."
        )
        return

    releases = [_release(s) for s in stacks]
    # Default to the two most recent releases: "what came in last night?" is the
    # question this tab exists to answer. Defaulting both pickers to the newest
    # would open the tab on "pick two different releases" instead.
    col_from, col_to = st.columns(2)
    with col_from:
        # Default the baseline to the second-newest release ("what came in last
        # night?"), except when a deep link seeds only `to`: then default `from`
        # to the release just older than it, so an open-ended blame link
        # (onset far in the past) opens on a valid older→newer range instead of
        # a reversed one.
        from_default = _from_default_for(releases)
        _seed("stack_from", PARAM_FROM, releases, from_default)
        base_release = st.selectbox(
            "From release", releases, key="stack_from",
            help=f"The older nightly tag — the accepted baseline. {_TAG_HELP}",
        )
    with col_to:
        _seed("stack_to", PARAM_TO, releases, releases[0])
        head_release = st.selectbox(
            "To release", releases, key="stack_to",
            help=f"The newer nightly tag. {_TAG_HELP}",
        )
    st.query_params[PARAM_FROM] = base_release
    st.query_params[PARAM_TO] = head_release

    if base_release == head_release:
        st.info("Pick two different releases to compare.")
        return
    # Ordered by position in the newest-first list rather than by comparing the
    # tags, so this and _span read the release order the same single way.
    if releases.index(base_release) < releases.index(head_release):
        # Reversing the range would report every change with its sign flipped.
        st.warning(
            f"**{base_release}** is newer than **{head_release}** — swap them to read "
            "the diff in the direction time runs.",
            icon="⚠️",
        )
        return

    span = _span(releases, base_release, head_release)
    base = _packages(data_url, platform, _PREFIX + base_release)
    head = _packages(data_url, platform, _PREFIX + head_release)
    missing = [r for r, p in ((base_release, base), (head_release, head)) if not p]
    if missing:
        st.warning(
            f"No stack provenance recorded for {', '.join(f'**{m}**' for m in missing)}. "
            "Releases benchmarked before provenance capture, or whose stack had already "
            "aged off CVMFS when the history was backfilled, cannot be diffed — but any "
            "regressions with onset in this range are still listed below.",
            icon="❔",
        )
    else:
        _render_diff(base, head, base_release, head_release, span)
    # Independent of the package diff: the reverse view is built from the
    # regression reports, so it renders even when provenance is unavailable. A
    # non-empty span means the *selected range* spans more than one release
    # step (see :func:`_span`) — the only case where the diff is cumulative.
    _render_regressions_in_range(
        data_url, platform, base_release, head_release,
        multi_release=bool(span), detector=detector, sample=sample,
    )


#: Cap on reverse-view rows so a month-wide range can't produce an unbounded
#: table; the largest by |Δ| are kept.
_MAX_REGRESSIONS = 30


def _regressions_in_range(
    reports, platform: str, older_release: str, newer_release: str,
    *, detector: str | None = None, sample: str | None = None,
) -> list:
    """Confirmed regressions across *reports* (raw report dicts) whose onset
    falls in ``(older_release, newer_release]``, deduped per distinct onset
    *run* and sorted worst-first with unknown magnitude last. *detector* and
    *sample* narrow the selection to one sidebar scope; ``None`` keeps the
    whole platform.

    Pure — no Streamlit, no network — so the selection, dedup and ordering are
    testable on their own. Dedup keys on the onset run, not just its release:
    several runs share a release, and a re-anchored series can confirm two
    separate steps whose onset runs differ but whose onset releases match.
    """
    hits, seen = [], set()
    for raw in reports:
        if not raw:
            continue
        for v in from_json(raw).regressions:
            if v.platform != platform or not _blame.onset_in_range(v, older_release, newer_release):
                continue
            if detector is not None and v.detector != detector:
                continue
            if sample is not None and v.sample != sample:
                continue
            # A same-release window (onset == baseline) means the stack did not
            # move across the step, so it cannot be an effect of any diff — drop
            # it rather than list it with a nonsensical X → X window.
            if _blame.classify(v) is _blame.WindowKind.SAME_STACK:
                continue
            key = (v.detector, v.sample, v.label, v.metric, v.sub_detector, v.onset_run_id)
            if key not in seen:
                seen.add(key)
                hits.append(v)
    # Unknown relative magnitude (a zero baseline, or an absolute-floor metric)
    # sorts last rather than as if it were 0 %.
    hits.sort(key=lambda v: (v.pct_change is None, -abs(v.pct_change or 0.0)))
    return hits


def _render_regressions_in_range(
    data_url: str, platform: str, base_release: str, head_release: str,
    *, multi_release: bool, detector: str, sample: str,
) -> None:
    """Reverse attribution: the confirmed regressions whose onset falls in this
    release range — candidate effects of the diff above. Same-release regressions
    are excluded (no stack moved for them).

    Scoped to the sidebar's *detector* and *sample* by default; an
    "All detectors" toggle widens it to the whole platform, since a package
    change can regress any detector that sources it.

    *multi_release* is whether the *selected* range spans more than one release
    step: only then is the diff cumulative, so only then do the per-regression
    blame-window column and its warning appear. It is not derived per regression
    — a regression's baseline can predate the selected base, which would trip a
    per-window check even for two neighbouring releases.
    """
    dates = _cached_list_report_dates(data_url)
    # A regression confirms no earlier than its onset release, so only reports
    # on/after the older end can carry one whose onset is in range. Fetch each
    # night individually so it is cached once and reused as the range changes.
    reports = [_cached_fetch_report(data_url, d) for d in dates if d >= base_release]
    all_hits = _regressions_in_range(reports, platform, base_release, head_release)
    scoped_hits = [
        v for v in all_hits if v.detector == detector and v.sample == sample
    ]

    st.markdown("##### Regressions this change may have caused")
    caption_col, toggle_col = st.columns([3, 1], vertical_alignment="center")
    with toggle_col:
        show_all = st.toggle(
            "All detectors",
            value=False,
            key="stack_regr_all",
            help="The list is scoped to the sidebar's detector and sample. A "
                 "package change can regress any detector that sources it — "
                 "switch on to see the whole platform.",
        )
    hits = all_hits if show_all else scoped_hits
    n_elsewhere = len(all_hits) - len(scoped_hits)
    with caption_col:
        scope = (
            "across **all detectors** on this platform" if show_all
            else f"for **{detector}** · **{_pretty_sample(sample)}**"
        )
        extra = (
            f" {n_elsewhere} more in other detectors/samples."
            if not show_all and hits and n_elsewhere else ""
        )
        st.caption(
            f"Confirmed regressions {scope} whose step first appeared inside "
            f"this release range — candidate effects of the changes above.{extra}"
        )
    if not hits:
        if n_elsewhere:
            st.info(
                f"No confirmed regression for **{detector}** · "
                f"**{_pretty_sample(sample)}** has its onset in this release range "
                f"— but {n_elsewhere} elsewhere on this platform do (switch on "
                "*All detectors*).",
                icon="✅",
            )
        else:
            st.info(
                "No confirmed regression has its onset in this release range.",
                icon="✅",
            )
        return

    shown = hits[:_MAX_REGRESSIONS]
    if multi_release:
        st.warning(
            "This is a **multi-release** range, so its package diff above is "
            "**cumulative**. Each regression below was caused by the changes in its own "
            "**blame window** (shown) — *not* necessarily by every package that differs "
            "across the whole range. Compare consecutive releases to pin a single step.",
            icon="⚠️",
        )

    # The same ledger the Regressions tab shows, plus each row's blame window —
    # the sub-range of this diff the step actually entered in.
    flag_table(shown, scope=show_all, blame_window=True)
    if len(hits) > _MAX_REGRESSIONS:
        st.caption(f"Showing the {_MAX_REGRESSIONS} largest of {len(hits)} by |Δ|.")

    _render_outlier_scatter(reports, platform, hits, head_release, scoped=not show_all)


#: Fill for the accepted-baseline band on the outlier plane and its marginals
#: — the same visual device as the Regressions drill-down's band.
_BASELINE_FILL = "rgba(31,119,180,0.08)"


# ── typical vs outlier: one config's runs in the CPU × memory plane ──────────

#: Selectable axes for the outlier plane, derived from the engine's own
#: metric→family map so the choices always track what the reports record.
_METRIC_FAMILIES = {**RUN_METRICS, **EVENT_METRICS}
_TIME_CHOICES = [m for m, fam in _METRIC_FAMILIES.items() if fam == "time"]
_MEMORY_CHOICES = [m for m, fam in _METRIC_FAMILIES.items() if fam == "memory"]

def _same_onset(a: MetricVerdict, b: MetricVerdict) -> bool:
    """Whether two verdicts' confirmed steps are the *same event* rather than
    two unrelated flags that merely fall in the same range: matched on the
    onset run id (a night can share a release with many others, so the run is
    the real identity), falling back to the onset release for reports written
    before run-id tracking."""
    if a.onset_run_id and b.onset_run_id:
        return a.onset_run_id == b.onset_run_id
    return bool(a.onset_run_date) and a.onset_run_date == b.onset_run_date


def _scatter_candidates(hits: list) -> list[tuple]:
    """One ``(detector, sample, label, x_metric, y_metric, both)`` per config
    with a confirmed step in range: *both* whether time AND memory stepped
    **at the same onset** — the diagonal-step case the scatter exists for.

    When such a pair exists, *x*/*y* are drawn from *that same pair* (worst
    first among ties), not independently from each family's own worst metric
    — picking axes and the *both* flag from two different pairs would let the
    label claim a joint step while plotting two unrelated ones. Only without
    any shared-onset pair do *x*/*y* fall back to each family's worst metric
    on its own (``wall_time_s``/``peak_rss_mb`` when a family has no flag).

    Ordered both-first, then by the config's worst |Δ|. Region-level rows
    carry no cross-metric story and are skipped. Pure — the unit-test surface.
    """
    by_cfg: dict[tuple, list] = {}
    for v in hits:
        if v.sub_detector is None:
            by_cfg.setdefault((v.detector, v.sample, v.label), []).append(v)
    ranked = []
    for (det, samp, label), vs in by_cfg.items():
        times = [v for v in vs if v.metric_family == "time"]
        mems = [v for v in vs if v.metric_family == "memory"]
        worst = max((abs(v.pct_change or 0.0) for v in vs), default=0.0)
        # vs (and so times/mems) preserves hits' worst-|Δ|-first order, so the
        # first matching pair is also the worst shared-onset pair available.
        paired = next(
            ((t, m) for t in times for m in mems if _same_onset(t, m)), None
        )
        if paired is not None:
            x_metric, y_metric, both = paired[0].metric, paired[1].metric, True
        else:
            x_metric = times[0].metric if times else "wall_time_s"
            y_metric = mems[0].metric if mems else "peak_rss_mb"
            both = False
        ranked.append(
            ((not both, -worst),
             (det, samp, label, x_metric, y_metric, both))
        )
    ranked.sort(key=lambda t: t[0])
    return [c for _, c in ranked]


def _series_points(
    reports, detector: str, platform: str, sample: str, label: str,
    x_metric: str, y_metric: str,
) -> pd.DataFrame:
    """One row per run night with both metrics' raw values for one config,
    read from the already-fetched nightly reports — a verdict of *any*
    severity carries the night's measured value, so the scatter costs no run
    downloads. Columns ``night, x, y, k4h_release``, sorted by night; a stale
    carried-forward group collapses onto its real run night. Pure.
    """
    rows: dict[str, tuple] = {}
    for raw in reports:
        if not raw:
            continue
        for g in from_json(raw).groups:
            if (g.detector, g.platform, g.sample) != (detector, platform, sample):
                continue
            vals = {
                v.metric: float(v.value)
                for v in g.verdicts
                if v.label == label and v.sub_detector is None
                and v.metric in (x_metric, y_metric) and v.value is not None
            }
            if x_metric in vals and y_metric in vals:
                rows[g.run_date] = (vals[x_metric], vals[y_metric], g.k4h_release)
    return pd.DataFrame(
        [(night, *rows[night]) for night in sorted(rows)],
        columns=["night", "x", "y", "k4h_release"],
    )


def _bound_to_release(pts: pd.DataFrame, head_release: str) -> pd.DataFrame:
    """Points measured at or before *head_release*. A historical range's
    reports are fetched with no upper date bound (a regression can *confirm*,
    and so first appear in a report, well after its onset — see
    :func:`_render_regressions_in_range`), so without this the outlier plane
    would silently plot runs from releases never part of the comparison, as
    if they were effects of it. ``k4h_release`` is ``key4hep-YYYY-MM-DD``,
    which orders chronologically as a plain string. Pure."""
    return pts[pts["k4h_release"].astype(str) <= _PREFIX + head_release]


def _onset_night(pts: pd.DataFrame, cfg_hits: list) -> str | None:
    """The run night the config's earliest in-range step appeared on. The
    onset run id *is* a night (run directories are named by date); reports
    written before onset-run tracking fall back to the first plotted night
    that measured the onset release. Never compares a run date to a release
    date — the nightly lags, so the two routinely differ."""
    ids = [v.onset_run_id for v in cfg_hits if v.onset_run_id]
    if ids:
        return min(ids)
    rels = [v.onset_run_date for v in cfg_hits if v.onset_run_date]
    if not rels:
        return None
    hit = pts[pts["k4h_release"].astype(str).str.endswith(min(rels))]
    return hit["night"].min() if not hit.empty else None


def _axis_title(metric: str) -> str:
    """Human-readable axis title with units, e.g. ``Wall time (s)``."""
    name = _METRIC_LABELS.get(metric, metric)
    name = name[:1].upper() + name[1:]
    unit = _METRIC_UNITS.get(metric, "")
    return f"{name} ({unit})" if unit else name


def _add_baseline_bands(
    fig: go.Figure, cfg_hits: list, x_metric: str, y_metric: str,
    *, row: int, col: int,
) -> None:
    """The judged baseline crosshair for one subplot cell: for each of the two
    axes that has a flagged verdict, the median (dashed) and the detection
    gate (median ± z·MAD) — the same band the Regressions drill-down draws."""
    for v in cfg_hits:
        med, mad = v.baseline_median, v.baseline_mad or 0.0
        if med is None:
            continue
        if v.metric == x_metric:
            fig.add_vline(x=med, line_dash="dash", line_color=PALETTE[0],
                          line_width=1, row=row, col=col)
            if mad > 0:
                fig.add_vrect(x0=med - Z_THRESHOLD * mad, x1=med + Z_THRESHOLD * mad,
                              fillcolor=_BASELINE_FILL, line_width=0,
                              row=row, col=col)
        elif v.metric == y_metric:
            fig.add_hline(y=med, line_dash="dash", line_color=PALETTE[0],
                          line_width=1, row=row, col=col)
            if mad > 0:
                fig.add_hrect(y0=med - Z_THRESHOLD * mad, y1=med + Z_THRESHOLD * mad,
                              fillcolor=_BASELINE_FILL, line_width=0,
                              row=row, col=col)


def _outlier_figure(
    pts: pd.DataFrame, cfg_hits: list, x_metric: str, y_metric: str, label: str
) -> go.Figure:
    """The config's nightly runs in the (time, memory) plane, with each
    metric's 1D distribution in a margin.

    Main cell: the accepted baseline crosshair per judged axis (median ± the
    detection gate, the same band the drill-down draws), the nights before the
    step in the palette hue, the nights from the onset on in the confirmed red
    — the "outlier" cluster the step created — and the onset night ringed with
    the standard confirmed halo. The margins histogram the same nights per
    metric separately (shared bins, before/after overlaid), so a step that
    moved only one of the two shows up in its own margin.
    """
    # Only the two plotted axes decide the split — an unrelated third flagged
    # metric on this config (a candidate can carry more than the pair shown)
    # must not pull its onset into a split neither axis was judged against.
    axis_hits = [v for v in cfg_hits if v.metric in (x_metric, y_metric)]
    onset = _onset_night(pts, axis_hits)
    before = pts if onset is None else pts[pts["night"] < onset]
    after = pts.iloc[0:0] if onset is None else pts[pts["night"] >= onset]

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.78, 0.22], row_heights=[0.78, 0.22],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.03, vertical_spacing=0.04,
    )

    _add_baseline_bands(fig, cfg_hits, x_metric, y_metric, row=1, col=1)
    # Repeat only the relevant half of the crosshair in each margin, so the
    # 1D distributions read against the same gate as the plane.
    _add_baseline_bands(fig, cfg_hits, x_metric, "", row=2, col=1)
    _add_baseline_bands(fig, cfg_hits, "", y_metric, row=1, col=2)

    hover = (
        "<b>%{customdata[0]}</b> (tag %{customdata[1]})<br>"
        f"{_axis_title(x_metric)}: %{{x:.4g}}<br>"
        f"{_axis_title(y_metric)}: %{{y:.4g}}<extra></extra>"
    )
    splits = (
        ("before the step", before, PALETTE[0]),
        ("from the onset on", after, FLAG_MARKS["CONFIRMED"]["color"]),
    )
    for name, frame, color in splits:
        if frame.empty:
            continue
        fig.add_trace(go.Scatter(
            x=frame["x"], y=frame["y"], mode="markers", name=name,
            legendgroup=name,
            marker=dict(size=9, color=_to_rgba(color, 0.55),
                        line=dict(color=color, width=1.5)),
            customdata=frame[["night", "k4h_release"]].values,
            hovertemplate=hover,
        ), row=1, col=1)
        # Marginal 1D distributions of the same nights. ``bingroup`` keys the
        # per-axis bin layout, so the before/after histograms stay comparable.
        fig.add_trace(go.Histogram(
            x=frame["x"], name=name, legendgroup=name, showlegend=False,
            bingroup="x", marker=dict(color=_to_rgba(color, 0.55)),
        ), row=2, col=1)
        fig.add_trace(go.Histogram(
            y=frame["y"], name=name, legendgroup=name, showlegend=False,
            bingroup="y", marker=dict(color=_to_rgba(color, 0.55)),
        ), row=1, col=2)
    if onset is not None:
        on = pts[pts["night"] == onset]
        if not on.empty:
            add_severity_markers(
                fig, on.assign(name=label), x_col="x", y_col="y", name_col="name",
                severity="CONFIRMED", hover_y="%{y:.4g}",
                row=1, col=1,
            )

    t_margin = 30
    plot_h = 430
    legend, b_margin = _legend_below(
        plot_h, 2, t_margin=t_margin, tick_clearance=40,
        entry_width=180, font_size=12,
    )
    fig.update_layout(
        template=_TEMPLATE,
        barmode="overlay",
        height=plot_h + t_margin + b_margin,
        margin=dict(l=10, r=10, t=t_margin, b=b_margin),
        legend=legend,
    )
    # Shared axes put the value labels on the bottom row / left column; the
    # count axes of the margins stay unlabelled (their magnitude is not the
    # point — the split and the position against the band are).
    fig.update_xaxes(title_text=_axis_title(x_metric), row=2, col=1)
    fig.update_yaxes(title_text=_axis_title(y_metric), row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    return fig


def _render_outlier_scatter(
    reports, platform: str, hits: list, head_release: str, *, scoped: bool
) -> None:
    """The "typical values and the outlier" view for one config of the range's
    regressions. Opens automatically when the top candidate stepped in CPU
    *and* memory — the diagonal-outlier case the plane exists for.

    *reports* is fetched with no upper date bound (a regression's *onset* can
    be in range while it only *confirms*, and so first appears in a report,
    later — see :func:`_render_regressions_in_range`), but the plotted points
    are capped at *head_release*: without that cap, a historical range would
    silently pull in runs from releases never part of the comparison, and
    "the runs from the onset on" would misrepresent stack changes that
    happened after this diff as effects of it.
    """
    candidates = _scatter_candidates(hits)
    if not candidates:
        return
    st.markdown("###### Typical vs outlier")
    st.caption(
        "One config's nightly runs in the CPU × memory plane, read from the "
        "already-fetched reports (no run downloads). The dashed crosshair is "
        "the accepted baseline each judged axis was gated on; red points are "
        "the runs from the plotted axes' earliest flagged onset on. "
        "“CPU + memory stepped” means both moved at the *same* onset — a "
        "genuine diagonal step; a config with two flagged metrics but "
        "different onsets is still two separate steps, not one cluster. "
        "The margins show each metric's own 1D distribution, so a step in "
        "only one of the two still stands out."
    )

    def _name(c) -> str:
        det, samp, label, _x, _y, both = c
        name = label if scoped else f"{label} — {det}, {_pretty_sample(samp)}"
        return f"{name} (CPU + memory stepped)" if both else name

    options = ["—"] + [_name(c) for c in candidates]
    picker_row = st.container(
        horizontal=True, vertical_alignment="bottom", width="content"
    )
    with picker_row:
        _drop_stale_selection("stack_outlier_cfg", options)
        choice = st.selectbox(
            "Config", options,
            index=1 if candidates[0][5] else 0,
            key="stack_outlier_cfg",
            width=320,
            help="Configs with a confirmed step in this range — one where CPU "
                 "and memory both stepped opens by default; pick “—” to hide "
                 "the plane.",
        )
        if choice == "—":
            return
        det, samp, label, x_default, y_default, _both = (
            candidates[options.index(choice) - 1]
        )
        # The same time/memory axis choice as the Overview tab, defaulting to
        # the config's own flagged metrics. Keys are scoped per (detector,
        # sample, config) rather than just the label — "All detectors" can
        # show several detectors sharing one label (e.g. ``baseline_all``),
        # and a label-only key would leak one detector's axis pick into
        # another's on selection.
        cfg_key = f"{det}_{samp}_{label}"
        x_metric = st.selectbox(
            "Time metric", _TIME_CHOICES,
            index=_TIME_CHOICES.index(x_default) if x_default in _TIME_CHOICES else 0,
            key=f"stack_outlier_tmetric_{cfg_key}",
            format_func=_axis_title, width=220,
        )
        y_metric = st.selectbox(
            "Memory metric", _MEMORY_CHOICES,
            index=(_MEMORY_CHOICES.index(y_default)
                   if y_default in _MEMORY_CHOICES else 0),
            key=f"stack_outlier_mmetric_{cfg_key}",
            format_func=_axis_title, width=220,
        )
    pts = _series_points(reports, det, platform, samp, label, x_metric, y_metric)
    pts = _bound_to_release(pts, head_release)
    if pts.empty:
        st.info(
            "The fetched reports carry no nightly values for this config on "
            "the selected metrics within the selected release range."
        )
        return
    cfg_hits = [
        v for v in hits
        if (v.detector, v.sample, v.label) == (det, samp, label)
        and v.sub_detector is None
    ]
    st.plotly_chart(
        _outlier_figure(pts, cfg_hits, x_metric, y_metric, label),
        width="stretch", key="stack_outlier_chart",
    )


def _span(releases: list[str], base_release: str, head_release: str) -> str:
    """How far apart the two releases are, as a sentence.

    A month-wide range is a *cumulative* diff, not one night's change — easy to
    misread when the table looks the same either way. Counting the releases in
    between says so; counting days would not, since the nightly skips days.
    """
    n_between = releases.index(base_release) - releases.index(head_release)
    if n_between == 1:
        return ""  # consecutive: the two tags in the heading already say it
    days = (pd.Timestamp(head_release) - pd.Timestamp(base_release)).days
    return (
        f"**{n_between} releases apart** ({_plural(days, 'day')}) — this is the "
        f"cumulative change across all {n_between}, not one night's."
    )


def _render_summary(
    n_changed: int, n_same: int, base_release: str, head_release: str, span: str
) -> None:
    """At-a-glance header, mirroring the Regressions tab's verdict banner."""
    with st.container(border=True):
        st.markdown(f"##### Stack diff — {base_release} → {head_release}")
        if span:
            st.caption(span)
        cols = st.columns(3)
        cols[0].metric(
            "Packages changed", n_changed,
            help="Packages whose upstream commit differs between the two releases. "
                 "These are the only places an upstream cause could come from.",
        )
        cols[1].metric(
            "Unchanged", n_same,
            help="Built from the identical commit in both releases.",
        )
        cols[2].metric(
            "Tracked", n_changed + n_same,
            help="Packages Key4hep builds from git, and whose commit is therefore "
                 "recorded. Release-tarball dependencies have no upstream commit "
                 "and are not tracked.",
        )


def _render_diff(
    base: dict, head: dict, base_release: str, head_release: str, span: str
) -> None:
    changes = diff_packages(base, head)
    same = unchanged_packages(base, head)
    _render_summary(len(changes), len(same), base_release, head_release, span)

    if not changes:
        # Not "nothing found" but "these are the same stack" — which rules an
        # upstream commit out entirely, and is an answer rather than a blank.
        st.success(
            f"**These two releases are the identical stack.** All {len(same)} tracked "
            "packages sit at the same commit, so nothing upstream changed between them: "
            "a metric that moved between these releases moved for another reason — the "
            "host, the sample, or noise.",
            icon="✅",
        )
        return

    df = pd.DataFrame([
        {
            "": _STATUS_BADGE[c.status],
            "Package": c.name,
            "From": (c.base_commit or "—")[:12],
            "To": (c.head_commit or "—")[:12],
            "Compare": c.compare_url,
        }
        for c in changes
    ])
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "": st.column_config.TextColumn(
                "", width="small",
                help="🔄 moved to a new commit · ➕ entered the stack · ➖ left it",
            ),
            "Package": st.column_config.TextColumn("Package", width="medium"),
            "From": st.column_config.TextColumn(
                "From", help="The package's commit in the older release.",
            ),
            "To": st.column_config.TextColumn(
                "To", help="The package's commit in the newer release.",
            ),
            "Compare": st.column_config.LinkColumn(
                "Compare", display_text="↗ commits", width="small",
                help="Every commit in this package's range, on its forge. Absent for a "
                     "package that was added or removed (there is no range to compare), "
                     "or hosted on a forge whose URL layout we do not know.",
            ),
        },
    )
    if same:
        st.caption(
            f"{_plural(len(same), 'tracked package')} unchanged between these releases."
        )
