"""Streamlit rendering for the shared run-reliability verdicts.

The evidence-building and verdict logic lives in
:mod:`k4bench.results.reliability_evidence` (Streamlit-free, shared with the
nightly regression report); this module keeps only the dashboard widgets.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from k4bench.analysis.loader import run_keys
from k4bench.results.reliability_evidence import reliability_verdict as _reliability_verdict


_RUN_SCOPE_OPTIONS = ("Reliable only", "All runs")


def render_reliability_scope(
    n_unreliable: int,
    dates: list[str],
    *,
    key: str,
    n_failed: int = 0,
    failed_dates: list[str] | None = None,
    inline: bool = False,
    slot=None,
) -> bool:
    """Render the dashboard-wide run-quality scope selector.

    Returns ``True`` for **Reliable only** and ``False`` for **All runs**.
    **Reliable only** hides both the *n_unreliable* runs that failed the
    host-condition check (on the nightly tags *dates*) and the *n_failed*
    failed configs (on *failed_dates*), whose partial values the views
    otherwise draw as failure markers. The counts are attached to the control
    label and the dates live in its help, so every historical view presents
    the same state, consequence, and explanation as one unit.

    The boolean migration preserves live sessions created by the former
    ``Exclude unreliable runs`` toggle without changing the existing widget
    keys shared by callers and Overview sub-views. Callers reserve *slot* on
    their existing controls line when the unreliable count is only known later;
    the selector then fills that position without creating another row.
    """
    stored = st.session_state.get(key)
    if isinstance(stored, bool):
        st.session_state[key] = (
            _RUN_SCOPE_OPTIONS[0] if stored else _RUN_SCOPE_OPTIONS[1]
        )
    elif stored is None and key in st.session_state:
        st.session_state[key] = _RUN_SCOPE_OPTIONS[0]
    elif stored not in (None, *_RUN_SCOPE_OPTIONS):
        st.session_state[key] = _RUN_SCOPE_OPTIONS[0]

    hidden = []
    if n_unreliable:
        affected = ", ".join(dates) if dates else "dates unavailable"
        hidden.append(
            f"runs that failed the conservative host-condition check "
            f"(nightly tags: {affected}; see Machine Info for the per-run verdict)"
        )
    if n_failed:
        affected = ", ".join(failed_dates) if failed_dates else "dates unavailable"
        hidden.append(
            f"failed configs, whose partial values All runs marks with ❌ "
            f"(nightly tags: {affected}; see Logs for the failure)"
        )
    counts = [
        *((f"⚠️ {n_unreliable} unreliable",) if n_unreliable else ()),
        *((f"❌ {n_failed} failed",) if n_failed else ()),
    ]
    if slot is not None:
        host = slot
    elif inline:
        host = st
    else:
        host = st.container(
            horizontal=True, horizontal_alignment="right",
            vertical_alignment="bottom", width="stretch",
        )
    runs = host.segmented_control(
        " · ".join(["Runs", *counts]),
        _RUN_SCOPE_OPTIONS,
        default=_RUN_SCOPE_OPTIONS[0],
        required=True,
        key=key,
        bind="query-params",
        help=f"Reliable only hides {' and '.join(hidden)}. All runs shows "
             "them.",
        width="content",
    ) or _RUN_SCOPE_OPTIONS[0]
    return runs == _RUN_SCOPE_OPTIONS[0]


def render_sidebar_run_quality(
    machine_info: dict | None,
    results: pd.DataFrame | None,
) -> None:
    """Render a compact run-quality status card for the selected run in the sidebar.

    Shows the same conservative pass/fail verdict the Machine Info tab reports for
    this run, placed right under the run/stack selector so the selected release's
    quality is visible at a glance on every tab. Nothing is drawn when there is no
    machine info, or no hard criterion could be judged (verdict ``None``).

    The context-switch baseline is intentionally omitted: it is advisory-only and
    never changes the pass/fail verdict (see :func:`run_reliability_map`), so this
    needs no trend history and works in local mode too.
    """
    if not machine_info:
        return
    verdict = _reliability_verdict(machine_info, results)
    reliable = verdict.reliable
    if reliable is None:
        return
    if reliable is False:
        # Name the hard checks that failed, mirroring the Machine Info banner, so the
        # card explains itself without opening the tab.
        names = ", ".join(c.name for c in verdict.failures)
        subtitle = (
            f"Failed: {names} — see Machine Info." if names
            else "Likely host contention — see the Machine Info tab."
        )
        accent, bg, icon, title = "#d63c3c", "rgba(214,60,60,0.08)", "⚠️", "Unreliable run"
    else:
        accent, bg, icon, title, subtitle = (
            "#2ea043", "rgba(46,160,67,0.07)", "✅", "Reliable run",
            "Passed the host-condition checks.",
        )
    st.markdown(
        f"""
        <div style="background:{bg};border:1px solid {accent}45;
                    border-left:3px solid {accent};border-radius:8px;
                    padding:0.5rem 0.7rem;margin:0.15rem 0 0.35rem 0;">
          <div style="display:flex;align-items:center;gap:0.5rem;">
            <span style="font-size:1.05rem;line-height:1;">{icon}</span>
            <div style="line-height:1.3;">
              <div style="font-size:0.66rem;text-transform:uppercase;letter-spacing:0.06em;
                          color:{accent};font-weight:700;">{title}</div>
              <div style="font-size:0.72rem;color:#9a9a9a;">{subtitle}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_reliability_filter(
    df: pd.DataFrame,
    reliability: dict[str, bool | None] | None,
    *,
    key: str,
    date_col: str = "x_date",
    slot=None,
    failed: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, set[str], pd.DataFrame | None]:
    """Render the shared reliable-only/all-runs choice and filter the frames.

    Shared by every tab that plots historical (multi-run) data so the run-scope
    choice behaves identically everywhere. *df* holds the judgeable rows and
    carries ``run_id`` and, when available, the release date ``x_date``.
    *reliability* maps their :func:`~k4bench.analysis.loader.run_key` to a verdict.
    *failed* holds the failed configs' rows the caller marks as failures.
    When no run in *df* is flagged unreliable — including when *reliability* is
    empty/``None`` (e.g. local mode, or no machine info) — and *failed* is empty
    or ``None``, both frames are returned unchanged and nothing is drawn.

    **Reliable only** is the default: runs that failed the conservative reliability
    check and every failed config are dropped unless the user chooses **All runs**.
    If that leaves nothing to plot, an explanatory warning is shown and the empty
    frames are returned, so the caller can ``return`` without plotting.

    *date_col* is the column whose dates are listed in the selector's help; it
    defaults to ``x_date`` (the nightly tag) so the dates match the plot x-axis and the
    Machine Info tab, rather than the CI run date.

    *slot* is an optional placeholder already positioned in the caller's control
    row. It is intentionally passed through to :func:`render_reliability_scope`
    only when a selector is needed, so reliable-only views leave no empty chrome.

    The returned frames are *df* and *failed* as they are to be plotted; the set of
    excluded run keys is empty whenever no unreliable run was dropped — none in *df*, or
    **All runs** selected — so a caller can treat it as "the runs that are no
    longer on the chart" without also reading the widget state. Callers that plot
    a *reduction* over several runs need it: a flag earned by one run must not
    survive on a sibling run of the same nightly tag once its own run is excluded,
    and the reduction can only honour that if it knows which runs are still
    standing. Callers that just plot the frame want :func:`render_reliability_filter`.
    """
    # Without a run_id to join verdicts on, or a verdict map at all (local mode /
    # no machine info), no run is flagged unreliable.
    unreliable = pd.Series(False, index=df.index)
    unreliable_ids = set()
    if "run_id" in df.columns and reliability:
        runs = run_keys(df)
        unreliable_ids = {run for run in runs.unique() if reliability.get(run) is False}
        unreliable = runs.isin(unreliable_ids)
    n_failed = failed_count(failed)
    if not unreliable_ids and not n_failed:
        return df, set(), failed

    exclude = render_reliability_scope(
        len(unreliable_ids),
        _tag_dates(df.loc[unreliable], date_col) if unreliable_ids else [],
        key=key,
        n_failed=n_failed,
        failed_dates=_tag_dates(failed, date_col) if n_failed else [],
        slot=slot,
    )
    if not exclude:
        return df, set(), failed
    if unreliable_ids:
        df = df[~unreliable]
    if failed is not None:
        failed = failed.iloc[0:0]
    if df.empty:
        st.warning(
            "Every run for the selected configurations failed or was excluded "
            "as unreliable — nothing left to plot. Choose All runs to see them."
        )
    return df, unreliable_ids, failed


def failed_count(failed: pd.DataFrame | None) -> int:
    """How many failed configs *failed* holds: its distinct ``(run, config)``
    pairs, or its rows when it does not name them."""
    if failed is None or failed.empty:
        return 0
    keys = [
        c for c in ("run_id", "x_date", "run_night", "detector", "label")
        if c in failed.columns
    ]
    return len(failed[keys].drop_duplicates()) if keys else len(failed)


def _tag_dates(df: pd.DataFrame | None, date_col: str) -> list[str]:
    """The nightly tags *df*'s rows carry in *date_col*, as sorted
    ``YYYY-MM-DD`` strings; none when the frame lacks the column, so the
    selector degrades to its counts rather than raising."""
    if df is None or date_col not in df.columns:
        return []
    return sorted(
        pd.to_datetime(df[date_col], errors="coerce")
        .dt.strftime("%Y-%m-%d").fillna("unknown").unique()
    )


def render_reliability_filter(
    df: pd.DataFrame,
    reliability: dict[str, bool | None] | None,
    *,
    key: str,
    date_col: str = "x_date",
    slot=None,
) -> pd.DataFrame:
    """:func:`resolve_reliability_filter` for callers that plot no failure
    markers and only need the frame."""
    return resolve_reliability_filter(
        df, reliability, key=key, date_col=date_col, slot=slot,
    )[0]


def unreliable_pairs(rel_hist: pd.DataFrame) -> set[tuple[str, str]]:
    """The ``(run_night, detector)`` runs that failed the host-reliability
    check. ``None`` (no evidence) never counts as unreliable.

    Addresses runs, not tags, so :func:`drop_unreliable_runs` can drop a
    contended run without taking a reliable rerun of the same nightly with it.
    """
    if rel_hist.empty:
        return set()
    flagged = (
        rel_hist.loc[rel_hist["reliable"].eq(False), ["run_night", "detector"]]
        .drop_duplicates()
    )
    return set(map(tuple, flagged.itertuples(index=False, name=None)))


def render_run_reliability_filter(
    rel_hist: pd.DataFrame, *, key: str, inline: bool = False, slot=None,
    failed: pd.DataFrame | None = None,
) -> tuple[set[tuple[str, str]], bool]:
    """A single explicit choice between reliable-only and all runs.

    The Overview tab's variant of :func:`render_reliability_filter`.
    *rel_hist* has columns ``night, run_night, detector, reliable`` (see
    :func:`tabs.reports.overview._data.reliability_history` — built per run group
    precisely because the flag belongs to the group and to no metric verdict);
    :func:`render_reliability_filter` keys on a global ``{run_id: verdict}``
    map, which cannot express a cross-detector frame where the same night is
    reliable for one detector and not another.
    ``None`` (no evidence) never excludes. The shared selector keeps this
    cross-detector view identical to every other historical dashboard view.

    *failed* holds the failed configs' history rows the caller draws as
    failure markers (columns ``night``, ``run_night`` and ``detector``);
    they are counted on the selector, and exclusion means the caller hides them.

    Returns the unreliable ``(run_night, detector)`` pairs and whether exclusion
    is active, so the caller can drop those *runs* from its history rows
    before it collapses them, and its failure markers. The count is of runs —
    the thing that actually gets dropped, and what the sentence says — while
    the dates listed are the nightly **tags** those runs carry, which is what
    the reader sees on the x-axis.
    """
    pairs = unreliable_pairs(rel_hist)
    n_failed = failed_count(failed)
    if not pairs and not n_failed:
        return set(), False

    dates = sorted(
        rel_hist.loc[rel_hist["reliable"].eq(False), "night"].unique()
    ) if pairs else []
    return pairs, render_reliability_scope(
        len(pairs), dates, key=key, n_failed=n_failed,
        failed_dates=_tag_dates(failed, "night") if n_failed else [],
        inline=inline, slot=slot,
    )
