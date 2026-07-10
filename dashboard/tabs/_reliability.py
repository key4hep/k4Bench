"""Shared run-reliability evidence/verdict helpers.

Split out of :mod:`tabs.machine_info` so the Trends tab (which only needs
:func:`run_reliability_map`) does not have to import that whole module —
avoiding the import-time coupling/circularity risk as the dashboard grows.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from k4bench.results.reliability import ReliabilityVerdict, evaluate_reliability
from ui_utils import _is_valid_df

#: Minimum number of historical per-config samples before a context-switch
#: baseline is trusted. Below this the criterion stays advisory (reported only).
_BASELINE_MIN_SAMPLES = 20


def _ctx_switch_baseline(
    trend_results_df: pd.DataFrame | None, platform: str | None = None
) -> float | None:
    """Median involuntary context switches **per CPU-second** across prior runs.

    Computed from the historical per-config results so the threshold adapts to
    the host's normal scheduler behaviour. Normalising by total CPU time makes it
    robust to run length and core count. When *platform* is given the pool is
    restricted to runs on that platform, since context-switch rates are
    host-specific. Returns ``None`` (criterion stays advisory) when too few
    samples exist to form a stable baseline.

    A median is used deliberately: it is unaffected by the occasional contaminated
    run in the window, so a few noisy runs cannot inflate the baseline and mask
    real interference.
    """
    if not _is_valid_df(trend_results_df):
        return None
    df = trend_results_df
    cols = set(df.columns)
    if "involuntary_ctx_switches" not in cols:
        return None
    if platform and "platform" in cols:
        df = df[df["platform"] == platform]
        if df.empty:
            return None
    if {"user_cpu_s", "sys_cpu_s"} <= cols:
        cpu = df["user_cpu_s"] + df["sys_cpu_s"]
    elif "user_cpu_s" in cols:
        cpu = df["user_cpu_s"]
    else:
        return None
    per_cpu_s = (
        df["involuntary_ctx_switches"] / cpu.replace(0, float("nan"))
    )
    per_cpu_s = per_cpu_s.replace([float("inf"), float("-inf")], float("nan")).dropna()
    per_cpu_s = per_cpu_s[per_cpu_s >= 0]
    if len(per_cpu_s) < _BASELINE_MIN_SAMPLES:
        return None
    return float(per_cpu_s.median())


def _reliability_evidence(
    machine_info: dict,
    results: pd.DataFrame | None,
    ctx_switch_baseline_per_cpu_s: float | None = None,
) -> dict:
    """Collect the inputs :func:`evaluate_reliability` needs from this run.

    Per-config metrics (CPU efficiency, context switches, total CPU) are averaged
    across the run's configs; machine-condition signals come straight from
    ``machine_info``. Missing values are left as ``None`` so the evaluator treats
    them as *unknown* rather than failing on them. *ctx_switch_baseline_per_cpu_s*
    is the historical baseline from :func:`_ctx_switch_baseline`, or ``None`` to
    keep the context-switch criterion advisory.
    """
    cpu_eff = total_cpu = invol = None
    if _is_valid_df(results):
        cols = set(results.columns)
        if {"user_cpu_s", "sys_cpu_s", "wall_time_s"} <= cols:
            tot = results["user_cpu_s"] + results["sys_cpu_s"]
            eff = (tot / results["wall_time_s"].replace(0, float("nan"))).dropna()
            if not eff.empty:
                cpu_eff = float(eff.mean())
            tot = tot.dropna()
            if not tot.empty:
                total_cpu = float(tot.mean())
        if "involuntary_ctx_switches" in cols:
            v = results["involuntary_ctx_switches"].dropna()
            if not v.empty:
                invol = float(v.mean())

    # Worst-case RAM utilisation across the start/end snapshots.
    ram_total = machine_info.get("ram_total_gb")
    avail = [machine_info.get(k) for k in ("ram_available_gb_start", "ram_available_gb_end")]
    avail = [a for a in avail if a is not None]
    if ram_total and avail:
        ram_used_fraction = max(0.0, min(1.0, 1 - min(avail) / ram_total))
    else:
        ram_used_fraction = None

    return {
        "cpu_efficiency":           cpu_eff,
        "total_cpu_s":              total_cpu,
        "involuntary_ctx_switches": invol,
        "ctx_switch_baseline_per_cpu_s": ctx_switch_baseline_per_cpu_s,
        "load_avg_1m_pre":          machine_info.get("load_avg_1m_start"),
        "load_avg_1m_post":         machine_info.get("load_avg_1m_end"),
        "physical_cores":           machine_info.get("cpu_physical_cores"),
        "swap_in_pages":            machine_info.get("swap_in_pages"),
        "swap_out_pages":           machine_info.get("swap_out_pages"),
        "thermal_throttle_events":  machine_info.get("thermal_throttle_events"),
        "ram_used_fraction":        ram_used_fraction,
    }


def _reliability_verdict(
    machine_info: dict,
    results: pd.DataFrame | None,
    ctx_switch_baseline_per_cpu_s: float | None = None,
) -> ReliabilityVerdict:
    """Build the conservative pass/fail verdict for the selected run."""
    return evaluate_reliability(
        **_reliability_evidence(machine_info, results, ctx_switch_baseline_per_cpu_s)
    )


def run_reliability_map(
    trend_results_df: pd.DataFrame | None,
    trend_machine_df: pd.DataFrame | None,
) -> dict[str, bool | None]:
    """Compute the conservative reliability verdict for every run in the trends.

    Reliability is a *per-run* property: one ``machine_info.json`` describes the
    host condition for the whole run, so the same verdict applies to all of that
    run's configs. This joins the per-run machine conditions in
    *trend_machine_df* with the per-config metrics in *trend_results_df* (on
    ``run_id``) and returns ``{run_id: reliable}``, where ``reliable`` is the
    :attr:`ReliabilityVerdict.reliable` tri-state (``True``/``False``/``None``).

    The context-switch baseline is left unset here (advisory only, so it never
    changes the pass/fail verdict), keeping this independent of run history.
    """
    if not _is_valid_df(trend_machine_df) or "run_id" not in trend_machine_df.columns:
        return {}
    # Group once up front so each run's slice is an O(1) dict lookup rather than
    # a fresh O(n_rows) scan of trend_results_df per run.
    have_results = _is_valid_df(trend_results_df) and "run_id" in trend_results_df.columns
    results_by_run = (
        {run_id: group for run_id, group in trend_results_df.groupby("run_id")}
        if have_results else {}
    )
    verdicts: dict[str, bool | None] = {}
    for mrow in trend_machine_df.to_dict("records"):
        run_id = mrow.get("run_id")
        if not run_id:
            continue
        # A missing numeric column arrives as NaN; coerce to None so the
        # conservative check treats it as *unknown* rather than a failure.
        machine = {k: (None if pd.isna(v) else v) for k, v in mrow.items()}
        results = results_by_run.get(run_id)
        verdict = evaluate_reliability(**_reliability_evidence(machine, results))
        verdicts[run_id] = verdict.reliable
    return verdicts


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
        <div class="k4-run-quality-card" title="Open the Machine Info tab"
             style="cursor:pointer;background:{bg};border:1px solid {accent}45;
                    border-left:3px solid {accent};border-radius:8px;
                    padding:0.5rem 0.7rem;margin:0.15rem 0 0.5rem 0;">
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
    # Make the card jump to the Machine Info tab on click. Streamlit has no API to
    # set the active st.tabs panel, so — as in app._force_plotly_relayout_on_tab_switch
    # — a tiny same-origin iframe script reaches the parent document and clicks the
    # "Machine Info" tab button. The handler is bound once per card (dataset flag,
    # idempotent across reruns) and resolves the tab button lazily at click time, so
    # it works even though the card (sidebar) renders before the tabs (main area).
    st.iframe(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          function bind() {
            doc.querySelectorAll('.k4-run-quality-card').forEach(function (card) {
              if (card.dataset.k4TabBound) return;
              card.dataset.k4TabBound = "1";
              card.addEventListener("click", function () {
                const tabs = doc.querySelectorAll('button[data-baseweb="tab"]');
                for (const t of tabs) {
                  if ((t.innerText || "").trim() === "Machine Info") { t.click(); break; }
                }
              });
            });
          }
          bind();
          setTimeout(bind, 800);  // rebind if the card mounts after this runs
        })();
        </script>
        """,
        height=1,
    )


def render_reliability_filter(
    df: pd.DataFrame,
    reliability: dict[str, bool | None] | None,
    *,
    key: str,
    date_col: str = "x_date",
) -> pd.DataFrame:
    """Render the unreliable-run warning + "Exclude unreliable runs" toggle.

    Shared by every tab that plots historical (multi-run) data so the warning text
    and the toggle behave identically everywhere. *df* must carry a ``run_id``
    column; *reliability* is the per-run verdict map from :func:`run_reliability_map`
    (``{run_id: reliable}``). When no run in *df* is flagged unreliable — including
    when *reliability* is empty/``None`` (e.g. local mode, or no machine info) — *df*
    is returned unchanged and nothing is drawn.

    The toggle defaults to *on*: runs that failed the conservative reliability check
    are dropped unless the user disables it. If that empties the frame, an
    explanatory warning is shown and the empty frame is returned, so the caller can
    ``return`` without plotting.

    *date_col* is the column whose dates are listed in the warning text; it defaults
    to ``x_date`` (the nightly tag) so the dates match the plot x-axis and the
    Machine Info tab, rather than the CI run date.
    """
    # No run_id to join verdicts on, or no verdict map at all (local mode / no
    # machine info) — nothing to flag or filter.
    if "run_id" not in df.columns or not reliability:
        return df
    unreliable_ids = {
        rid for rid in df["run_id"].unique() if reliability.get(rid) is False
    }
    if not unreliable_ids:
        return df

    n = len(unreliable_ids)
    # List the affected dates when the column is present; degrade to a count-only
    # message for any future caller whose frame lacks it, rather than raising.
    if date_col in df.columns:
        flagged = df.loc[df["run_id"].isin(unreliable_ids), date_col]
        dates = sorted(
            pd.to_datetime(flagged, errors="coerce")
            .dt.strftime("%Y-%m-%d").fillna("unknown").unique()
        )
        where = f": {', '.join(dates)}"
    else:
        where = ""
    warn_col, toggle_col = st.columns([3, 1], vertical_alignment="center")
    with warn_col:
        st.warning(
            f"⚠️ {n} unreliable run{'s' if n != 1 else ''} detected in this "
            "window — likely affected by host contention (see the Machine "
            f"Info tab for the per-run verdict){where}."
        )
    with toggle_col:
        exclude = st.toggle(
            "Exclude unreliable runs",
            value=True,
            key=key,
            help="Drop runs that failed the conservative reliability check "
                 "from the plots below. On by default; disable to include them.",
        )
    if exclude:
        df = df[~df["run_id"].isin(unreliable_ids)]
        if df.empty:
            st.warning(
                "Every run for the selected configurations was excluded as "
                "unreliable — nothing left to plot."
            )
    return df
