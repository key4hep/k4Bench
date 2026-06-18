"""Page chrome and small sidebar utilities for the dashboard entry point.

Footers, the stale-selection helper — presentation/plumbing that would otherwise
clutter ``app.main()``.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st


def _failed_labels(results: "pd.DataFrame") -> list[str]:
    """Return the labels of configs whose returncode is non-zero (or missing)."""
    if "returncode" not in results.columns or "label" not in results.columns:
        return []
    rc = results["returncode"]
    return sorted(results.loc[rc.fillna(-1) != 0, "label"].astype(str))


def render_run_status(results: "pd.DataFrame | None", run_meta: dict | None) -> None:
    """Top-of-page alert banners for the selected run.

    The source of truth for failure is each config's ``returncode`` (loaded into
    *results*); ``run_meta`` (from ``run_info.json``) supplies the CI run link and
    the ``machine_consistent`` flag. Per-config detail + logs live in the Logs tab
    (:func:`render_logs_tab`).
    """
    if results is None or "label" not in results.columns:
        return
    meta = run_meta or {}

    failed = _failed_labels(results)
    if failed:
        run_url = meta.get("github_run_url")
        link = f" · [CI run logs]({run_url})" if run_url else ""
        st.error(
            f"**{len(failed)} of {len(results)} config(s) failed** in this run: "
            f"{', '.join(f'`{c}`' for c in failed)}. See the **Logs** tab.{link}"
        )
    if meta.get("machine_consistent") is False:
        machines = meta.get("machines") or []
        st.warning(
            "This sweep ran across **more than one machine**, so absolute timing "
            "comparisons between configs may be noisy: "
            + ", ".join(f"`{m}`" for m in machines)
        )


def render_logs_tab(results: "pd.DataFrame | None", run_dir: str | None) -> None:
    """Per-config status overview and a log viewer for the selected run.

    Logs are already in the local cache (``{label}.log`` in *run_dir*), so the
    viewer reads them directly with no extra download.
    """
    if results is None or "label" not in results.columns:
        st.info("No run data available for the current selection.")
        return

    labels = sorted(results["label"].astype(str))
    failed = _failed_labels(results)

    # ── Summary chips ───────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Configs", len(labels))
    c2.metric("Passed", len(labels) - len(failed))
    c3.metric("Failed", len(failed), delta=None if not failed else f"-{len(failed)}",
              delta_color="inverse")

    # ── Status table ────────────────────────────────────────────────────────────
    rc_by_label = dict(zip(results["label"].astype(str), results["returncode"])) \
        if "returncode" in results.columns else {}
    status_df = pd.DataFrame({
        "Config": labels,
        "Status": [
            "✅ Passed" if (pd.notna(rc_by_label.get(lbl)) and rc_by_label.get(lbl) == 0)
            else f"❌ Failed (rc={rc_by_label.get(lbl)})"
            for lbl in labels
        ],
    })
    st.dataframe(
        status_df, hide_index=True, width="stretch",
        column_config={
            "Config": st.column_config.TextColumn("Config", width="large"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
        },
    )

    # ── Log viewer ──────────────────────────────────────────────────────────────
    st.divider()
    if not run_dir:
        st.caption("Logs are only available when viewing cached/remote runs.")
        return

    # Default to the first failed config so its log is one click away.
    default_idx = labels.index(failed[0]) if failed else 0
    chosen = st.selectbox("Log for config", labels, index=default_idx)
    log_path = Path(run_dir) / f"{chosen}.log"
    if not log_path.exists():
        st.caption(f"No log file found for `{chosen}`.")
        return

    text = log_path.read_text(errors="replace")
    lines = text.splitlines()
    # Cap displayed lines so st.code stays responsive (it highlights every line);
    # the full file is always one click away via download.
    _MAX_LINES = 500
    truncated = len(lines) > _MAX_LINES

    head, dl = st.columns([3, 1])
    with head:
        st.caption(
            f"`{log_path.name}` · {len(lines):,} lines · {log_path.stat().st_size / 1024:.0f} KB"
            + (f"  —  showing last {_MAX_LINES:,} lines" if truncated else "")
        )
    with dl:
        st.download_button(
            "⬇ Download full log", log_path.read_bytes(),
            file_name=log_path.name, mime="text/plain",
            use_container_width=True,
        )
    if truncated:
        text = "\n".join(lines[-_MAX_LINES:])
    # Fixed-height container -> the log scrolls within the window instead of
    # pushing the rest of the page down.
    with st.container(height=520):
        st.code(text, language="log")


def _render_footer() -> None:
    """Render a CERN / FCC copyright footer at the bottom of the page."""
    year = date.today().year
    st.markdown(
        f"""
        <hr class="k4-footer" style="border:none;border-top:1px solid rgba(128,128,128,0.25);margin:2.5rem 0 0.8rem 0;">
        <div style="
            display:flex;
            justify-content:center;
            align-items:center;
            gap:1.2rem;
            padding:0.2rem 0 1.2rem 0;
            font-size:0.80rem;
            color:#9a9a9a;
            line-height:1.7;
            text-align:center;
        ">
            <span style="font-size:1.8rem;opacity:0.75;">⚛️</span>
            <div>
                <strong style="color:#c0c0c0;letter-spacing:0.02em;">© {year} CERN</strong>
                &nbsp;·&nbsp;
                For the benefit of the&nbsp;<a
                    href="https://fcc.web.cern.ch/"
                    target="_blank"
                    rel="noopener noreferrer"
                    style="color:#5b9bd5;text-decoration:none;font-weight:600;"
                >FCC project</a>
                <br>
                Created by <strong style="color:#c0c0c0;">Joshua Falco Beirer</strong>
                &nbsp;<span style="opacity:0.6;">(CERN)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_footer() -> None:
    """Render a compact attribution note at the bottom of the sidebar."""
    year = date.today().year
    st.markdown(
        f"""
        <hr style="border:none;border-top:1px solid rgba(128,128,128,0.2);margin:1.5rem 0 0.6rem 0;">
        <div style="font-size:0.72rem;color:#888;text-align:center;line-height:1.6;padding-bottom:0.4rem;">
            <strong style="color:#a0a0a0;">© {year} CERN</strong><br>
            For the benefit of the<br>
            <a href="https://fcc.web.cern.ch/" target="_blank" rel="noopener noreferrer"
               style="color:#5b9bd5;text-decoration:none;">FCC project</a><br>
            <span style="opacity:0.7;">J. F. Beirer (CERN)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _drop_stale_selection(key: str, options: list[str]) -> None:
    """Clear a keyed selectbox's stored value when it's no longer a valid option.

    The dependent dropdowns (Platform → Sample → Stack) rebuild their option lists
    whenever an upstream selection changes, so a value left in ``session_state``
    from the old options can be fed back into ``st.selectbox`` as an invalid
    selection. Popping it *before* the widget is created (the only point at which
    a widget-backed key may be mutated) lets the selectbox re-default cleanly to a
    valid option. A no-op when the stored value is still present in *options*.
    """
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
