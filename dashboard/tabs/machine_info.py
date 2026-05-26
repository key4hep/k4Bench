"""Machine Info tab — displays hardware and OS details for the selected run."""

from __future__ import annotations

import streamlit as st


def render(machine_info: dict | None, run_meta: dict | None = None) -> None:
    """Render the Machine Info tab.

    Parameters
    ----------
    machine_info:
        Dict loaded from ``machine_info.json``, or ``None`` if unavailable.
    run_meta:
        Optional run metadata dict (from ``_parse_run_dir``) for CI links.
    """
    if machine_info is None:
        st.info(
            "No machine info available for this run. "
            "Machine info is written by CI jobs running the new directory layout."
        )
        return

    # ── CI / run context ──────────────────────────────────────────────────────
    if run_meta:
        ctx_cols = st.columns(3)
        with ctx_cols[0]:
            if run_meta.get("github_run_url"):
                st.link_button("🔗 CI Run", run_meta["github_run_url"], use_container_width=True)
        with ctx_cols[1]:
            if run_meta.get("commit_sha"):
                sha = run_meta["commit_sha"][:8]
                st.caption(f"**Commit** `{sha}`")
        with ctx_cols[2]:
            if run_meta.get("n_events"):
                st.caption(f"**Events** {run_meta['n_events']:,}")

    st.divider()

    # ── CPU ───────────────────────────────────────────────────────────────────
    st.subheader("🖥️ CPU")
    cpu_cols = st.columns(4)
    cpu_cols[0].metric("Model",          machine_info.get("cpu_model", "N/A"))
    cpu_cols[1].metric("Physical cores", machine_info.get("cpu_physical_cores", "N/A"))
    cpu_cols[2].metric("Logical cores",  machine_info.get("cpu_logical_cores",  "N/A"))

    flags = machine_info.get("cpu_flags", [])
    if flags:
        with st.expander(f"CPU flags ({len(flags)} shown)"):
            st.code(" ".join(flags), language=None)

    st.divider()

    # ── Memory ────────────────────────────────────────────────────────────────
    st.subheader("🧠 Memory")
    mem_cols = st.columns(4)
    ram_total = machine_info.get("ram_total_gb", 0)
    ram_start = machine_info.get("ram_available_gb_start")
    ram_end   = machine_info.get("ram_available_gb_end")
    swap      = machine_info.get("swap_total_gb", 0)

    def _gb(v: float | None) -> str:
        return f"{v:.1f} GB" if v is not None else "N/A"

    mem_cols[0].metric("Total RAM",       _gb(ram_total))
    mem_cols[1].metric(
        "Available (start)",
        _gb(ram_start),
        help="Free RAM measured immediately before the benchmark started.",
    )
    mem_cols[2].metric(
        "Available (end)",
        _gb(ram_end),
        delta=f"{ram_end - ram_start:.1f} GB" if (ram_start is not None and ram_end is not None) else None,
        delta_color="inverse",
        help="Free RAM measured after all benchmark runs completed.",
    )
    mem_cols[3].metric("Swap total",      _gb(swap))

    st.divider()

    # ── System load ───────────────────────────────────────────────────────────
    st.subheader("⚡ System Load")
    load_cols = st.columns(4)

    n_cores     = machine_info.get("cpu_logical_cores") or 1
    l1_start    = machine_info.get("load_avg_1m_start")
    l5_start    = machine_info.get("load_avg_5m_start")
    l1_end      = machine_info.get("load_avg_1m_end")
    l5_end      = machine_info.get("load_avg_5m_end")

    def _load_metric(col: st.delta_generator.DeltaGenerator,
                     label: str,
                     value: float | None,
                     delta: float | None = None,
                     help_text: str = "") -> None:
        if value is None:
            col.metric(label, "N/A", help=help_text)
            return
        pct = value / n_cores * 100
        annotation = ""
        if pct > 80:
            annotation = " ⚠️"
        col.metric(
            label,
            f"{value:.2f}{annotation}",
            delta=f"{delta:.2f}" if delta is not None else None,
            delta_color="inverse",
            help=help_text + (f"\n\n{pct:.0f}% of logical cores." if n_cores else ""),
        )

    _load_metric(load_cols[0], "1-min load (start)", l1_start,
                 help_text="1-minute load average before benchmark.")
    _load_metric(load_cols[1], "5-min load (start)", l5_start,
                 help_text="5-minute load average before benchmark.")
    _load_metric(load_cols[2], "1-min load (end)", l1_end,
                 delta=(l1_end - l1_start) if (l1_end is not None and l1_start is not None) else None,
                 help_text="1-minute load average after benchmark.")
    _load_metric(load_cols[3], "5-min load (end)", l5_end,
                 delta=(l5_end - l5_start) if (l5_end is not None and l5_start is not None) else None,
                 help_text="5-minute load average after benchmark.")

    # Warn if the runner was under significant load, which could skew results
    if l1_start is not None and n_cores and (l1_start / n_cores) > 0.5:
        st.warning(
            f"⚠️ High system load at benchmark start "
            f"({l1_start:.2f} / {n_cores} cores = {l1_start/n_cores*100:.0f}%). "
            "Timing results may be affected by competing workloads."
        )

    st.divider()

    # ── OS / environment ──────────────────────────────────────────────────────
    st.subheader("🐧 Environment")
    env_cols = st.columns(4)
    env_cols[0].metric("OS",        machine_info.get("os",       "N/A"))
    env_cols[1].metric("Kernel",    machine_info.get("kernel",   "N/A"))
    env_cols[2].metric("Hostname",  machine_info.get("hostname", "N/A"))
    env_cols[3].metric("Container", "Yes" if machine_info.get("in_container") else "No")
