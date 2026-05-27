"""Machine Info tab — displays hardware and OS details for the selected run."""

from __future__ import annotations

import streamlit as st

# CPU flags relevant to simulation / floating-point heavy workloads.
# Presence of these affects vectorisation and therefore benchmark performance.
_KEY_FLAGS: dict[str, str] = {
    "avx512f": "AVX-512",
    "avx2":    "AVX2",
    "avx":     "AVX",
    "fma":     "FMA",
    "sse4_2":  "SSE4.2",
}


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

    # ── Extract values used across multiple sections ───────────────────────────
    cpu_physical = machine_info.get("cpu_physical_cores") or 0
    cpu_logical  = machine_info.get("cpu_logical_cores")  or 0
    governor     = machine_info.get("cpu_governor")
    flags        = machine_info.get("cpu_flags", [])
    l1_start     = machine_info.get("load_avg_1m_start")
    l5_start     = machine_info.get("load_avg_5m_start")
    l1_end       = machine_info.get("load_avg_1m_end")
    l5_end       = machine_info.get("load_avg_5m_end")
    ram_total    = machine_info.get("ram_total_gb")
    ram_start    = machine_info.get("ram_available_gb_start")
    ram_end      = machine_info.get("ram_available_gb_end")
    swap_used       = machine_info.get("swap_used_gb_start")
    throttle_events = machine_info.get("thermal_throttle_events")
    freq_start      = machine_info.get("cpu_freq_mhz_start")
    freq_end        = machine_info.get("cpu_freq_mhz_end")

    # ── CPU ───────────────────────────────────────────────────────────────────
    st.subheader("🖥️ CPU")

    # Row 1: hardware characteristics
    cpu_cols = st.columns(6)
    cpu_cols[0].metric("Model",          machine_info.get("cpu_model", "N/A"))
    cpu_cols[1].metric("Physical cores", cpu_physical or "N/A")
    cpu_cols[2].metric("Logical cores",  cpu_logical  or "N/A")

    if cpu_physical > 0 and cpu_logical > 0:
        ht_on = cpu_logical > cpu_physical
        cpu_cols[3].metric(
            "Hyperthreading",
            "On ⚠️" if ht_on else "Off ✅",
            help=(
                "Hyperthreading is enabled. The benchmark thread shares execution units "
                "(caches, branch predictor, execution ports) with its HT sibling — "
                "this can add variance to single-core timings even when no other processes are running."
                if ht_on else
                "Hyperthreading is disabled — the benchmark thread has exclusive use of its physical core."
            ),
        )
    else:
        cpu_cols[3].metric("Hyperthreading", "N/A")

    cpu_cols[4].metric(
        "CPU freq",
        f"{freq_start:.0f} MHz" if freq_start is not None else "N/A",
        delta=(
            f"{freq_end - freq_start:.0f} MHz"
            if (freq_start is not None and freq_end is not None) else None
        ),
        delta_color="normal",
        help=(
            "Kernel-reported CPU frequency before the benchmark (delta = change by end). "
            "A drop indicates the CPU slowed down — possible throttling due to heat or governor policy. "
            "Note: this is a snapshot reading and may not reflect exact effective frequency."
        ),
    )

    cpu_cols[5].metric(
        "CPU governor",
        governor if governor else "N/A",
        help=(
            "The Linux CPU frequency scaling governor active during the benchmark. "
            "'performance' locks the CPU at max frequency — best for reproducible results. "
            "'powersave' or 'schedutil' may throttle the clock and inflate timings."
            if governor else
            "Not available — likely running inside a container without cpufreq access."
        ),
    )

    # Row 2: SIMD features relevant to simulation workloads
    if flags:
        flag_set  = set(flags)
        feat_cols = st.columns(len(_KEY_FLAGS))
        for col, (flag_key, label) in zip(feat_cols, _KEY_FLAGS.items(), strict=True):
            present = flag_key in flag_set
            col.metric(
                label,
                "✅" if present else "—",
                help=f"{'Supported' if present else 'Not supported'} by this CPU.",
            )
        with st.expander(f"All CPU flags ({len(flags)} total)"):
            st.code(" ".join(flags), language=None)

    st.divider()

    # ── Memory ────────────────────────────────────────────────────────────────
    st.subheader("🧠 Memory")
    mem_cols = st.columns(5)

    def _gb(v: float | None) -> str:
        return f"{v:.1f} GB" if v is not None else "N/A"

    # Memory pressure verdict
    if ram_start is None or not ram_total:
        mem_cols[0].metric("Memory pressure", "Unknown",
                           help="RAM availability was not recorded for this run.")
    else:
        avail_pct = ram_start / ram_total * 100
        if avail_pct >= 50:
            mem_cols[0].metric("Memory pressure", "✅ None",
                               help=f"{avail_pct:.0f}% of RAM was free — no memory pressure.")
        elif avail_pct >= 25:
            mem_cols[0].metric("Memory pressure", "🟡 Low",
                               help=f"{avail_pct:.0f}% of RAM was free — adequate, but some pages may be reclaimed under load.")
        elif avail_pct >= 10:
            mem_cols[0].metric("Memory pressure", "🟠 Moderate",
                               help=f"Only {avail_pct:.0f}% of RAM was free — OS may have been reclaiming pages, which can affect timings.")
        else:
            mem_cols[0].metric("Memory pressure", "🔴 High",
                               help=f"Only {avail_pct:.0f}% of RAM was free — system was likely swapping. Timing results are unreliable.")

    mem_cols[1].metric("Total RAM", _gb(ram_total))
    mem_cols[2].metric(
        "Available (start)",
        _gb(ram_start),
        help="Free RAM measured immediately before the benchmark started.",
    )
    mem_cols[3].metric(
        "Available (end)",
        _gb(ram_end),
        delta=f"{ram_end - ram_start:.1f} GB" if (ram_start is not None and ram_end is not None) else None,
        delta_color="off",
        help="Free RAM measured after all benchmark runs completed. "
             "A drop here is normal — the benchmark consumed memory during the run.",
    )
    swap_label = (
        f"{swap_used:.2f} GB {'⚠️' if swap_used > 0 else ''}"
        if swap_used is not None else "N/A"
    )
    mem_cols[4].metric(
        "Swap in use",
        swap_label,
        help="Swap actively in use before the benchmark. "
             "Any non-zero value means the OS was paging to disk — timings will be inflated.",
    )

    st.divider()

    # ── System load ───────────────────────────────────────────────────────────
    st.subheader("⚡ System Load")

    _LOAD_HELP = (
        "Linux load average — counts processes actively running or waiting for a CPU core. "
        "Since the benchmark is single-core, values ≥ 1.0 indicate competing processes "
        "that may have caused context-switches and inflated timings."
    )

    load_cols = st.columns(6)

    if l1_start is None:
        load_cols[0].metric("Run reliability", "Unknown",
                            help="No load average was recorded before this run.")
    elif l1_start < 0.5:
        load_cols[0].metric("Run reliability", "✅ Clean",
                            help="Machine was idle before the benchmark started.")
    elif l1_start < 1.0:
        load_cols[0].metric("Run reliability", "🟡 Probably fine",
                            help="Some background activity was present, but below one full competing process.")
    elif l1_start < 2.0:
        load_cols[0].metric("Run reliability", "🟠 Caution",
                            help="At least one other process was competing for CPU — timings may be inflated.")
    else:
        load_cols[0].metric("Run reliability", "🔴 Unreliable",
                            help="Multiple processes were competing for CPU — timings are likely skewed.")

    if throttle_events is None:
        load_cols[1].metric("Thermal throttling", "N/A",
                            help="Throttle counters not available — likely running inside a container.")
    elif throttle_events == 0:
        load_cols[1].metric("Thermal throttling", "✅ None",
                            help="No thermal throttle events were recorded during the benchmark. "
                                 "The CPU was not forced to reduce its clock speed due to heat.")
    else:
        load_cols[1].metric("Thermal throttling", "⚠️ Detected",
                            help="The CPU was thermally throttled during the benchmark — it reduced its "
                                 "clock speed due to heat. Timings may be inflated and less reproducible. "
                                 "(Note: the raw event count is not shown as kernel counters increment "
                                 "per-core and can overcount a single thermal incident.)")

    def _fmt(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    load_cols[2].metric("Load 1-min (start)", _fmt(l1_start), help=_LOAD_HELP)
    load_cols[3].metric("Load 5-min (start)", _fmt(l5_start), help=_LOAD_HELP)
    load_cols[4].metric("Load 1-min (end)",   _fmt(l1_end),   help=_LOAD_HELP)
    load_cols[5].metric("Load 5-min (end)",   _fmt(l5_end),   help=_LOAD_HELP)

    st.divider()

    # ── OS / environment ──────────────────────────────────────────────────────
    st.subheader("🐧 Environment")
    env_cols = st.columns(4)
    env_cols[0].metric("OS",        machine_info.get("os",       "N/A"))
    env_cols[1].metric("Kernel",    machine_info.get("kernel",   "N/A"))
    env_cols[2].metric("Hostname",  machine_info.get("hostname", "N/A"))
    env_cols[3].metric("Container", "Yes" if machine_info.get("in_container") else "No")
