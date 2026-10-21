"""Logs tab — per-config status and the log viewer for the selected run.

The configurations render as one row of clickable chips, failures first, and
selecting one shows its extracted error, the k4Bench plugin messages and the
full log, all read from the run's local cache.

What counts as an error or a warning depends on what wrote the log: a ddsim
log is Geant4's (:data:`GEANT4`), a k4run log Gaudi's and its components'
(:data:`GAUDI`). Each page passes the :class:`LogFormat` of its suite.
"""

from __future__ import annotations

import glob
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from k4bench.artifacts import JOBOPTIONS_SUFFIX, LOG_SUFFIX, label_path
from ui_chrome import failed_labels

# Every benchmark log ends with GNU time's verbose resource table, a tab-indented
# block that begins with this line. It's pure noise when hunting a failure cause.
_TIME_TRAILER_RE = re.compile(r"^\tCommand being timed:")

# Lines that signal an error/failure — used both to auto-surface a failure cause
# and for the "Errors" filter. We deliberately do NOT match the bare word
# "exception": it appears in benign prose (e.g. "the exception severity, use
# G4PrimaryTransformer::SetKETolerance()"). Real Geant4 exceptions are captured
# instead via their structured G4Exception blocks (see _filter_by_severity).
_ERROR_RE = re.compile(
    r"traceback \(most recent call last\)"
    r"|\berror\b|\bfatal\b|\babort(?:ed|ing)?\b"
    r"|segmentation fault|core dumped|terminate called|what\(\)\s*:"
    r"|\bassert(?:ion)?\b|runtime_error"
    r"|command exited with non-zero status"
    r"|\bEEEE\b",  # Geant4 fatal/error G4Exception severity marker
    re.IGNORECASE,
)

# Geant4/DD4hep component lines read "<Component> WARN  <msg>".
_WARN_RE = re.compile(r"\bwarn(?:ing)?\b", re.IGNORECASE)

# The k4Bench timing/region plugins log under component names beginning with
# "k4Bench" / "DD4bench" (the prefix varies by release). Anchored at line start so
# the same token inside the echoed ddsim command line is not matched.
_K4BENCH_RE = re.compile(r"^(?:k4bench|dd4bench)\w*", re.IGNORECASE)

# A Gaudi job's messages read "<Component>  <LEVEL> <msg>"; Marlin processors
# wrapped by k4MarlinWrapper print "[ <LEVEL> "<processor>"] <msg>", k4run's own
# logger "[k4run - <LEVEL>] <msg>" and ROOT "Error in <Class::method>: <msg>".
# Only those levels count, never the word "error" in a message's prose (the
# CLD reconstruction prints "[setVariable error]" in a WARN line).
_GAUDI_ERROR_RE = re.compile(
    r"^\S+\s+(?:ERROR|FATAL)\b"
    r"|^\[\s*(?:ERROR|FATAL)\b"
    r"|^\[k4run - (?:ERROR|CRITICAL)\]"
    r"|^(?:Error|Fatal|SysError) in\b"
    r"|(?i:traceback \(most recent call last\)"
    r"|segmentation violation|segmentation fault|core dumped|terminate called|what\(\)\s*:"
    r"|terminated with error|command exited with non-zero status)"
)
_GAUDI_WARN_RE = re.compile(
    r"^\S+\s+(?:WARNING|WARN)\b|^\[\s*WARNING\b|^\[k4run - WARNING\]|^Warning in\b"
)
# What ends a Gaudi job: its first FATAL message, a crash or k4run's own error.
# Marlin processors log ERROR lines that do not stop the job (the CLD tracking
# prints them every few events), so the last error line is a poor guess at why
# a job failed.
_GAUDI_CAUSE_RE = re.compile(
    r"^\S+\s+FATAL\b|^\[\s*FATAL\b|^\[k4run - (?:ERROR|CRITICAL)\]"
    r"|(?i:segmentation violation|segmentation fault|terminate called|what\(\)\s*:"
    r"|terminated with error)"
)
# The component a Gaudi or Marlin message names.
_GAUDI_SOURCE_RE = re.compile(
    r'^\[\s*(?:ERROR|FATAL|WARNING)\s+"(?P<processor>[^"]+)"\]'
    r"|^(?P<component>[^\s\[]\S*)\s+(?:ERROR|FATAL|WARNING|WARN)\b"
)


@dataclass(frozen=True)
class LogFormat:
    """How one kind of log marks its errors and warnings, and what else the
    tab shows next to it."""

    error_re: re.Pattern
    warn_re: re.Pattern
    #: Help of the Severity filter, naming what each class keeps.
    severity_help: str
    #: Whether Geant4's multi-line ``G4Exception`` blocks are kept whole.
    g4_exception_blocks: bool = False
    #: Lines that end the job: the first one names a failure's cause better
    #: than the last error line does. ``None`` reads the last error line.
    cause_re: re.Pattern | None = None
    #: Picks the component a message names: group ``processor`` or ``component``.
    source_re: re.Pattern | None = None
    #: Whether the k4Bench plugins log lines of their own to surface.
    plugin_messages: bool = False
    #: Whether the configuration's ``_joboptions.opts`` is offered.
    joboptions: bool = False


#: ddsim's logs.
GEANT4 = LogFormat(
    error_re=_ERROR_RE,
    warn_re=_WARN_RE,
    severity_help=(
        "Keep only lines of the chosen severity. "
        "Errors = ERROR/FATAL lines, Python tracebacks and Geant4 "
        "G4Exception error blocks; Warnings = WARN lines and G4Exception "
        "warning blocks (whole blocks are kept intact). Matches are found "
        "across the full log, combined with the Filter box, then trimmed to "
        "the last 'Max lines'."
    ),
    g4_exception_blocks=True,
    plugin_messages=True,
)

#: k4run's logs.
GAUDI = LogFormat(
    error_re=_GAUDI_ERROR_RE,
    warn_re=_GAUDI_WARN_RE,
    severity_help=(
        "Keep only lines of the chosen severity. "
        "Errors = ERROR/FATAL messages of Gaudi components and Marlin "
        "processors, ROOT errors, Python tracebacks and crashes; Warnings = "
        "WARNING messages. Matches are found across the full log, combined "
        "with the Filter box, then trimmed to the last 'Max lines'."
    ),
    cause_re=_GAUDI_CAUSE_RE,
    source_re=_GAUDI_SOURCE_RE,
    joboptions=True,
)


@st.cache_data(show_spinner=False)
def _load_log(path: str, mtime: float, size: int) -> tuple[bytes, list[str]]:
    """Read a log once and return ``(raw_bytes, decoded_lines)``.

    Cached on ``path`` + ``mtime`` + ``size`` so repeated reruns (every filter
    keystroke, severity/config switch) don't re-read or re-split a 10k+ line log;
    the entry refreshes when the file changes. The raw bytes feed the download
    button (a faithful copy of the file); the decoded lines feed display and
    filtering — so the file is read exactly once per (re)load.
    """
    raw = Path(path).read_bytes()
    return raw, raw.decode("utf-8", errors="replace").splitlines()


def _log_body(lines: list[str]) -> list[str]:
    """Return *lines* with the trailing ``/usr/bin/time -v`` table removed.

    The ``Command exited with non-zero status N`` line that immediately precedes
    the table is *kept* — it's the one line of the trailer worth seeing.
    """
    for i, ln in enumerate(lines):
        if _TIME_TRAILER_RE.match(ln):
            return lines[:i]
    return lines


def _extract_error_excerpt(
    lines: list[str], max_lines: int = 40, log_format: LogFormat = GEANT4,
) -> str | None:
    """Best-effort isolation of the part of a log that explains a failure.

    Strategy, in order of preference:

    1. The **last** Python / cppyy traceback — captured from its
       ``Traceback (most recent call last):`` through the end of the run output.
       Using the last (not first) traceback surfaces the fatal one in logs that
       recover from earlier tracebacks before ultimately failing.
    2. Otherwise, for a format with a :attr:`LogFormat.cause_re`, a window
       from just before its first match on.
    3. Otherwise, a window centred on the last error-looking line.
    4. Otherwise, the tail of the output.

    Returns ``None`` only for an empty log.
    """
    body = _log_body(lines)
    if not body:
        return None

    tb_starts = [
        i for i, ln in enumerate(body)
        if "traceback (most recent call last)" in ln.lower()
    ]
    if tb_starts:
        i = tb_starts[-1]
        return "\n".join(body[i:][-max_lines:]).strip() or None

    if log_format.cause_re is not None:
        cause = next((i for i, ln in enumerate(body) if log_format.cause_re.search(ln)), None)
        if cause is not None:
            lo = max(0, cause - 6)
            return "\n".join(body[lo:lo + max_lines]).strip() or None

    hits = [i for i, ln in enumerate(body) if log_format.error_re.search(ln)]
    if hits:
        last = hits[-1]
        lo = max(0, last - max_lines + 6)
        hi = min(len(body), last + 6)
        return "\n".join(body[lo:hi]).strip() or None

    return "\n".join(body[-max_lines:]).strip() or None


def _filter_by_severity(
    lines: list[str], want_errors: bool, want_warnings: bool,
    log_format: LogFormat = GEANT4,
) -> list[str]:
    """Keep only error- and/or warning-related lines.

    Geant4 prints structured, multi-line ``G4Exception`` blocks delimited by
    ``… EEEE … G4Exception-START …`` (error/fatal) or ``… WWWW …`` (warning)
    banners. The *whole* block is kept when its class is requested, so the
    message body — not just the banner — survives the filter. Every other line is
    classified per line by its severity token (``ERROR``/``FATAL`` for errors,
    ``WARN`` for warnings), which is why prose mentioning "exception" is not
    swept in.
    """
    if not (want_errors or want_warnings):
        return lines

    def _wanted(cls: str) -> bool:
        return (cls == "error" and want_errors) or (cls == "warning" and want_warnings)

    out: list[str] = []
    block: str | None = None  # severity class while inside a G4Exception block
    blocks = log_format.g4_exception_blocks
    for ln in lines:
        if blocks and "G4Exception-START" in ln:
            block = "error" if "EEEE" in ln else "warning"
            if _wanted(block):
                out.append(ln)
            continue
        if blocks and "G4Exception-END" in ln:
            if block is not None and _wanted(block):
                out.append(ln)
            block = None
            continue
        if block is not None:
            if _wanted(block):
                out.append(ln)
            continue
        if want_errors and log_format.error_re.search(ln):
            out.append(ln)
        elif want_warnings and log_format.warn_re.search(ln):
            out.append(ln)
    return out


def _autoscroll_log_to_bottom(nonce: str) -> None:
    """Pin the log explorer's fixed-height container to its **bottom**.

    ``st.code`` always renders pinned to the top and Streamlit offers no "scroll
    to end", so a long log would open on its first line instead of the failure +
    run tail at the end. This injects a one-off script that finds the log's
    scroll container (via a hidden sentinel placed just after the code) and pins
    it to the bottom — on load, and again via a ``ResizeObserver`` the moment the
    container gains size, i.e. when the initially-hidden Logs tab is revealed.

    *nonce* (a digest of the current view) is embedded so the iframe re-runs on
    every config/filter change and re-pins the freshly rendered log.
    """
    # st.iframe embeds an HTML string via the iframe's srcdoc (scripts run, same
    # as the deprecated st.components.v1.html). height must be ≥ 1; 1px is
    # effectively invisible for this script-only injector.
    st.iframe(
        f"""
        <script>
        // view:{nonce}
        (function () {{
          const doc = window.parent.document;
          function findBox() {{
            const m = doc.querySelector('.k4-log-end');
            if (!m) return null;
            let el = m.parentElement;
            while (el && el !== doc.body) {{
              const oy = getComputedStyle(el).overflowY;
              if (oy === 'auto' || oy === 'scroll' || oy === 'overlay') return el;
              el = el.parentElement;
            }}
            return null;
          }}
          let attempts = 0;
          function setup() {{
            const box = findBox();
            if (!box) {{ if (attempts++ < 50) setTimeout(setup, 100); return; }}
            const pin = function () {{ box.scrollTop = box.scrollHeight; }};
            pin();
            setTimeout(pin, 80);
            setTimeout(pin, 350);
            // Replace any prior observer (a previous rerun's iframe may have left
            // a now-dead reference on this element) so exactly one stays active.
            try {{ if (box.__k4Observer) box.__k4Observer.disconnect(); }} catch (e) {{}}
            try {{
              box.__k4Observer = new ResizeObserver(pin);
              box.__k4Observer.observe(box);
            }} catch (e) {{}}
          }}
          setup();
        }})();
        </script>
        """,
        height=1,
    )


def _sources(lines: list[str], log_format: LogFormat) -> pd.DataFrame:
    """How many error and warning messages each component logged, most
    errors first."""
    counts: dict[str, list[int]] = {}
    for ln in lines:
        is_error = bool(log_format.error_re.search(ln))
        if not (is_error or log_format.warn_re.search(ln)):
            continue
        match = log_format.source_re.match(ln)
        if match is None:
            continue
        name = match.group("processor") or match.group("component")
        counts.setdefault(name, [0, 0])[0 if is_error else 1] += 1
    frame = pd.DataFrame(
        [(name, errors, warnings) for name, (errors, warnings) in counts.items()],
        columns=["Component", "Errors", "Warnings"],
    )
    return frame.sort_values(["Errors", "Warnings", "Component"],
                             ascending=[False, False, True], ignore_index=True)


def _cause_source(excerpt: str, log_format: LogFormat) -> str | None:
    """The component the excerpt's first ending line names, if any."""
    if log_format.cause_re is None or log_format.source_re is None:
        return None
    for ln in excerpt.splitlines():
        if log_format.cause_re.search(ln) and (match := log_format.source_re.match(ln)):
            return match.group("processor") or match.group("component")
    return None


def _render_joboptions(run_dir: Path, label: str) -> None:
    """The configuration's resolved job options, filterable, in an expander."""
    path = label_path(run_dir, label, JOBOPTIONS_SUFFIX)
    try:
        stat = path.stat()
    except OSError:
        return
    raw, lines = _load_log(str(path), stat.st_mtime, stat.st_size)
    with st.expander(f"Job options · `{path.name}`"):
        st.caption(
            "Every property the job ran with, as Gaudi resolved it after the "
            "options files ran: one `Component.Property = value;` per line."
        )
        f_search, f_dl = st.columns([5, 1.5], vertical_alignment="bottom")
        query = f_search.text_input(
            "Filter properties", key="logs_joboptions_filter",
            placeholder="🔎 component or property, case-insensitive…",
        )
        f_dl.download_button(
            "⬇ Download", raw, file_name=path.name, mime="text/plain",
            use_container_width=True, key="logs_joboptions_download",
        )
        shown = [ln for ln in lines if query.lower() in ln.lower()] if query else lines
        with st.container(height=420):
            st.code("\n".join(shown) or "— no matching properties —", language="text")
        st.caption(f"{len(shown):,} of {len(lines):,} lines · {stat.st_size / 1024:,.0f} KB")


def _log_files(label: str) -> tuple[str, ...]:
    """What this view reads of one configuration: its log and job options."""
    name = glob.escape(label)
    return f"{name}{LOG_SUFFIX}", f"{name}{JOBOPTIONS_SUFFIX}"


def render(
    results: "pd.DataFrame | None",
    run_files: Callable[[tuple[str, ...]], str] | None,
    run_meta: dict | None = None,
    log_format: LogFormat = GEANT4,
) -> None:
    """Status overview and log viewer for the selected run, merged into one flow.

    Failure-first: the configurations render as a single row of clickable chips
    (failures sorted first and shown in red), and selecting one loads its logs
    right below — its extracted error (if it failed), the k4Bench plugin messages,
    and the full ddsim log. There is no separate status table or config dropdown;
    the chips *are* both the status overview and the selector.

    *run_files* gives the run's directory once it holds the files of the
    given names (``data.RunData.directory``): only the selected
    configuration's are asked for, so a remote run's other logs are never
    downloaded for this view.
    """
    if results is None or "label" not in results.columns:
        st.info("No run data available for the current selection.")
        return

    meta = run_meta or {}
    labels = sorted(results["label"].astype(str).unique())
    failed = failed_labels(results)
    failed_set = set(failed)
    rc_by_label = (
        dict(zip(results["label"].astype(str), results["returncode"]))
        if "returncode" in results.columns
        else {}
    )
    if not labels:
        st.info("No configurations found in this run.")
        return

    # ── Run context (release / date / commit / events) ──────────────────────────
    bits: list[str] = []
    if meta.get("k4h_release"):
        bits.append(f"release `{meta['k4h_release']}`")
    run_date = meta.get("run_date")
    if run_date is not None and pd.notna(run_date):
        bits.append(f"run {pd.to_datetime(run_date):%Y-%m-%d}")
    if meta.get("commit_sha"):
        bits.append(f"commit `{str(meta['commit_sha'])[:8]}`")
    if meta.get("n_events"):
        bits.append(f"{int(meta['n_events']):,} events/config")
    if bits:
        st.caption(" · ".join(bits))

    # ── Headline verdict ────────────────────────────────────────────────────────
    n = len(labels)
    plural = "s" if n != 1 else ""
    run_url = meta.get("github_run_url")
    if failed:
        link = f"  ·  [Open CI run logs ↗]({run_url})" if run_url else ""
        st.error(
            f"**{len(failed)} of {n} configuration{plural} failed.** "
            f"Pick a red configuration below to see why.{link}",
            icon="🔴",
        )
    else:
        st.success(
            f"**All {n} configuration{plural} completed successfully.**", icon="✅"
        )

    # ── Configurations: status overview + selector in one ───────────────────────
    # Failed configs sort first and render red; click any chip to load its logs
    # below. This merges the old status grid with the separate config dropdown.
    ordered = sorted(labels, key=lambda lbl: (lbl not in failed_set, lbl))
    default_label = failed[0] if failed else labels[0]

    def _fmt_config(lbl: str) -> str:
        if lbl in failed_set:
            rc = rc_by_label.get(lbl)
            rc_txt = f" · rc {int(rc)}" if pd.notna(rc) else ""
            return f":red[**✕ {lbl}{rc_txt}**]"
        return lbl

    st.markdown("##### Configurations")
    selected = st.pills(
        "Configurations", ordered, selection_mode="single",
        default=default_label, format_func=_fmt_config,
        label_visibility="collapsed",
        help="Click a configuration to view its logs below. Failed configs sort "
             "first and show in red; the highlighted chip is the one on view.",
    )
    chosen = selected or default_label

    # ── Logs for the selected configuration ─────────────────────────────────────
    if run_files is None:
        st.caption("Logs are only available when viewing cached or remote runs.")
        return

    run_dir = run_files(_log_files(chosen))
    log_path = label_path(Path(run_dir), chosen, LOG_SUFFIX)
    try:
        stat = log_path.stat()
    except OSError:
        st.caption(f"No log file found for `{chosen}`.")
        return

    raw, all_lines = _load_log(str(log_path), stat.st_mtime, stat.st_size)

    # For a failed config, surface the extracted cause up front.
    if chosen in failed_set:
        rc = rc_by_label.get(chosen)
        rc_txt = f"exit code {int(rc)}" if pd.notna(rc) else "no exit code"
        excerpt = _extract_error_excerpt(all_lines, log_format=log_format)
        if excerpt:
            source = _cause_source(excerpt, log_format)
            raised = f", raised by `{source}`" if source else ""
            st.error(f"**Likely cause of failure** ({rc_txt}{raised}):")
            st.code(excerpt, language="log")
        else:
            st.error(
                f"This configuration failed ({rc_txt}); no error message could be "
                "isolated — see the full log below."
            )

    # ── k4Bench plugin messages (always surfaced for the selected config) ───────
    # The timing/region plugins are the bench-specific instrumentation; pulling
    # their lines out confirms they ran (and where they wrote their JSON) without
    # scrolling the full ddsim log.
    if log_format.plugin_messages:
        k4_lines = [ln for ln in all_lines if _K4BENCH_RE.match(ln)]
        with st.container(border=True):
            st.markdown("##### 🔧 k4Bench plugin messages")
            if k4_lines:
                st.code("\n".join(k4_lines), language="log")
            else:
                st.caption(
                    "No k4Bench plugin messages in this log — the timing/region "
                    "plugins may not have run (e.g. the config failed before events)."
                )

    # ── Errors and warnings by the component that logged them ───────────────────
    if log_format.source_re is not None:
        sources = _sources(_log_body(all_lines), log_format)
        with st.container(border=True):
            st.markdown("##### Errors and warnings by component")
            if sources.empty:
                st.caption("No component logged an error or a warning.")
            else:
                st.caption(
                    "Messages each component logged at ERROR/FATAL or WARNING "
                    "level. Not every error stops the job: the Severity filter "
                    "below shows them in context."
                )
                st.dataframe(sources, hide_index=True, width="content")

    # ── Full log ────────────────────────────────────────────────────────────────
    st.markdown("##### Full log")
    # Toolbar: search · line cap · severity · download (one aligned row). Labels
    # stay visible (not collapsed) so each control keeps its hover "?" help icon —
    # Streamlit drops the help icon together with a collapsed label.
    f_search, f_cap, f_sev, f_dl = st.columns(
        [5, 1.7, 2.3, 1.5], vertical_alignment="bottom"
    )
    query = f_search.text_input(
        "Filter",
        placeholder="🔎 substring, case-insensitive…",
        help="Show only lines containing this text (case-insensitive). Applied "
             "on top of the Severity filter, and still bounded by 'Max lines'.",
    )
    # Cap counted from the *bottom* of the log (where failures surface) so the
    # view stays responsive on 10k+ line logs.
    max_lines = f_cap.selectbox(
        "Max lines", [500, 1000, 2000, 5000, 20000], index=0,
        format_func=lambda n: f"last {n:,} lines",
        help="How many lines to render, counted from the bottom of the log "
             "(where failures and the run tail are). When a Severity or Filter "
             "is active this caps how many of the matching lines are shown. "
             "Download always returns the complete log.",
    )
    # Severity filter. Defaults to "All": for a failed config the cause is already
    # surfaced in the extracted-error panel above, so the full log keeps full
    # context here rather than silently hiding everything but error lines.
    severity = f_sev.selectbox(
        "Severity", ["All", "Errors", "Warnings", "Errors + warnings"],
        index=0,
        help=log_format.severity_help,
    )
    show_errors = severity in ("Errors", "Errors + warnings")
    show_warnings = severity in ("Warnings", "Errors + warnings")

    lines = _filter_by_severity(all_lines, show_errors, show_warnings, log_format)
    if query:
        q = query.lower()
        lines = [ln for ln in lines if q in ln.lower()]
    filtering = bool(show_errors or show_warnings or query)
    total = len(lines)
    truncated = total > max_lines
    if truncated:
        lines = lines[-max_lines:]

    f_dl.download_button(
        "⬇ Download", raw,
        file_name=log_path.name, mime="text/plain",
        use_container_width=True,
    )

    # ── Log (syntax-highlighted, opens scrolled to the bottom) ──────────────────
    body = "\n".join(lines) if lines else "— no matching lines —"
    with st.container(height=520):
        st.code(body, language="log")
        # Sentinel the autoscroll script walks up from to find this scroll box.
        st.markdown(
            '<span class="k4-log-end" style="display:none"></span>',
            unsafe_allow_html=True,
        )
    # A deterministic digest (not Python's per-process-salted hash()) so it's
    # stable across restarts; hex-only, so a query containing e.g. "</script>"
    # can't break the injected component. Only needs to change when the view does.
    nonce = hashlib.sha256(
        repr((chosen, total, max_lines, query, show_errors, show_warnings)).encode()
    ).hexdigest()[:16]
    _autoscroll_log_to_bottom(nonce)

    # ── File / match summary (below the log) ────────────────────────────────────
    cap = (
        f"`{log_path.name}` · {len(all_lines):,} lines · "
        f"{stat.st_size / 1024:,.0f} KB"
    )
    if filtering:
        cap += f"  —  {total:,} match(es)"
        if truncated:
            cap += f", showing last {max_lines:,}"
    elif truncated:
        cap += f"  —  showing last {max_lines:,} lines"
    st.caption(cap)

    if log_format.joboptions:
        _render_joboptions(Path(run_dir), chosen)
