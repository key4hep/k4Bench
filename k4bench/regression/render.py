"""Render a :class:`~k4bench.regression.models.NightlyReport` as markdown,
self-contained HTML (for the e-group email) and JSON (for the EOS artifact the
dashboard reads back).

Ordering everywhere: hard failures first, then confirmed regressions (any
direction — "regression" here means any confirmed step beyond the baseline,
not just a bad one; none of this is judged good or bad), then a short
WATCH/OK summary — never a row per OK metric.
"""

from __future__ import annotations

import dataclasses
import html
import math
import re
from datetime import date
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from k4bench.blame.models import BlameEntry, BlameReport
from k4bench.regression.models import (
    Direction,
    MetricVerdict,
    NightlyReport,
    RunGroupReport,
    Severity,
)

#: Badge vocabulary, matching the dashboard's (✅/🔴/⚠️/➖/❔). A confirmed
#: regression gets one badge regardless of direction — red for attention, not
#: as a good/bad judgment: the report does not claim to know whether faster
#: or slower is good news, only that it's worth a look.
_BADGES = {
    Severity.CONFIRMED: "🔴 Regression",
    Severity.WATCH: "⚠️ Watch",
    Severity.FAILURE: "❌ Failure",
}


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.4g}"


def _fmt_pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:+.1%}"


def _metric_name(v: MetricVerdict) -> str:
    return f"{v.metric} [{v.sub_detector}]" if v.sub_detector else v.metric


def _badge(v: MetricVerdict) -> str:
    return _BADGES.get(v.severity, v.severity.value)


def _status_cell(v: MetricVerdict) -> str:
    """The report table's status text: the severity badge, plus a repeat
    marker when a later night of the release re-confirms a change — so a
    reader of consecutive nightly emails can tell "still present in this
    release" from fresh news."""
    badge = _badge(v)
    if v.first_confirmed_run_id and v.first_confirmed_run_id != v.run_id:
        return f"{badge} (repeat — first confirmed {v.first_confirmed_run_id})"
    return badge


def _detector_badge(groups: list[RunGroupReport]) -> str:
    """One glance-able status emoji for a whole detector (which can span
    several (platform, sample) run groups) — worst-first, matching the
    severity ordering everywhere else in the report."""
    if any(g.failures or g.job_failures for g in groups):
        return "❌"
    if any(g.regressions for g in groups):
        return "🔴"
    if any(g.watches for g in groups):
        return "⚠️"
    if all(not g.verdicts and g.notes for g in groups):
        return "❔"
    return "✅"


def _table_rows(group: RunGroupReport) -> list[MetricVerdict]:
    """The verdicts that get their own table row, in display order."""
    return group.failures + group.regressions


def _quiet_summary(group: RunGroupReport) -> str:
    n_watch, n_ok = len(group.watches), sum(
        1 for v in group.verdicts if v.severity is Severity.OK
    )
    n_unknown = sum(1 for v in group.verdicts if v.severity is Severity.UNKNOWN)
    parts = [f"{n_ok} OK"]
    if n_watch:
        parts.append(f"{n_watch} on watch (unconfirmed — flagged once, awaiting the next night)")
    if n_unknown:
        parts.append(f"{n_unknown} with insufficient history")
    return ", ".join(parts)


#: Recognized generator/beam/particle tokens for :func:`_pretty_sample`. Any
#: sample name that doesn't match one of the two known layouts below (or that
#: uses a token not listed here) falls back to the raw directory name
#: unchanged, so an unrecognized future sample degrades gracefully instead of
#: producing garbled physics notation.
_GENERATOR_LABELS = {"p8": "Pythia8", "p6": "Pythia6"}
_BEAM_LABELS = {"ee": "e⁺e⁻", "pp": "pp", "ep": "ep"}
_PARTICLE_LABELS = {
    "e-": "e⁻", "e+": "e⁺", "mu-": "μ⁻", "mu+": "μ⁺", "gamma": "γ",
    "pi+": "π⁺", "pi-": "π⁻", "pi0": "π⁰", "proton": "p", "kaon+": "K⁺", "kaon-": "K⁻",
}
#: A two-letter-plus final state after a capitalized boson reads as a decay,
#: e.g. "Zbb" -> "Z → bb"; anything else (e.g. "WW", "ZH", "qq") is left as-is
#: rather than guessed at.
_PROCESS_SPLIT_RE = re.compile(r"^([A-Z])([a-z]{2,})$")


def _pretty_sample(sample: str) -> str:
    """Human-readable label for an EOS sample directory name.

    Covers the two naming layouts currently produced by the benchmark:
    single-particle guns (``single_{particle}_{energy}``) and generator
    samples (``{gen}_{beams}_{process}_ecm{energy}``, e.g.
    ``p8_ee_Zbb_ecm91``). Anything else is returned unchanged.
    """
    tokens = sample.split("_")
    if (
        len(tokens) == 3 and tokens[0] == "single"
        and re.fullmatch(r"\d+(\.\d+)?(GeV|MeV|TeV)", tokens[2])
    ):
        particle = _PARTICLE_LABELS.get(tokens[1], tokens[1])
        return f"Single {particle} · {tokens[2]}"

    if len(tokens) == 4 and re.fullmatch(r"ecm\d+(\.\d+)?", tokens[3]):
        gen = _GENERATOR_LABELS.get(tokens[0], tokens[0])
        beams = _BEAM_LABELS.get(tokens[1], tokens[1])
        process = tokens[2]
        if m := _PROCESS_SPLIT_RE.match(process):
            process = f"{m.group(1)} → {m.group(2)}"
        ecm = tokens[3].removeprefix("ecm")
        if ecm.endswith(".0"):
            ecm = ecm[:-2]
        return f"{gen}: {beams} → {process} ({ecm} GeV)"

    return sample


#: LCG/Spack-style platform triplet vocabulary for :func:`_pretty_platform`.
_OS_LABELS = {"almalinux": "AlmaLinux", "centos": "CentOS", "ubuntu": "Ubuntu"}
_COMPILER_LABELS = {"gcc": "GCC", "clang": "Clang", "icc": "ICC"}
_BUILD_TYPE_LABELS = {"opt": "optimized", "dbg": "debug", "reldbg": "release+debug"}
_VERSIONED_TOKEN_RE = re.compile(r"^([a-zA-Z]+)(\d.*)$")


def _pretty_platform(platform: str) -> str:
    """Human-readable label for an ``{arch}-{os}{ver}-{compiler}{ver}-{type}``
    platform string, e.g. ``x86_64-almalinux9-gcc14.2.0-opt`` ->
    ``AlmaLinux 9 · GCC 14.2.0 (optimized)``. Falls back to the raw string for
    anything that doesn't match this exact 4-part layout."""
    parts = platform.split("-")
    if len(parts) != 4:
        return platform
    _arch, os_part, compiler_part, build = parts
    os_m = _VERSIONED_TOKEN_RE.match(os_part)
    compiler_m = _VERSIONED_TOKEN_RE.match(compiler_part)
    if not os_m or not compiler_m:
        return platform
    os_name = _OS_LABELS.get(os_m.group(1).lower(), os_m.group(1).capitalize())
    compiler_name = _COMPILER_LABELS.get(compiler_m.group(1).lower(), compiler_m.group(1).upper())
    build_name = _BUILD_TYPE_LABELS.get(build, build)
    return f"{os_name} {os_m.group(2)} · {compiler_name} {compiler_m.group(2)} ({build_name})"


def _group_title(group: RunGroupReport) -> str:
    return f"{_pretty_sample(group.sample)} · {_pretty_platform(group.platform)}"


def _dashboard_link(dashboard_url: str, **params: str) -> str:
    """*dashboard_url* with *params* merged into its query string (overriding
    any of the same name already present), so it works whether the caller
    passed a bare base URL or one that already carries its own query —
    e.g. a deep link to one run group's Regressions view."""
    split = urlsplit(dashboard_url)
    query = dict(parse_qsl(split.query))
    query.update(params)
    return urlunsplit(split._replace(query=urlencode(query)))


def _group_links(
    dashboard_url: str | None, group: RunGroupReport, report_night: str
) -> list[tuple[str, str]]:
    """``(text, href)`` dashboard links for one run group. Both targets are
    scoped by the full ``(detector, platform, sample)`` triple: the Regressions
    tab reads exactly that scope from the sidebar, the same way Run Trends
    does, so a detector-only link could land on the wrong sample.

    The Regressions link also pins the release (``stack``) and the exact report
    night (``report``): a release is routinely re-benchmarked, and although
    every night of it is judged against the same baseline, nights can still
    differ (a WATCH night preceding the confirmation, a marginal OK night, or
    a report predating the release-grouped engine) — pinning guarantees the
    link opens the exact report the email described. ``stack`` is omitted when
    the group carries no release (a stale/missing-run group)."""
    if not dashboard_url:
        return []
    scope = dict(detector=group.detector, platform=group.platform, sample=group.sample)
    regr = dict(scope)
    if group.k4h_release:
        regr["stack"] = group.k4h_release
    if report_night:
        regr["report"] = report_night
    return [
        ("Regressions", _dashboard_link(dashboard_url, tab="Regressions", **regr)),
        ("Run Trends", _dashboard_link(dashboard_url, tab="Run Trends", **scope)),
    ]


def _footer_year(report: NightlyReport) -> str:
    """The report's own year (from ``generated_at``) rather than wall-clock
    time, so re-rendering an old report doesn't backdate its copyright line."""
    year = report.generated_at[:4]
    return year if year.isdigit() else str(date.today().year)


#: Attribution footer, mirroring the dashboard's own CERN/FCC banner
#: (:func:`dashboard.ui_chrome._render_footer`) so the two surfaces match.
_FCC_URL = "https://fcc.web.cern.ch/"
_CONTACT_EMAIL = "jbeirer@cern.ch"
_CONTACT_NAME = "Joshua Falco Beirer"


def _markdown_footer(report: NightlyReport) -> str:
    return (
        "---\n\n"
        f"⚛️ **© {_footer_year(report)} CERN** · For the benefit of the "
        f"[FCC project]({_FCC_URL})  \n"
        f"Created by **{_CONTACT_NAME}** (CERN) — questions to {_CONTACT_EMAIL}"
    )


def _html_footer(report: NightlyReport) -> str:
    return (
        '<hr style="border:none;border-top:1px solid #e5e5e5;margin:28px 0 14px;">'
        '<p style="text-align:center;color:#9a9a9a;font-size:12px;line-height:1.7;">'
        '<span style="font-size:1.3em;">⚛️</span><br>'
        f'<strong style="color:#666;">© {_footer_year(report)} CERN</strong> · '
        f'For the benefit of the <a href="{_FCC_URL}" '
        'style="color:#5b9bd5;text-decoration:none;font-weight:600;">FCC project</a><br>'
        f'Created by <strong style="color:#666;">{_CONTACT_NAME}</strong> (CERN) — '
        f'questions to <a href="mailto:{_CONTACT_EMAIL}" '
        f'style="color:#5b9bd5;text-decoration:none;">{_CONTACT_EMAIL}</a>'
        "</p>"
    )


# ── Blame (model-ranked candidate PRs) ───────────────────────────────────────
#
# The regression report and its email never depend on blame: it is a separate,
# best-effort sidecar (``blame.json``), joined back to a verdict only when the
# file is present *and* its ranking stage actually judged the candidates. The
# framing is deliberate — "a lead for a human, not evidence" — mirroring the
# dashboard's candidate ledger and the repo's "no evidence ⇒ no verdict" culture:
# a confident wrong culprit is worse than none.

#: Top candidates surfaced under a regression in the email — a short lead; the
#: full ranked ledger lives on the dashboard's Regressions tab.
_MAX_EMAIL_CANDIDATES = 2


def _ranked_candidates(entry: BlameEntry) -> list:
    """*entry*'s candidates the ranking stage actually judged (a non-zero score
    or a description), worst-first, capped — empty when nothing was ranked."""
    ranked = [c for c in entry.candidates if c.score or c.description]
    return ranked[:_MAX_EMAIL_CANDIDATES]


def _blame_for(blame: BlameReport | None, v: MetricVerdict) -> list:
    """The ranked candidates attributing verdict *v*, or ``[]`` when blame is
    absent, has no entry for *v*, or left it unranked."""
    if blame is None:
        return []
    entry = blame.entry_for(v)
    return _ranked_candidates(entry) if entry is not None else []


def _blame_markdown_lines(group: RunGroupReport, blame: BlameReport | None) -> list[str]:
    """Suggested-cause bullets for a group's confirmed regressions, or ``[]``."""
    lines: list[str] = []
    for v in group.regressions:
        ranked = _blame_for(blame, v)
        if not ranked:
            continue
        lead = ", ".join(
            f"[`{c.repo}#{c.number}`]({c.url}) ({c.score:.0f}%)"
            + (f" — {c.description}" if c.description else "")
            for c in ranked
        )
        lines.append(f"- _{_metric_name(v)}_ — most likely: {lead}")
    if lines:
        lines.insert(0, "**Suggested causes** — a lead for a human, not evidence:")
    return lines


def _blame_html_block(group: RunGroupReport, blame: BlameReport | None) -> str:
    """Suggested-cause list for a group's confirmed regressions, or ``""``.

    Everything read from ``blame.json`` is escaped on the way into markup: the
    reason is model output, and even the PR URL is file content rather than
    something this process fetched from GitHub itself — neither may inject
    markup into the email body."""
    items: list[str] = []
    for v in group.regressions:
        ranked = _blame_for(blame, v)
        if not ranked:
            continue
        leads = "; ".join(
            f'<a href="{html.escape(c.url, quote=True)}" style="color:#5b9bd5;text-decoration:none;">'
            f"{c.repo}#{c.number}</a> ({c.score:.0f}%)"
            + (f" — {html.escape(c.description)}" if c.description else "")
            for c in ranked
        )
        items.append(
            f'<li><strong>{_metric_name(v)}</strong> — most likely: {leads}</li>'
        )
    if not items:
        return ""
    return (
        '<p style="color:#555;font-size:13px;margin:8px 0 2px;">'
        "Suggested causes — a lead for a human, not evidence:</p>"
        '<ul style="font-size:13px;color:#333;margin:0 0 10px;padding-left:18px;">'
        + "".join(items)
        + "</ul>"
    )


# ── Markdown ──────────────────────────────────────────────────────────────────

def to_markdown(
    report: NightlyReport,
    *,
    dashboard_url: str | None = None,
    blame: BlameReport | None = None,
) -> str:
    lines = [
        f"# k4Bench nightly regression report — {report.report_night or 'no data'}",
        "",
        f"Generated {report.generated_at}. "
        f"{len(report.by_detector())} detector(s) checked: "
        f"🔴 {len(report.regressions)} regression(s), "
        f"❌ {len(report.failures) + len(report.job_failures)} failure(s), "
        f"⚠️ {len(report.watches)} on watch.",
        "",
    ]
    for detector, groups in report.by_detector().items():
        lines.append(f"## {_detector_badge(groups)} {detector}")
        lines.append("")
        for group in groups:
            # Both dashboard views are scoped by the (detector, platform,
            # sample) triple, so the links ride on the group, never the
            # detector heading (see _group_links).
            links = " · ".join(
                f"[↗ {text}]({href})"
                for text, href in _group_links(dashboard_url, group, report.report_night)
            )
            if len(groups) > 1:
                group_heading = f"### {_group_title(group)}"
                if links:
                    group_heading += f" · {links}"
                lines.append(group_heading)
                lines.append("")
            elif links:
                lines.append(links)
                lines.append("")
            for msg in group.job_failures:
                lines.append(f"- ❌ **{msg}**")
            for msg in group.notes:
                lines.append(f"- ❔ {msg}")
            if group.job_failures or group.notes:
                lines.append("")
            rows = _table_rows(group)
            if rows:
                lines.append("| Metric | Config | Current | Baseline (median) | Δ | Status |")
                lines.append("|---|---|---|---|---|---|")
                for v in rows:
                    lines.append(
                        f"| {_metric_name(v)} | {v.label} | {_fmt(v.value)} "
                        f"| {_fmt(v.baseline_median)} | {_fmt_pct(v.pct_change)} "
                        f"| {_status_cell(v)} |"
                    )
                lines.append("")
            blame_lines = _blame_markdown_lines(group, blame)
            if blame_lines:
                lines.extend(blame_lines)
                lines.append("")
            if group.verdicts:
                lines.append(f"_{_quiet_summary(group)}_")
                lines.append("")
    lines.append(_markdown_footer(report))
    return "\n".join(lines)


# ── HTML (email body: self-contained, inline styles only) ────────────────────

_ROW_STYLE = "padding:4px 10px;border-bottom:1px solid #e5e5e5;text-align:left;"
#: One color per severity — same red for CONFIRMED and FAILURE (both worth
#: attention); no red/blue good-bad coding *by direction* within CONFIRMED.
_SEVERITY_COLORS = {
    Severity.CONFIRMED: "#d63c3c",
    Severity.WATCH: "#b58900",
    Severity.FAILURE: "#d63c3c",
}


def _view_link(href: str, text: str) -> str:
    """Small inline "↗ ..." link, styled the same wherever a heading links
    out to the dashboard (detector → Regressions, group → Run Trends)."""
    return (
        f' <a href="{href}" style="font-size:11px;font-weight:normal;'
        f'color:#5b9bd5;text-decoration:none;">↗ {text}</a>'
    )


def to_html(report: NightlyReport, *, dashboard_url: str | None = None,
            actions_url: str | None = None,
            blame: BlameReport | None = None) -> str:
    """Self-contained HTML for the e-group email (inline styles, no CSS/JS)."""
    parts = [
        '<div style="font-family:Helvetica,Arial,sans-serif;max-width:860px;">',
        f'<h2 style="margin-bottom:0.2em;">k4Bench nightly regression report — '
        f"{report.report_night or 'no data'}</h2>",
        f'<p style="color:#555;margin-top:0;">Generated {report.generated_at}. '
        f"{len(report.by_detector())} detector(s) checked: "
        f"🔴 {len(report.regressions)} regression(s), "
        f"❌ {len(report.failures) + len(report.job_failures)} failure(s), "
        f"⚠️ {len(report.watches)} on watch.</p>",
    ]
    links = []
    if dashboard_url:
        # The cross-detector at-a-glance summary lives in the Overview tab;
        # the per-group links below land on the scoped Regressions view.
        overview_href = _dashboard_link(dashboard_url, tab="Overview")
        links.append(f'<a href="{overview_href}">Dashboard — Overview</a>')
    if actions_url:
        links.append(f'<a href="{actions_url}">GitHub Actions run</a>')
    if links:
        parts.append(f'<p>{" · ".join(links)}</p>')

    for detector, groups in report.by_detector().items():
        parts.append(
            '<h3 style="border-bottom:2px solid #ddd;padding-bottom:2px;">'
            f"{_detector_badge(groups)} {detector}</h3>"
        )
        for group in groups:
            # Both dashboard views are scoped by the (detector, platform,
            # sample) triple, so the links ride on the group, never the
            # detector heading (see _group_links).
            links_html = "".join(
                _view_link(href, text)
                for text, href in _group_links(dashboard_url, group, report.report_night)
            )
            if len(groups) > 1:
                parts.append(
                    f'<h4 style="margin-bottom:0.3em;">{_group_title(group)}{links_html}</h4>'
                )
            elif links_html:
                parts.append(f'<p style="margin:0 0 6px;">{links_html.strip()}</p>')
            for msg in group.job_failures:
                parts.append(f'<p style="color:#d63c3c;font-weight:bold;">❌ {msg}</p>')
            for msg in group.notes:
                parts.append(f'<p style="color:#777;">❔ {msg}</p>')
            rows = _table_rows(group)
            if rows:
                parts.append(
                    '<table style="border-collapse:collapse;font-size:14px;">'
                    "<tr>"
                    + "".join(
                        f'<th style="{_ROW_STYLE}background:#f5f5f5;">{h}</th>'
                        for h in ("Metric", "Config", "Current",
                                  "Baseline (median)", "Δ", "Status")
                    )
                    + "</tr>"
                )
                for i, v in enumerate(rows):
                    color = _SEVERITY_COLORS.get(v.severity, "#333")
                    row_bg = "background:#fafafa;" if i % 2 else ""
                    parts.append(
                        f'<tr style="{row_bg}">'
                        f'<td style="{_ROW_STYLE}">{_metric_name(v)}</td>'
                        f'<td style="{_ROW_STYLE}">{v.label}</td>'
                        f'<td style="{_ROW_STYLE}">{_fmt(v.value)}</td>'
                        f'<td style="{_ROW_STYLE}">{_fmt(v.baseline_median)}</td>'
                        f'<td style="{_ROW_STYLE}">{_fmt_pct(v.pct_change)}</td>'
                        f'<td style="{_ROW_STYLE}color:{color};font-weight:bold;">'
                        f"{_status_cell(v)}</td>"
                        "</tr>"
                    )
                parts.append("</table>")
            blame_block = _blame_html_block(group, blame)
            if blame_block:
                parts.append(blame_block)
            if group.verdicts:
                parts.append(
                    f'<p style="color:#777;font-size:13px;">{_quiet_summary(group)}</p>'
                )
    parts.append(_html_footer(report))
    parts.append("</div>")
    return "\n".join(parts)


# ── JSON (EOS artifact) ───────────────────────────────────────────────────────

def _sanitize(obj):
    """Make the dataclass dump strictly JSON-serializable: enums → values,
    non-finite floats → None (strict JSON has no Infinity/NaN)."""
    if isinstance(obj, Severity | Direction):
        return obj.value
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_sanitize(v) for v in obj]
    return obj


def to_json(report: NightlyReport) -> dict:
    """Plain-dict form of the report for ``json.dump``.

    The ``summary`` block duplicates the convenience properties so the CI
    workflow (and the dashboard banner) can gate on the report without
    re-deriving anything.
    """
    data = _sanitize(dataclasses.asdict(report))
    data["summary"] = {
        "report_night":     report.report_night,
        "n_detectors":      len(report.by_detector()),
        "n_regressions":    len(report.regressions),
        "n_watches":        len(report.watches),
        "n_failures":       len(report.failures) + len(report.job_failures),
        "has_alertable":    report.has_alertable,
    }
    return data


#: Field names :class:`MetricVerdict` accepts. Reports are read back by
#: whatever dashboard is deployed, which is not necessarily built from the
#: commit that wrote them, so unknown keys are dropped rather than raising —
#: a report gaining a field must not break older readers. Derived from the
#: dataclass, not hand-listed, on purpose: the accepted set *is* the
#: constructor's parameters by definition, and a second, hand-maintained list
#: would silently drift the day a field is added and the list is not.
_VERDICT_FIELDS = frozenset(f.name for f in dataclasses.fields(MetricVerdict))


def from_json(data: dict) -> NightlyReport:
    """Rebuild a :class:`NightlyReport` from :func:`to_json` output (used by
    the dashboard when reading ``_reports/{date}/report.json`` off EOS)."""
    groups = []
    for g in data.get("groups", []):
        verdicts = [
            MetricVerdict(**{
                **{k: val for k, val in v.items() if k in _VERDICT_FIELDS},
                "severity": Severity(v["severity"]),
                "direction": Direction(v["direction"]),
            })
            for v in g.get("verdicts", [])
        ]
        groups.append(RunGroupReport(
            detector=g["detector"], platform=g["platform"], sample=g["sample"],
            k4h_release=g.get("k4h_release", ""), run_date=g.get("run_date", ""),
            run_id=g.get("run_id", ""), verdicts=verdicts,
            job_failures=list(g.get("job_failures", [])),
            notes=list(g.get("notes", [])),
            reliable=g.get("reliable"),
        ))
    return NightlyReport(generated_at=data.get("generated_at", ""), groups=groups)
