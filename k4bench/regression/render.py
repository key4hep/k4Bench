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
import math
import re
from datetime import date
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
    e.g. a deep link to the Regressions tab, optionally scoped to one
    detector's expander."""
    split = urlsplit(dashboard_url)
    query = dict(parse_qsl(split.query))
    query.update(params)
    return urlunsplit(split._replace(query=urlencode(query)))


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


# ── Markdown ──────────────────────────────────────────────────────────────────

def to_markdown(report: NightlyReport, *, dashboard_url: str | None = None) -> str:
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
        heading = f"## {_detector_badge(groups)} {detector}"
        if dashboard_url:
            href = _dashboard_link(dashboard_url, tab="Regressions", detector=detector)
            heading += f" · [↗ Regressions]({href})"
        lines.append(heading)
        lines.append("")
        for group in groups:
            # Run Trends needs a (detector, platform, sample) triple, unlike the
            # detector-wide Regressions link above, so it's scoped per group.
            trends_href = (
                _dashboard_link(
                    dashboard_url, tab="Run Trends", detector=group.detector,
                    platform=group.platform, sample=group.sample,
                ) if dashboard_url else None
            )
            if len(groups) > 1:
                group_heading = f"### {_group_title(group)}"
                if trends_href:
                    group_heading += f" · [↗ Run Trends]({trends_href})"
                lines.append(group_heading)
                lines.append("")
            elif trends_href:
                lines.append(f"[↗ Run Trends]({trends_href})")
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
                        f"| {_badge(v)} |"
                    )
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
            actions_url: str | None = None) -> str:
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
        overview_href = _dashboard_link(dashboard_url, tab="Regressions")
        links.append(f'<a href="{overview_href}">Dashboard — Regressions tab</a>')
    if actions_url:
        links.append(f'<a href="{actions_url}">GitHub Actions run</a>')
    if links:
        parts.append(f'<p>{" · ".join(links)}</p>')

    for detector, groups in report.by_detector().items():
        detector_link = (
            _view_link(_dashboard_link(dashboard_url, tab="Regressions", detector=detector),
                       "Regressions")
            if dashboard_url else ""
        )
        parts.append(
            '<h3 style="border-bottom:2px solid #ddd;padding-bottom:2px;">'
            f"{_detector_badge(groups)} {detector}{detector_link}</h3>"
        )
        for group in groups:
            # Run Trends needs a (detector, platform, sample) triple, unlike the
            # detector-wide Regressions link above, so it's scoped per group.
            trends_link = (
                _view_link(
                    _dashboard_link(
                        dashboard_url, tab="Run Trends", detector=group.detector,
                        platform=group.platform, sample=group.sample,
                    ),
                    "Run Trends",
                ) if dashboard_url else ""
            )
            if len(groups) > 1:
                parts.append(
                    f'<h4 style="margin-bottom:0.3em;">{_group_title(group)}{trends_link}</h4>'
                )
            elif trends_link:
                parts.append(f'<p style="margin:0 0 6px;">{trends_link.strip()}</p>')
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
                        f"{_badge(v)}</td>"
                        "</tr>"
                    )
                parts.append("</table>")
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


def from_json(data: dict) -> NightlyReport:
    """Rebuild a :class:`NightlyReport` from :func:`to_json` output (used by
    the dashboard when reading ``_reports/{date}/report.json`` off EOS)."""
    groups = []
    for g in data.get("groups", []):
        verdicts = [
            MetricVerdict(**{
                **v,
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
