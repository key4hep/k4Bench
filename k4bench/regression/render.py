"""Shared rendering primitives for the nightly regression report and the JSON
artifact the dashboard reads back (``_reports/{date}/report.json``).

This module owns two things:

* the human-label helpers the dashboard *and* the email both draw with —
  severity badges, metric/sample/platform prettifiers, the dashboard deep-link
  builder;
* the ``to_json``/``from_json`` round-trip for the EOS artifact.

The e-group email body (subject, "Needs attention", ranked candidate cards and
the bounded detailed report) is rendered by :mod:`k4bench.regression.email`,
which imports the helpers below. Keeping the JSON renderer and the dashboard's
imports here — stable — while the email evolves separately is deliberate: the
email is redesigned often, the on-disk report schema almost never.

Ordering vocabulary shared everywhere: hard failures first, then confirmed
regressions (any direction — "regression" here means any confirmed step beyond
the baseline, not just a bad one; none of this is judged good or bad), then a
short WATCH/OK summary — never a row per OK metric.
"""

from __future__ import annotations

import dataclasses
import math
import re
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
        "n_new":            len(report.new_regressions),
        "n_reconfirmed":    len(report.reconfirmed_regressions),
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


# Compatibility shims for callers that imported the old public renderer path.
# Imports are deliberately lazy: ``email`` imports the shared helpers above, so
# importing it at module load time here would create a cycle.
def to_html(report: NightlyReport, **kwargs) -> str:
    """Render the HTML email via :mod:`k4bench.regression.email`.

    New code should import from ``k4bench.regression.email`` directly; this
    wrapper keeps existing integrations working after the renderer split.
    """
    from k4bench.regression.email import to_html as render_email_html

    return render_email_html(report, **kwargs)


def to_markdown(report: NightlyReport, **kwargs) -> str:
    """Render the text alternative via :mod:`k4bench.regression.email`.

    See :func:`to_html` for why this compatibility wrapper is lazy.
    """
    from k4bench.regression.email import to_markdown as render_email_markdown

    return render_email_markdown(report, **kwargs)
