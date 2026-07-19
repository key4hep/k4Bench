"""Decide which pull requests hear about a regression, and what they are told.

The nightly ranker already answers "which PR most likely caused this step"
(:mod:`k4bench.blame.rank`), but that answer only reaches people who read the
e-group mail or open the dashboard — never the author of the change. This module
turns a night's ``report.json`` + ``blame.json`` into a set of pull-request
comments; :mod:`k4bench.blame.publish` posts them.

Everything here is pure — no network, no token, no clock — so the whole
"who gets told what" decision is unit-testable, and the CLI can print exactly
what would be posted (``--dry-run``) without touching GitHub.

Commenting in someone else's repository is an outward-facing act on the strength
of a model's judgement, so the gates are deliberately narrow and all of them
must pass:

* the candidate's repository is on the **allowlist** — an empty allowlist means
  the bot is inert, which is how it ships;
* the ranker's likelihood is at or above ``min_score`` (default 80);
* the pull request is **merged** — an open PR cannot have shipped in a release;
* the blame entry's candidate discovery was **complete**
  (:attr:`~k4bench.blame.models.BlameEntry.discovery_incomplete`) — naming one PR
  out of a knowingly partial set is exactly the overclaim the ranker itself
  refuses to make;
* the night is under the ``max_comments`` cap — a storm is a bug, not a night.

One comment covers one ``(pull request, change window)`` pair, no matter how
many metrics or benchmark configurations that window moved: the reader's
question is "did my change do this?", asked once. :func:`marker_for` gives that
pair a stable hidden key so a later night edits the existing comment instead of
posting a second one.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from k4bench.blame.models import RANKING_DISCLOSURE, BlameReport, CandidatePR
from k4bench.labels import pretty_platform, pretty_sample
from k4bench.regression.models import MetricVerdict, NightlyReport
from k4bench.regression.render import (
    stack_changes_href,
    window_href,
    window_token,
)

_log = logging.getLogger(__name__)

#: Marker format version. Bumping it makes every existing comment invisible to
#: the upsert (a *new* comment is posted rather than the old one edited), so it
#: changes only when a body is no longer an in-place successor of the old one.
MARKER_VERSION = "v1"

#: Metric rows shown before the table defers to the dashboard, and candidate
#: rows shown for the rest of the window. Both are display caps: the selection
#: above them is complete, only the rendering is bounded.
_MAX_METRIC_ROWS = 8
_MAX_OTHER_CANDIDATES = 5

#: Longest ranker explanation quoted verbatim. The contract asks for one
#: sentence; a model that ignores it must not paste an essay into someone's PR.
_MAX_DESCRIPTION_CHARS = 400

_DEFAULT_MIN_SCORE = 80.0
_DEFAULT_MAX_COMMENTS = 10


class CommentConfigError(ValueError):
    """The comment config is not shaped like a :class:`CommentPolicy`.

    Raised rather than defaulted: every field here decides whether — and where —
    the bot writes to a repository it does not own, so a typo must stop the
    step, never silently widen or narrow its reach."""


@dataclass(frozen=True)
class CommentPolicy:
    """Who may be commented on, and how confidently.

    ``repos`` holds lowercase ``owner/repo`` slugs; GitHub slugs are
    case-insensitive, so matching is done on the lowered form while the
    candidate's own spelling is what gets displayed. An empty ``repos`` disables
    the bot entirely — the shipped default.
    """

    min_score: float = _DEFAULT_MIN_SCORE
    repos: frozenset[str] = frozenset()
    max_comments: int = _DEFAULT_MAX_COMMENTS

    @property
    def enabled(self) -> bool:
        return bool(self.repos)

    def allows(self, candidate: CandidatePR) -> bool:
        """True when *candidate* clears the repo, score and merged gates."""
        return (
            candidate.repo.lower() in self.repos
            and math.isfinite(candidate.score)
            and candidate.score >= self.min_score
            and bool(candidate.merged_at)
        )

    @classmethod
    def from_config(cls, data: dict[str, Any] | None) -> CommentPolicy:
        """Build a policy from the parsed ``.github/blame-comments.yml``.

        Unknown keys, wrong types and out-of-range values raise
        :class:`CommentConfigError` — see the class docstring for why this one
        config is strict where the report schemas are forgiving.
        """
        data = data or {}
        if not isinstance(data, dict):
            raise CommentConfigError("comment config must be a mapping")
        unknown = set(data) - {"min_score", "max_comments", "repos"}
        if unknown:
            raise CommentConfigError(f"unknown key(s): {', '.join(sorted(unknown))}")

        min_score = _positive_number(
            data.get("min_score", _DEFAULT_MIN_SCORE), "min_score"
        )
        if min_score > 100:
            raise CommentConfigError("min_score must be between 0 and 100")
        max_comments = int(
            _positive_number(data.get("max_comments", _DEFAULT_MAX_COMMENTS),
                             "max_comments")
        )

        raw_repos = data.get("repos") or []
        if not isinstance(raw_repos, list):
            raise CommentConfigError("repos must be a list of owner/repo slugs")
        repos = set()
        for slug in raw_repos:
            if not isinstance(slug, str) or slug.count("/") != 1 or slug.startswith("/") \
                    or slug.endswith("/"):
                raise CommentConfigError(f"not an owner/repo slug: {slug!r}")
            repos.add(slug.strip().lower())
        return cls(min_score=min_score, repos=frozenset(repos), max_comments=max_comments)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CommentConfigError(f"{name} must be a number")
    if not math.isfinite(value) or value < 0:
        raise CommentConfigError(f"{name} must be a non-negative number")
    return float(value)


@dataclass(frozen=True)
class PRComment:
    """One rendered comment and where it goes.

    ``marker`` is the hidden key the upsert recognises; it is also the first
    line of ``body``, so a comment always carries the key that identifies it.
    """

    repo: str
    number: int
    marker: str
    body: str
    score: float

    @property
    def target(self) -> str:
        """``owner/repo#123`` — how this comment is named in logs."""
        return f"{self.repo}#{self.number}"


def marker_for(base_release: str | None, onset_release: str | None) -> str:
    """The hidden HTML key identifying a comment about one change window.

    The window is the identity because it is what the comment is *about*: the
    same PR implicated in a genuinely different window gets its own comment,
    while tonight's re-confirmation of the same window edits the one already
    there. Reuses :func:`~k4bench.regression.render.window_token` so the key and
    the dashboard link it carries name the window identically.
    """
    return (
        f"<!-- k4bench-blame-comment:{MARKER_VERSION} "
        f"window={window_token(base_release, onset_release)} -->"
    )


@dataclass
class _Bucket:
    """Everything one comment is rendered from, accumulated across the metrics
    that share a ``(pull request, window)``."""

    candidate: CandidatePR
    base_release: str | None
    onset_release: str
    #: The run group's Key4hep release directory (``key4hep-2026-07-04``) — the
    #: dashboard's ``?stack=`` vocabulary, which is the release *directory* name
    #: rather than the bare date a verdict carries.
    stack: str
    verdicts: list[MetricVerdict]
    others: dict[tuple[str, int], CandidatePR]


def select(
    report: NightlyReport,
    blame: BlameReport,
    policy: CommentPolicy,
    *,
    dashboard_url: str | None = None,
    actions_url: str | None = None,
) -> list[PRComment]:
    """The comments this night warrants, worst-first and capped.

    Driven from the *report*'s confirmed regressions rather than from the
    sidecar's entries, so a comment can only ever describe a regression that is
    confirmed in tonight's report — a stale entry has nothing to attach to.
    """
    if not policy.enabled:
        return []

    buckets: dict[tuple[str, int, str | None, str], _Bucket] = {}
    # Walked group by group rather than through ``report.regressions`` so each
    # verdict keeps its group's release directory — the dashboard links need it.
    for group in report.groups:
        for verdict in group.regressions:
            entry = blame.entry_for(verdict)
            if entry is None or entry.discovery_incomplete:
                continue
            candidates = entry.candidates
            for candidate in candidates:
                if not policy.allows(candidate):
                    continue
                key = (
                    candidate.repo.lower(), candidate.number,
                    entry.base_release, entry.onset_release,
                )
                bucket = buckets.get(key)
                if bucket is None:
                    bucket = buckets[key] = _Bucket(
                        candidate=candidate,
                        base_release=entry.base_release,
                        onset_release=entry.onset_release,
                        stack=group.k4h_release,
                        verdicts=[],
                        others={},
                    )
                elif candidate.score > bucket.candidate.score:
                    # Several metrics share the window; keep the highest
                    # judgement the night produced as the headline likelihood.
                    bucket.candidate = candidate
                bucket.verdicts.append(verdict)
                for other in candidates:
                    if (other.repo.lower(), other.number) != (key[0], key[1]):
                        bucket.others.setdefault(
                            (other.repo.lower(), other.number), other
                        )

    comments = [
        _render(
            bucket, report_night=report.report_night,
            dashboard_url=dashboard_url, actions_url=actions_url,
        )
        for bucket in sorted(
            buckets.values(),
            key=lambda b: (-b.candidate.score, b.candidate.repo, b.candidate.number),
        )
    ]
    if len(comments) > policy.max_comments:
        dropped = ", ".join(c.target for c in comments[policy.max_comments:])
        _log.warning(
            "select: %d comments exceed the max_comments cap of %d — not posting: %s",
            len(comments), policy.max_comments, dropped,
        )
        comments = comments[:policy.max_comments]
    return comments


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render(
    bucket: _Bucket,
    *,
    report_night: str,
    dashboard_url: str | None,
    actions_url: str | None,
) -> PRComment:
    """One bucket as a GitHub-flavoured Markdown comment."""
    marker = marker_for(bucket.base_release, bucket.onset_release)
    verdicts = sorted(bucket.verdicts, key=_verdict_sort_key)
    primary = verdicts[0]
    scopes = {(v.detector, v.platform, v.sample) for v in verdicts}

    body = "\n".join(
        part for part in (
            marker,
            "### ⚠️ Possible performance regression traced to this pull request",
            "",
            _lead(primary, bucket, len(scopes)),
            "",
            f"**Likelihood this pull request is the cause: "
            f"{_pct(bucket.candidate.score)}**",
            _quote(bucket.candidate.description),
            _metrics_table(verdicts, multi_scope=len(scopes) > 1),
            _others_table(bucket),
            _where_to_look(
                bucket, primary, scopes, report_night=report_night,
                dashboard_url=dashboard_url, actions_url=actions_url,
            ),
            "",
            f"<sub>{RANKING_DISCLOSURE} Posted automatically by "
            "[k4Bench](https://github.com/key4hep/k4Bench); this comment is "
            "updated in place while the regression stands. If the attribution "
            "is wrong, say so here — nothing is blocked by it.</sub>",
        ) if part is not None
    )
    return PRComment(
        repo=bucket.candidate.repo,
        number=bucket.candidate.number,
        marker=marker,
        body=body,
        score=bucket.candidate.score,
    )


def _lead(primary: MetricVerdict, bucket: _Bucket, n_scopes: int) -> str:
    """The opening sentence: what was measured, and which release window the
    step entered in — the window being the reason this PR is implicated."""
    where = (
        f"**{primary.detector}** · {pretty_sample(primary.sample)} · "
        f"{pretty_platform(primary.platform)}"
        if n_scopes == 1
        else f"**{n_scopes} benchmark configurations**"
    )
    window = (
        f"with Key4hep release **{bucket.onset_release}**, the first release "
        f"benchmarked after **{bucket.base_release}**"
        if bucket.base_release
        else f"at or before Key4hep release **{bucket.onset_release}** "
             "(no earlier settled measurement bounds the window)"
    )
    return (
        f"k4Bench's nightly benchmarks confirmed a step in {where} that entered "
        f"{window} — the change window this pull request merged in."
    )


def _quote(description: str) -> str | None:
    """The ranker's one-line reason as a blockquote, or nothing when it declined
    to explain (a scored-but-unexplained candidate is not comment-worthy prose,
    and the score already stands on its own line)."""
    text = _one_line(description, _MAX_DESCRIPTION_CHARS)
    return f"\n> {text}" if text else None


def _metrics_table(verdicts: list[MetricVerdict], *, multi_scope: bool) -> str:
    """What actually moved: metric, benchmark config, and by how much.

    Metric columns keep their raw names — they are the identifiers the dashboard
    the links point at is labelled with, so a reader can find the exact series.
    """
    header = ["Metric", "Config", "Change"]
    if multi_scope:
        header.insert(0, "Benchmark")
    rows = []
    for v in verdicts[:_MAX_METRIC_ROWS]:
        metric = f"`{v.metric}`" + (f" · {_cell(v.sub_detector)}" if v.sub_detector else "")
        row = [metric, f"`{_cell(v.label)}`", _signed_pct(v.pct_change)]
        if multi_scope:
            row.insert(0, f"{_cell(v.detector)} · {_cell(v.sample)}")
        rows.append(row)

    lines = [
        "",
        "#### What moved",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    omitted = len(verdicts) - len(rows)
    if omitted > 0:
        lines += ["", f"_…and {omitted} more metric(s) in the same change window._"]
    return "\n".join(lines)


def _others_table(bucket: _Bucket) -> str | None:
    """The rest of the window's candidates with their likelihoods — the reader
    needs to see what else was in the frame to weigh the claim against this PR,
    including the case where nothing else was."""
    others = sorted(
        bucket.others.values(), key=lambda c: (-c.score, c.repo, c.number)
    )
    lines = ["", "#### Other pull requests in the same change window", ""]
    if not others:
        lines.append(
            "_None — this was the only pull request found across every tracked "
            "package that changed in this window._"
        )
        return "\n".join(lines)

    shown = others[:_MAX_OTHER_CANDIDATES]
    lines += [
        "| Pull request | Likelihood |",
        "|---|---|",
        *(
            f"| [{_cell(c.repo)}#{c.number}]({c.url}) — {_cell(_one_line(c.title, 80))} "
            f"| {_pct(c.score)} |"
            for c in shown
        ),
    ]
    if len(others) > len(shown):
        lines += ["", f"_…and {len(others) - len(shown)} more candidate(s)._"]
    return "\n".join(lines)


def _where_to_look(
    bucket: _Bucket,
    primary: MetricVerdict,
    scopes: set[tuple[str, str, str]],
    *,
    report_night: str,
    dashboard_url: str | None,
    actions_url: str | None,
) -> str | None:
    """The links that let a reader check the claim rather than take it.

    Scoped to the *primary* benchmark configuration — the one carrying the
    largest step — with the others named as a count, since a dashboard view is
    always one configuration.
    """
    regressions = window_href(
        dashboard_url,
        detector=primary.detector, platform=primary.platform, sample=primary.sample,
        base_release=bucket.base_release, onset_release=bucket.onset_release,
        stack=bucket.stack, report_night=report_night,
    )
    packages = stack_changes_href(
        dashboard_url,
        detector=primary.detector, platform=primary.platform, sample=primary.sample,
        base_release=bucket.base_release, onset_release=bucket.onset_release,
    )
    items = [
        (regressions, "Review this regression in the dashboard"),
        (packages, "Every package that changed across this window"),
        (actions_url, "The nightly benchmark run that produced this"),
    ]
    links = [f"- [{text}]({href})" for href, text in items if href]
    if not links:
        return None
    lines = ["", "#### Where to look", "", *links]
    if len(scopes) > 1:
        lines.append(
            f"- _…{len(scopes) - 1} further benchmark configuration(s) moved in "
            "the same window; the dashboard has them all._"
        )
    return "\n".join(lines)


def _verdict_sort_key(v: MetricVerdict) -> tuple:
    """Biggest movement first, ties broken by identity so a body is stable
    across nights — a reordered table would look like a change and trigger a
    pointless edit."""
    magnitude = abs(v.pct_change) if v.pct_change is not None else 0.0
    return (-magnitude, v.detector, v.sample, v.label, v.metric, v.sub_detector or "")


def _pct(score: float) -> str:
    return f"{int(round(score))}%"


def _signed_pct(pct_change: float | None) -> str:
    """A metric's step as a signed percentage. ``pct_change`` is a fraction on
    :class:`MetricVerdict`, matching the report's own formatting."""
    if pct_change is None or not math.isfinite(pct_change):
        return "—"
    return f"{pct_change:+.1%}"


def _one_line(text: str, limit: int) -> str:
    """Model- or GitHub-authored text flattened to one line and clipped.

    Newlines would break out of a table cell or a blockquote, so they are
    collapsed rather than escaped."""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def _cell(text: str | None) -> str:
    """Text safe inside a Markdown table cell: a pipe would end the column."""
    return (text or "").replace("|", "\\|")
