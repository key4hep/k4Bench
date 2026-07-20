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
  the bot is inert;
* the ranker's likelihood is at or above ``min_score`` (default 80);
* the pull request is **merged** — an open PR cannot have shipped in a release;
* the blame entry's candidate discovery was **complete**
  (:attr:`~k4bench.blame.models.BlameEntry.discovery_incomplete`) — naming one PR
  out of a knowingly partial set is exactly the overclaim the ranker itself
  refuses to make;
* the night is under the ``max_comments`` cap — a storm is a bug, not a night.

One comment covers one ``(pull request, change window)`` pair — the reader's
question is "did my change do this?", asked once — and :func:`marker_for` gives
that pair a stable hidden key so a later night edits the existing comment
instead of posting a second one. Inside that one comment, each benchmark scope
(``detector, platform, sample``) the window moved is its own subsection: the
ranker scores a candidate once per scope, so two independent judgements (a 95%
ALLEGRO and an 81% IDEA, say) are shown side by side, never flattened into a
single headline number.
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


class CommentStormError(RuntimeError):
    """More comments than ``max_comments`` — the attribution is suspect, so the
    whole night is suppressed rather than posting the loudest few accusations
    into repositories k4Bench does not own.

    Raised rather than returned as an empty list so a caller can tell a *tripped
    circuit breaker* (something is wrong with tonight's attribution) apart from
    an *ordinary quiet night* (nothing crossed the threshold) — the two look
    identical in the comment count but mean opposite things to whoever is
    watching the bot."""

    def __init__(self, count: int, cap: int, targets: list[str]):
        self.count = count
        self.cap = cap
        self.targets = tuple(targets)
        super().__init__(
            f"{count} comments exceed the max_comments cap of {cap}: "
            + ", ".join(targets)
        )


@dataclass(frozen=True)
class CommentPolicy:
    """Who may be commented on, and how confidently.

    ``repos`` holds lowercase ``owner/repo`` slugs; GitHub slugs are
    case-insensitive, so matching is done on the lowered form while the
    candidate's own spelling is what gets displayed. An empty ``repos`` disables
    the bot entirely.
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
        if data is None:
            data = {}
        if not isinstance(data, dict):
            # A falsey-but-present document (``false``, ``0``, ``[]``) is
            # malformed, not "no config": only an absent one defaults to inert.
            raise CommentConfigError("comment config must be a mapping")
        unknown = set(data) - {"min_score", "max_comments", "repos"}
        if unknown:
            raise CommentConfigError(f"unknown key(s): {', '.join(sorted(unknown))}")

        min_score = _positive_number(
            data.get("min_score", _DEFAULT_MIN_SCORE), "min_score"
        )
        if min_score > 100:
            raise CommentConfigError("min_score must be between 0 and 100")
        max_comments = _positive_int(
            data.get("max_comments", _DEFAULT_MAX_COMMENTS), "max_comments"
        )

        raw_repos = data.get("repos", [])
        if raw_repos is None:  # `repos:` with no value is an empty allowlist
            raw_repos = []
        if not isinstance(raw_repos, list):
            # ``repos: false`` / ``repos: k4geo`` — a scalar is not an allowlist.
            raise CommentConfigError("repos must be a list of owner/repo slugs")
        repos = set()
        for slug in raw_repos:
            # Validate the *stripped* slug: ``"owner/ "`` must not slip through
            # the slash check and then be stored as the truncated ``"owner/"``.
            cleaned = slug.strip() if isinstance(slug, str) else slug
            if not isinstance(cleaned, str) or cleaned.count("/") != 1 \
                    or cleaned.startswith("/") or cleaned.endswith("/"):
                raise CommentConfigError(f"not an owner/repo slug: {slug!r}")
            repos.add(cleaned.lower())
        return cls(min_score=min_score, repos=frozenset(repos), max_comments=max_comments)


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CommentConfigError(f"{name} must be a number")
    if not math.isfinite(value) or value < 0:
        raise CommentConfigError(f"{name} must be a non-negative number")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    """A count that must be a whole number, at least one. A float like ``2.9``
    is a typo, not a rounding hint — silently truncating it would post one fewer
    comment than the config appears to ask for; a zero disables the bot in a way
    an empty ``repos`` already expresses more honestly."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CommentConfigError(f"{name} must be a whole number")
    if value < 1:
        raise CommentConfigError(f"{name} must be a positive integer")
    return value


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
class _Scope:
    """One benchmark scope's slice of a comment.

    The ranker scores each candidate once per ``(detector, platform, sample)``
    scope — every metric in that scope shares the one judgement — so a scope,
    not a metric, is the unit that carries its own likelihood, reason and
    competing candidates. A comment renders one subsection per scope rather than
    flattening two independent judgements into a single headline number.
    """

    detector: str
    platform: str
    sample: str
    #: This scope's Key4hep release directory (``key4hep-2026-07-04``) — the
    #: dashboard's ``?stack=`` vocabulary, the release *directory* name rather
    #: than the bare date a verdict carries. Kept per scope so each subsection's
    #: dashboard link names the stack the scope actually ran against.
    stack: str
    candidate: CandidatePR
    verdicts: list[MetricVerdict]
    others: dict[tuple[str, int], CandidatePR]


@dataclass
class _Bucket:
    """Everything one comment is rendered from: one pull request, one change
    window, and every benchmark scope of that window the PR was scored in."""

    repo: str
    number: int
    base_release: str | None
    onset_release: str
    scopes: dict[tuple[str, str, str], _Scope]

    @property
    def top_score(self) -> float:
        """The strongest likelihood across the PR's scopes — used only to order
        and log comments, never rendered as a single combined judgement."""
        return max(s.candidate.score for s in self.scopes.values())


def select(
    report: NightlyReport,
    blame: BlameReport,
    policy: CommentPolicy,
    *,
    dashboard_url: str | None = None,
) -> list[PRComment]:
    """The comments this night warrants, worst-first.

    Driven from the *report*'s confirmed regressions rather than from the
    sidecar's entries, so a comment can only ever describe a regression that is
    confirmed in tonight's report — a stale entry has nothing to attach to.

    The rendered body carries nothing that varies from night to night (no run
    URL, no report-night query param): a regression that stands for a week is
    one comment, and the upsert must see an unchanged body so it edits nothing
    and re-notifies no one. Overshooting ``max_comments`` raises
    :class:`CommentStormError` rather than returning a truncated list — a night
    that loud is a bug, not a night, and blind-posting ten accusations into
    repositories k4Bench does not own is the exact harm the gates exist to
    prevent. It is raised, not returned empty, so the caller can tell it apart
    from an ordinary night that simply implicated no one.
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
                ident = (candidate.repo.lower(), candidate.number)
                bkey = (*ident, entry.base_release, entry.onset_release)
                bucket = buckets.get(bkey)
                if bucket is None:
                    bucket = buckets[bkey] = _Bucket(
                        repo=candidate.repo, number=candidate.number,
                        base_release=entry.base_release,
                        onset_release=entry.onset_release,
                        scopes={},
                    )
                skey = (verdict.detector, verdict.platform, verdict.sample)
                scope = bucket.scopes.get(skey)
                if scope is None:
                    scope = bucket.scopes[skey] = _Scope(
                        detector=verdict.detector, platform=verdict.platform,
                        sample=verdict.sample, stack=group.k4h_release,
                        candidate=candidate, verdicts=[], others={},
                    )
                elif candidate.score > scope.candidate.score:
                    # Every metric in a scope shares one ranking, so these are
                    # equal in valid builder output; keep the max defensively so
                    # the choice never depends on which metric was walked first.
                    scope.candidate = candidate
                scope.verdicts.append(verdict)
                for other in candidates:
                    other_ident = (other.repo.lower(), other.number)
                    if other_ident == ident:
                        continue
                    prev = scope.others.get(other_ident)
                    if prev is None or other.score > prev.score:
                        scope.others[other_ident] = other

    comments = [
        _render(bucket, dashboard_url=dashboard_url)
        for bucket in sorted(
            buckets.values(),
            key=lambda b: (-b.top_score, b.repo, b.number),
        )
    ]
    if len(comments) > policy.max_comments:
        _log.warning(
            "select: %d comments exceed the max_comments cap of %d — a night this "
            "loud is a bug, not a night; posting none of them",
            len(comments), policy.max_comments,
        )
        raise CommentStormError(
            len(comments), policy.max_comments, [c.target for c in comments]
        )
    return comments


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render(
    bucket: _Bucket,
    *,
    dashboard_url: str | None,
) -> PRComment:
    """One bucket as a GitHub-flavoured Markdown comment.

    A single comment for the ``(pull request, window)``, but one subsection per
    benchmark scope: each scope carries the ranker's own likelihood and reason
    for that scope, so two independent judgements are never collapsed into a
    single headline number."""
    marker = marker_for(bucket.base_release, bucket.onset_release)
    scopes = sorted(
        bucket.scopes.values(),
        key=lambda s: (-s.candidate.score, s.detector, s.platform, s.sample),
    )

    body = "\n".join(
        part for part in (
            marker,
            "### 📉 Possible performance regression traced to this pull request",
            "",
            _alert(bucket, len(scopes)),
            "",
            _window_line(bucket),
            *(_scope_section(scope, bucket, dashboard_url=dashboard_url)
              for scope in scopes),
            "",
            "---",
            "",
            f"<sub>🤖 {RANKING_DISCLOSURE} Posted automatically by "
            "[k4Bench](https://github.com/key4hep/k4Bench).</sub>",
        ) if part is not None
    )
    return PRComment(
        repo=bucket.repo,
        number=bucket.number,
        marker=marker,
        body=body,
        score=bucket.top_score,
    )


def _alert(bucket: _Bucket, n_scopes: int) -> str:
    """The headline claim as a GitHub warning alert — one sentence; the
    per-scope subsections below carry the specifics."""
    tail = (
        ""
        if bucket.base_release
        else " The window is open-ended: no earlier settled measurement "
             "bounds it."
    )
    what = (
        "a regression in the change window"
        if n_scopes == 1
        else (
            f"regressions in {_count(n_scopes, 'benchmark configuration')} "
            "of the change window"
        )
    )
    return (
        "> [!WARNING]\n"
        f"> k4Bench's nightly benchmarks confirmed {what} this PR merged "
        f"in.{tail}"
    )


def _window_line(bucket: _Bucket) -> str:
    """The change window as a single caption line — the Key4hep release dates
    that bound the step, shared by every scope below."""
    window = (
        f"`{bucket.base_release}` → `{bucket.onset_release}`"
        if bucket.base_release
        else f"≤ `{bucket.onset_release}`"
    )
    return f"📆 **Change window:** {window}"


def _scope_section(
    scope: _Scope, bucket: _Bucket, *, dashboard_url: str | None
) -> str:
    """One benchmark scope as a subsection: its likelihood and detector in the
    heading, then the ranker's reason, what moved, the other candidates scored
    for that scope, and the links to check it — all named to the one scope the
    ranker actually judged."""
    verdicts = sorted(scope.verdicts, key=_verdict_sort_key)
    heading = f"#### 🎯 {_pct(scope.candidate.score)} — {_cell(scope.detector)}"
    caption = (
        f"<sub>{pretty_sample(scope.sample)} · "
        f"{pretty_platform(scope.platform)}</sub>"
    )
    return "\n".join(
        part for part in (
            "",
            heading,
            "",
            caption,
            _quote(scope),
            _metrics_table(verdicts),
            _others_section(scope),
            _where_to_look(scope, bucket, dashboard_url=dashboard_url),
        ) if part is not None
    )


def _quote(scope: _Scope) -> str | None:
    """The ranker's one-line reason as a labelled blockquote — the label is
    where the comment openly says an AI made this call. It claims "the most
    likely cause" only when this PR outranks every other candidate *in this
    scope*; a comment can fire on any score above ``min_score``, and a PR the
    ranker placed second here must not be told it came first. Nothing is
    rendered when the ranker declined to explain (a scored-but-unexplained
    candidate is not comment-worthy prose, and the score already stands in the
    heading)."""
    text = _one_line(scope.candidate.description, _MAX_DESCRIPTION_CHARS)
    if not text:
        return None
    outranks_all = all(
        other.score < scope.candidate.score for other in scope.others.values()
    )
    claim = "the most likely" if outranks_all else "a likely"
    return (
        f"\n> 🤖 **The AI ranker judged this PR {claim} cause of the "
        f"regression:** {text}"
    )


def _metrics_table(verdicts: list[MetricVerdict]) -> str:
    """What actually moved in this scope: metric, benchmark label, and by how
    much. Metric columns keep their raw names — they are the identifiers the
    dashboard the links point at is labelled with, so a reader can find the
    exact series."""
    header = ["Metric", "Config", "Change"]
    align = [":---", ":---", "---:"]
    rows = [
        [
            f"`{v.metric}`" + (f" · {_cell(v.sub_detector)}" if v.sub_detector else ""),
            f"`{_cell(v.label)}`",
            _change_cell(v.pct_change),
        ]
        for v in verdicts[:_MAX_METRIC_ROWS]
    ]
    lines = [
        "",
        "##### 📊 What moved",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(align) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    omitted = len(verdicts) - len(rows)
    if omitted > 0:
        lines += [
            "",
            f"_…and {_count(omitted, 'more metric')} in this configuration._",
        ]
    return "\n".join(lines)


def _others_section(scope: _Scope) -> str:
    """The rest of the candidates scored for this scope, with their
    likelihoods — the reader needs to see what else was in the frame to weigh
    the claim against this PR, including the case where nothing else was.
    Collapsed by default: the count in the summary line carries the weight, the
    rows are one click away."""
    others = sorted(
        scope.others.values(), key=lambda c: (-c.score, c.repo, c.number)
    )
    if not others:
        return "\n".join([
            "",
            "> [!NOTE]",
            "> This was the only pull request found across every tracked "
            "package that changed in this configuration's window.",
        ])

    shown = others[:_MAX_OTHER_CANDIDATES]
    lines = [
        "",
        "<details>",
        "<summary><b>Other pull requests scored for this configuration</b> — "
        f"{_count(len(others), 'candidate')}</summary>",
        "",
        "| Pull request | Likelihood |",
        "|:---|---:|",
        *(
            f"| [{_cell(c.repo)}#{c.number}]({c.url}) — {_cell(_one_line(c.title, 80))} "
            f"| {_pct(c.score)} |"
            for c in shown
        ),
    ]
    if len(others) > len(shown):
        lines += ["", f"_…and {_count(len(others) - len(shown), 'more candidate')}._"]
    lines += ["", "</details>"]
    return "\n".join(lines)


def _where_to_look(
    scope: _Scope, bucket: _Bucket, *, dashboard_url: str | None
) -> str | None:
    """The links that let a reader check this scope's claim rather than take it.

    Scoped to this subsection's ``(detector, platform, sample)`` and the stack
    the scope actually ran against, since a dashboard view is always one
    configuration. Every link names the *window*, which does not change from one
    night to the next: no ``report=`` night and no CI-run URL, both of which
    would vary nightly and edit a standing comment for no reason."""
    regressions = window_href(
        dashboard_url,
        detector=scope.detector, platform=scope.platform, sample=scope.sample,
        base_release=bucket.base_release, onset_release=bucket.onset_release,
        stack=scope.stack,
    )
    packages = stack_changes_href(
        dashboard_url,
        detector=scope.detector, platform=scope.platform, sample=scope.sample,
        base_release=bucket.base_release, onset_release=bucket.onset_release,
    )
    items = [
        (regressions, "📈", "Review this regression in the dashboard"),
        (packages, "📦", "Every package that changed across this window"),
    ]
    links = [f"- {icon} [{text}]({href})" for href, icon, text in items if href]
    if not links:
        return None
    return "\n".join(["", "##### 🔎 Where to look", "", *links])


def _verdict_sort_key(v: MetricVerdict) -> tuple:
    """Biggest movement first, ties broken by identity so a body is stable
    across nights — a reordered table would look like a change and trigger a
    pointless edit."""
    magnitude = abs(v.pct_change) if v.pct_change is not None else 0.0
    return (
        -magnitude, v.detector, v.platform, v.sample,
        v.label, v.metric, v.sub_detector or "",
    )


def _pct(score: float) -> str:
    return f"{int(round(score))}%"


def _change_cell(pct_change: float | None) -> str:
    """A metric's step as a signed percentage with a direction marker.
    ``pct_change`` is a fraction on :class:`MetricVerdict`, matching the
    report's own formatting. Both arrows are red on purpose: whichever way a
    confirmed regression moved, it moved the wrong way."""
    if pct_change is None or not math.isfinite(pct_change):
        return "—"
    arrow = "🔺" if pct_change >= 0 else "🔻"
    return f"{arrow} **{pct_change:+.1%}**"


def _count(n: int, noun: str) -> str:
    """``3 candidates`` / ``1 candidate`` — every noun this comment counts
    pluralises with a plain ``s``."""
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _one_line(text: str, limit: int) -> str:
    """Model- or GitHub-authored text flattened to one line, defanged, clipped.

    Newlines would break out of a table cell or a blockquote, so they are
    collapsed rather than escaped; :func:`_defang` then pulls the teeth from any
    Markdown/HTML the prose carries before it lands in an outward-facing
    comment."""
    flat = _defang(" ".join((text or "").split()))
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def _defang(text: str) -> str:
    """Neutralise the active Markdown/HTML in quoted, externally-authored prose.

    A PR title or a model's one-line reason is untrusted text pasted into a
    comment the bot posts in someone else's repository. Left as-is it could:

    * ``@login`` — ping a person on every nightly edit (the same ban the whole
      bot honours by never rendering an author with an ``@``);
    * ``<!-- … -->`` / ``<tag>`` — hide following content, or inject markup;
    * ``![alt](url)`` — pull in a remote image on every render.

    A zero-width space after each trigger character breaks the sequence GitHub
    would act on while leaving the text visually unchanged. Table pipes are left
    to :func:`_cell`, which the cell paths still apply on top of this."""
    zwsp = "\u200b"  # U+200B zero-width space
    return (
        text.replace("@", "@" + zwsp)
        .replace("<", "<" + zwsp)
        .replace("![", "!" + zwsp + "[")
    )


def _cell(text: str | None) -> str:
    """Text safe inside a Markdown table cell: a pipe would end the column."""
    return (text or "").replace("|", "\\|")
