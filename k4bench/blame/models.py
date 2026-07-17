"""Serialized shapes for ``_reports/{night}/blame.json``.

The file is a :class:`BlameReport`: one :class:`BlameEntry` per confirmed
regression, each carrying the release window it entered in and the repos that
moved across that window, and within each repo the ranked candidate pull
requests. The identity fields on :class:`BlameEntry` (``detector`` … ``metric``,
``sub_detector``) are exactly a :class:`~k4bench.regression.models.MetricVerdict`'s
identity, so the dashboard joins an entry back to the verdict it explains with
:meth:`BlameReport.entry_for` — matched on that tuple *and* the blame window, so
an entry written for an earlier build of the same night can never attach to a
verdict whose window has since moved.

Everything here is a plain, frozen dataclass with explicit JSON round-tripping.
:func:`from_json` drops unknown keys rather than raising: ``blame.json`` is read
by whatever dashboard is deployed, not necessarily one built from the commit
that wrote the file, so a schema that gains a field must not break older readers
(the same forward-compatibility rule :mod:`k4bench.regression.render` follows for
``report.json``). Structurally wrong JSON, on the other hand, raises
:class:`BlameSchemaError` — one dedicated exception the readers at the
integration boundaries (dashboard, notifier) catch to hide blame rather than
crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


class BlameSchemaError(ValueError):
    """Parsed as JSON, but not shaped like a :class:`BlameReport`.

    A ``ValueError`` subclass so any boundary already catching bad JSON
    (``json.loads`` raises ``ValueError`` too) contains a bad schema the same
    way — a malformed sidecar must never crash the dashboard or block the
    nightly email."""


def _only_known(cls: type, data: dict) -> dict:
    """*data* restricted to *cls*'s constructor fields — the forward-compatible
    read that lets a newer writer add a key without breaking this reader."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


@dataclass(frozen=True)
class CandidatePR:
    """One pull request that could have caused the regression.

    ``score`` (a 0–100 likelihood this PR is the cause) and ``description`` (a
    one-line "why") are the **ranker's** output. Several PRs can land in one
    package's commit range, so each is scored independently — the ranker judges
    every candidate of a regression together and assigns each its own
    likelihood. Both are empty on a candidate the ranker has not judged —
    ``score`` 0.0, ``description`` ""; the pipeline collects every PR in the
    window first and the ranking stage fills these in, so an unranked candidate
    is a PR awaiting judgement, not one ruled out.
    """

    repo: str  # "owner/repo" slug on GitHub
    number: int
    title: str
    author: str
    url: str
    merged_at: str | None = None
    files: tuple[str, ...] = ()
    additions: int = 0
    deletions: int = 0
    score: float = 0.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "number": self.number,
            "title": self.title,
            "author": self.author,
            "url": self.url,
            "merged_at": self.merged_at,
            "files": list(self.files),
            "additions": self.additions,
            "deletions": self.deletions,
            "score": self.score,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CandidatePR:
        d = _only_known(cls, data)
        d["files"] = tuple(d.get("files") or ())
        return cls(**d)


@dataclass(frozen=True)
class RepoBlame:
    """One repository that moved across the blame window.

    ``repo`` is the ``owner/repo`` slug when the package lives on GitHub (the
    only forge whose PRs are resolvable), else ``None`` — the package still
    reports its commit range and ``compare_url`` (GitLab compare links resolve),
    it just has no ``candidates``. ``commits_unavailable`` marks a range whose
    PRs could not be enumerated at all — a compare that 404'd (``develop``
    force-pushed, base commit gone; both SHAs are still shown), a rate-limited
    or errored resolution; ``truncated`` marks a candidate list known to be
    incomplete — the range passed GitHub's 250-commit compare cap or a local
    resolution bound, or a discovered PR failed to fetch. Either flag means the
    candidate set must not be presented as the complete population of the range.
    """

    package: str  # Key4hep package name, e.g. "k4geo"
    repo: str | None
    base_commit: str | None
    head_commit: str | None
    compare_url: str | None
    status: str  # CHANGED / ADDED / REMOVED, from provenance.diff
    candidates: tuple[CandidatePR, ...] = ()
    commits_unavailable: bool = False
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "head_commit": self.head_commit,
            "compare_url": self.compare_url,
            "status": self.status,
            "candidates": [c.to_dict() for c in self.candidates],
            "commits_unavailable": self.commits_unavailable,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RepoBlame:
        d = _only_known(cls, data)
        d["candidates"] = tuple(
            CandidatePR.from_dict(c) for c in data.get("candidates") or ()
        )
        return cls(**d)


@dataclass(frozen=True)
class BlameEntry:
    """Blame for one confirmed regression.

    The first seven fields are a :class:`MetricVerdict`'s identity, so
    :meth:`BlameReport.entry_for` joins this entry to the verdict it explains.
    ``base_release`` / ``onset_release`` are the window's ends
    (``last_accepted_run_date`` and ``onset_run_date``); ``base_release`` is
    ``None`` for an open-ended window. ``n_unchanged`` is the count of tracked
    packages that did *not* move — context for sizing the diff, kept as a number
    rather than a list.
    """

    detector: str
    platform: str
    sample: str
    label: str
    metric: str
    sub_detector: str | None
    base_release: str | None
    onset_release: str
    repos: tuple[RepoBlame, ...] = ()
    n_unchanged: int = 0

    @property
    def key(self) -> tuple:
        """The verdict identity this entry attributes — the dashboard's join key."""
        return (
            self.detector, self.platform, self.sample,
            self.label, self.metric, self.sub_detector,
        )

    @property
    def candidates(self) -> list[CandidatePR]:
        """Every candidate PR across the changed repos, worst-first (highest
        score, then repo/number for a stable order) — the flat ledger the UI and
        the email render."""
        flat = [c for r in self.repos for c in r.candidates]
        return sorted(flat, key=lambda c: (-c.score, c.repo, c.number))

    @property
    def discovery_incomplete(self) -> bool:
        """True when any repo's candidate list is known not to be the full
        population of its range (unavailable or truncated) — the builder then
        refuses to rank, and completeness checks exempt this entry: calling one
        of a partial set "most likely" would be worse than no ranking."""
        return any(r.commits_unavailable or r.truncated for r in self.repos)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "platform": self.platform,
            "sample": self.sample,
            "label": self.label,
            "metric": self.metric,
            "sub_detector": self.sub_detector,
            "base_release": self.base_release,
            "onset_release": self.onset_release,
            "repos": [r.to_dict() for r in self.repos],
            "n_unchanged": self.n_unchanged,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BlameEntry:
        d = _only_known(cls, data)
        d["repos"] = tuple(RepoBlame.from_dict(r) for r in data.get("repos") or ())
        return cls(**d)


@dataclass(frozen=True)
class BlameReport:
    """One night's blame across every confirmed regression."""

    generated_at: str
    report_night: str
    entries: tuple[BlameEntry, ...] = field(default_factory=tuple)

    def entry_for(self, verdict) -> BlameEntry | None:
        """The entry attributing *verdict*, or ``None`` when this night has no
        blame for it.

        Matched on the shared identity tuple **and** the blame window: a rerun
        can re-anchor a verdict's window, and a sidecar left over from an
        earlier build must never have its ranking joined to a regression whose
        window it did not examine."""
        key = (
            verdict.detector, verdict.platform, verdict.sample,
            verdict.label, verdict.metric, verdict.sub_detector,
        )
        return next(
            (
                e for e in self.entries
                if e.key == key
                and e.base_release == verdict.last_accepted_run_date
                and e.onset_release == verdict.onset_run_date
            ),
            None,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "report_night": self.report_night,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_json(cls, data: dict) -> BlameReport:
        """Parse *data*, raising :class:`BlameSchemaError` when it is not shaped
        like a blame report — a top-level list, an entry missing required
        fields, a candidate that is not an object. Unknown *extra* keys are
        still dropped silently (forward compatibility); only structure that
        cannot be read raises."""
        try:
            return cls(
                generated_at=str(data.get("generated_at", "")),
                report_night=str(data.get("report_night", "")),
                entries=tuple(
                    BlameEntry.from_dict(e) for e in data.get("entries") or ()
                ),
            )
        except (AttributeError, KeyError, TypeError) as exc:
            raise BlameSchemaError(f"not a blame report: {exc}") from exc


def ranking_coverage(blame: BlameReport) -> tuple[int, int, list[str]]:
    """Return ``(ranked, expected, missing)`` over rankable candidate rows.

    The builder ranks each regression on its own, so every candidate of every
    entry is expected to carry the model's judgement — except entries whose
    :attr:`~BlameEntry.discovery_incomplete` is set: the builder deliberately
    leaves those unranked (a partial candidate set must not produce a "most
    likely" claim), so they are exempt rather than counted as failures.

    A zero score with a non-empty explanation is a valid ranking. An empty
    description is incomplete regardless of score: the contract asks the model
    to explain every judgement, including why a PR is unlikely.
    """
    expected: set[tuple] = set()
    ranked: set[tuple] = set()
    for entry in blame.entries:
        if entry.discovery_incomplete:
            continue
        for candidate in entry.candidates:
            key = (*entry.key, candidate.repo, candidate.number)
            expected.add(key)
            if candidate.description:
                ranked.add(key)
    missing = sorted({f"{key[-2]}#{key[-1]}" for key in expected - ranked})
    return len(ranked), len(expected), missing
