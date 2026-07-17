"""Serialized shapes for ``_reports/{night}/blame.json``.

The file is a :class:`BlameReport`: one :class:`BlameEntry` per confirmed
regression, each carrying the release window it entered in and the repos that
moved across that window, and within each repo the ranked candidate pull
requests. The identity fields on :class:`BlameEntry` (``detector`` … ``metric``,
``sub_detector``) are exactly a :class:`~k4bench.regression.models.MetricVerdict`'s
identity, so the dashboard joins an entry back to the verdict it explains with
:meth:`BlameReport.entry_for` — the two files stay decoupled, keyed only by that
tuple.

Everything here is a plain, frozen dataclass with explicit JSON round-tripping.
:func:`from_json` drops unknown keys rather than raising: ``blame.json`` is read
by whatever dashboard is deployed, not necessarily one built from the commit
that wrote the file, so a schema that gains a field must not break older readers
(the same forward-compatibility rule :mod:`k4bench.regression.render` follows for
``report.json``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


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
    it just has no ``candidates``. ``commits_unavailable`` marks a compare that
    404'd (``develop`` force-pushed, base commit gone — both SHAs are still
    shown); ``truncated`` marks a range past GitHub's 250-commit compare cap.
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
        """The entry attributing *verdict*, matched on the shared identity
        tuple, or ``None`` when this night has no blame for it."""
        key = (
            verdict.detector, verdict.platform, verdict.sample,
            verdict.label, verdict.metric, verdict.sub_detector,
        )
        return next((e for e in self.entries if e.key == key), None)

    def to_json(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "report_night": self.report_night,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_json(cls, data: dict) -> BlameReport:
        return cls(
            generated_at=str(data.get("generated_at", "")),
            report_night=str(data.get("report_night", "")),
            entries=tuple(BlameEntry.from_dict(e) for e in data.get("entries") or ()),
        )


def ranking_coverage(blame: BlameReport) -> tuple[int, int, list[str]]:
    """Return ``(ranked, expected, missing)`` for distinct window candidates.

    Candidate rows repeat for every metric sharing one release window, while the
    builder performs one model inference per window. Count each
    ``(platform, base, onset, repo, PR)`` once so completeness reflects actual
    model work rather than serialized duplication.

    A zero score with a non-empty explanation is a valid ranking. An empty
    description is incomplete regardless of score: the contract asks the model
    to explain every judgement, including why a PR is unlikely.
    """
    expected: set[tuple] = set()
    ranked: set[tuple] = set()
    for entry in blame.entries:
        window = (entry.platform, entry.base_release, entry.onset_release)
        for candidate in entry.candidates:
            key = (*window, candidate.repo, candidate.number)
            expected.add(key)
            if candidate.description:
                ranked.add(key)
    missing = [f"{key[-2]}#{key[-1]}" for key in sorted(expected - ranked)]
    return len(ranked), len(expected), missing
