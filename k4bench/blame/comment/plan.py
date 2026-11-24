"""What one pull-request comment is decided from — the plan :func:`~k4bench.blame.comment.selection.select` builds and every later step reads.

A :class:`CommentPlan` holds one pull request, one change window, every
confirmed regression of that window as a :class:`RegressionRow`, and the
evidence around them; :class:`ScopeStanding` records how the first pass ranked
the pull request against its field in each rank group.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from k4bench.blame.attribute import (
    PackageChangeFact,
    ScopeCandidateState,
)
from k4bench.blame.comment.state import (
    ChangeWindow,
    _verdict_identity,
)
from k4bench.blame.evidence import (
    ExternalChange,
    ScopeOutcome,
)
from k4bench.blame.models import (
    CandidatePR,
    HistoricalRef,
    RankGroupKey,
)
from k4bench.blame.sweep import ScopeSweep
from k4bench.layout import SIM
from k4bench.regression.models import MetricVerdict


@dataclass(frozen=True)
class RegressionRow:
    """One confirmed regression this pull request is being asked about — one row
    of the comment's table.

    ``scope_state`` is what the first pass knew about this pull request *in this
    row's own scope* (:data:`~k4bench.blame.attribute.ScopeCandidateState`), and
    ``scope_score``/``scope_reason`` carry the ranker's judgement when that state
    is ``"ranked"`` — the prior the review starts from, and the likelihood shown
    when no review runs. Every other state leaves ``scope_score`` at ``None``:
    the row is real evidence about the window (it is why it is collected at all),
    but nobody scored this pull request against it, and a 0% in that cell would
    be an accusation's worth of difference from the truth.

    ``fact_id`` is the opaque handle the review echoes back (see
    :class:`~k4bench.blame.attribute.RegressionFact`); it is assigned in identity
    order, never score order, so the same night's rows always carry the same ids.
    ``stack`` is the group's Key4hep release *directory* — the dashboard's
    ``?stack=`` vocabulary, kept per row so each link names the stack that row
    actually ran against.
    """

    verdict: MetricVerdict
    stack: str
    fact_id: str = ""
    scope_score: float | None = None
    scope_reason: str = ""
    scope_state: ScopeCandidateState = "discovery_incomplete"
    #: ``release -> tracked packages that moved entering it`` across this
    #: metric's history tail, from the sidecar entry that examined it. This pass
    #: has no provenance access of its own, so without the entry's record every
    #: boundary would render as unread — and the release where the software was
    #: identical and the metric moved anyway is the sharpest evidence a step is
    #: noise. Empty when no entry covers this row (or the sidecar predates the
    #: field), which renders honestly as "not read".
    boundary_changes: dict[str, int] = field(default_factory=dict)

    @property
    def scope(self) -> tuple[str, str, str]:
        return (self.verdict.detector, self.verdict.platform, self.verdict.sample)


@dataclass(frozen=True)
class ScopeStanding:
    """This pull request against its field in one rank group, as the first pass
    scored it there.

    A competitor's score is only comparable with the subject's when one ranking
    call produced both: the first pass ranks each rank group on its own, so a
    95 on one detector and a 70 on another are answers to different questions.
    ``rival`` is the strongest *other* ranked candidate of the same group, and
    ``unranked_rivals`` whether that group also held candidates nobody scored."""

    score: float
    rival: CandidatePR | None
    unranked_rivals: bool

    @property
    def margin(self) -> float | None:
        """Subject minus rival, or ``None`` when the group had no ranked rival."""
        return None if self.rival is None else self.score - self.rival.score


@dataclass
class CommentPlan:
    """Everything one comment is decided from: one pull request, one change
    window, every regression of that window, and the evidence around them.

    ``outcomes`` is the negative evidence — the configurations that measured the
    same window and did *not* confirm — which only :func:`select` can compute,
    because only the report knows which groups ran at all.

    ``rows`` holds **every** confirmed regression whose onset falls in this
    window, whatever this pull request's standing in each — scored badly,
    unscored, or not a candidate there at all; ``selected`` records whether one
    complete, ranked judgement scored high enough to warrant the comment. The two
    are deliberately separate — see :func:`select`.

    A plan is keyed by pull request and window, never by platform, so its rows
    can come from several build platforms at once. The release diff is therefore
    kept per platform in ``package_facts`` and ``unchanged``, and handed to the
    review that way: two platforms' diffs are two measurements, not one.
    ``report_night`` is presentation metadata for an archived history link; it
    is deliberately not part of the facts digest.
    """

    repo: str
    number: int
    subject: CandidatePR
    base_release: str | None
    onset_release: str
    report_night: str = ""
    #: The suite whose report the rows are from: what the benchmark runs, and
    #: which comment lineage on the pull request this one continues.
    suite: str = SIM
    rows: list[RegressionRow] = field(default_factory=list)
    #: ``(repo, number) -> (candidate, scope label)`` — the strongest sighting
    #: of each competing pull request, and which run scope judged it that way.
    others: dict[tuple[str, int], tuple[CandidatePR, str]] = field(default_factory=dict)
    #: ``(repo, number) -> (detector, platform, sample)`` of the sighting kept in
    #: ``others``, for rendering where that score was given.
    others_scope: dict[tuple[str, int], tuple[str, str, str]] = field(default_factory=dict)
    #: ``rank group -> standing`` for every group whose first pass ranked this
    #: pull request (:class:`ScopeStanding`).
    standings: dict[RankGroupKey, ScopeStanding] = field(default_factory=dict)
    outcomes: tuple[ScopeOutcome, ...] = ()
    #: Loaded environment changes observed within this comment's window.
    external_changes: set[ExternalChange] = field(default_factory=set)
    #: ``platform -> {(package, status): compare_url}`` — one release diff per
    #: platform, exactly as provenance recorded it.
    package_facts: dict[str, dict[tuple[str, str], str | None]] = field(
        default_factory=dict
    )
    #: ``platform -> tracked packages that stood still`` on that platform.
    unchanged: dict[str, int] = field(default_factory=dict)
    #: Every platform that contributed a row with a sidecar entry — including
    #: those whose entries all measured a *narrower* window than this comment's
    #: and so contributed no package facts. Kept apart from ``package_facts`` so
    #: :attr:`packages_unavailable_on` can name them rather than let the prompt
    #: read as though those platforms had nothing to report.
    platforms_seen: set[str] = field(default_factory=set)
    #: ``(base platform, onset platform)`` for every row whose window was
    #: measured across a platform migration. Its release diff is then filed
    #: under :func:`_window_platform`, never under either platform alone.
    platform_switches: set[tuple[str, str]] = field(default_factory=set)
    #: ``(base detector, onset detector)`` for every row whose window was
    #: measured across a geometry version switch.
    detector_switches: set[tuple[str, str]] = field(default_factory=set)
    #: ``(repo, number) -> reference`` for the older-boundary pull requests the
    #: first pass read before scoring this window
    #: (:class:`~k4bench.blame.models.HistoricalRef`). De-duplicated across
    #: entries because one rank group's whole selection is recorded on each of
    #: its entries, and a comment can draw on several of them.
    #:
    #: These are **analogues, never targets**. They are collected from a field of
    #: their own, never from ``entry.candidates``, so neither :func:`_targets`
    #: nor :func:`_record_others` can ever see one — a pull request from before
    #: the window cannot be selected, accused, listed as an alternative, or shown
    #: in the dashboard's candidate ledger. The only thing they do is travel to
    #: the cross-configuration review, so that both passes weigh the same
    #: material.
    historical: dict[tuple[str, int], HistoricalRef] = field(default_factory=dict)
    #: The compact geometry files the rows' run groups load (see
    #: :attr:`~k4bench.regression.models.RunGroupReport.geometry_path`), sorted.
    #: Only :func:`select` can fill it, because only the report records them;
    #: the review samples each diff with these directories first.
    geometry_paths: tuple[str, ...] = ()
    #: Every scope of the report that measured this window
    #: (:func:`~k4bench.blame.sweep.window_sweeps`), and the compact file every
    #: benchmarked detector loads: the removal sweeps and controls the review
    #: is shown. Evidence for the review only — neither enters the facts
    #: digest, which already covers the rows and outcomes they are built from.
    sweeps: tuple[ScopeSweep, ...] = ()
    geometry: dict[str, str] = field(default_factory=dict)
    selected: bool = False

    @property
    def target(self) -> str:
        return f"{self.repo}#{self.number}"

    @property
    def window(self) -> ChangeWindow:
        return (self.base_release, self.onset_release)

    @property
    def historical_refs(self) -> tuple[HistoricalRef, ...]:
        """The historical references in a stable identity order.

        Ordered by identity rather than by the order the sidecar happened to list
        them: the digest hashes this, and a night that reordered an unchanged
        evidence set would edit a comment whose body did not move."""
        return tuple(
            sorted(
                self.historical.values(),
                key=lambda h: (h.repo.lower(), h.pr, h.boundary_id),
            )
        )

    @property
    def packages_by_platform(self) -> dict[str, tuple[PackageChangeFact, ...]]:
        """The window's package diff, per platform, as the review is shown it.

        Never unioned across platforms. Provenance is read per platform, so a
        package can move on one and stand still on another, or move to a
        different status — and a union paired with a single unchanged count
        would quote a "N of M tracked" ratio that no platform ever measured."""
        return {
            platform: tuple(
                PackageChangeFact(package=package, status=status, compare_url=url)
                for (package, status), url in sorted(facts.items())
            )
            for platform, facts in sorted(self.package_facts.items())
        }

    @property
    def packages_unavailable_on(self) -> tuple[str, ...]:
        """Platforms that regressed in this window but whose release diff for
        *exactly* this window was never read.

        Their rows entered on narrower ranges of their own, and those ranges'
        package sets are not this window's. Named rather than silently missing:
        "no diff was read for this platform" and "nothing changed on this
        platform" are opposite claims."""
        return tuple(sorted(self.platforms_seen - set(self.package_facts)))

    @property
    def platforms(self) -> set[str]:
        return {row.verdict.platform for row in self.rows}

    @property
    def top_score(self) -> float:
        """The strongest per-configuration likelihood across this window — what
        selection was made on, and how comments are ordered. Only rows the first
        pass actually judged carry one."""
        return max(
            (row.scope_score for row in self.rows if row.scope_score is not None),
            default=0.0,
        )

    @property
    def scopes(self) -> set[tuple[str, str, str]]:
        return {row.scope for row in self.rows}


def _row_identity(row: RegressionRow) -> tuple:
    return _verdict_identity(row.verdict)
