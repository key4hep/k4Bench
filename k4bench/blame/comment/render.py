"""A drafted comment as the GitHub-flavoured Markdown that is posted.

:class:`CommentDraft` is what :func:`~k4bench.blame.comment.build.build_comments`
hands the publisher, and :meth:`CommentDraft.render` turns it into a
:class:`PRComment` once the comments already posted in its lineage are known —
including the facts digest that decides whether a standing comment is edited.
Externally authored prose (titles, model summaries) is defanged on its way in.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from k4bench.blame.attribute import Attribution
from k4bench.blame.comment.competitors import (
    _crowded_note,
    _others_section,
)
from k4bench.blame.comment.digest import (
    Reproducers,
    _digest_from,
    _retained_fact,
)
from k4bench.blame.comment.headline import (
    _alert,
    _assessment,
    _window_line,
)
from k4bench.blame.comment.lineage import (
    _association_summary,
    _observation_for,
    _observation_history,
    _retained_state,
)
from k4bench.blame.comment.markdown import REGRESSION, _noun_for
from k4bench.blame.comment.plan import (
    CommentPlan,
    _row_identity,
)
from k4bench.blame.comment.selection import (
    _DEFAULT_MIN_SCORE,
)
from k4bench.blame.comment.state import (
    _FACTS_MARKER_PREFIX,
    _MAX_OBSERVATIONS,
    CommentObservation,
    CumulativeEntry,
    LineageState,
    RetainedRow,
    _cumulative_marker,
    _observation_key,
    _observation_marker,
    _retained_identity,
    _retained_marker,
    _retained_sort_key,
)
from k4bench.blame.comment.supersede import moved_before, superseded_note, withdrawn
from k4bench.blame.comment.table import (
    _link_definitions,
    _past_links,
    _row_links,
    _row_sort_key,
    _score_source,
    _selected_rows,
    _table,
)
from k4bench.blame.models import (
    RANKING_DISCLOSURE,
)
from k4bench.regression.render import (
    nightly_report_href,
)

#: Where the comment's own footer points: the page describing how a regression
#: is attributed to a pull request, on the published docs site (``site_url`` in
#: mkdocs.yml + the page's nav path).
_METHOD_URL = "https://key4hep.github.io/k4Bench/user-guide/features/pr-comments/"

#: Who to write to about a comment. Rendered as a ``mailto:`` link, so a reader
#: who thinks the bot got it wrong can reach a person in one click rather than
#: replying into a thread nobody may be watching.
_CONTACT_EMAIL = "jbeirer@cern.ch"


@dataclass(frozen=True)
class PRComment:
    """One rendered comment and where it goes.

    ``marker`` is the hidden key the upsert recognises and ``facts_digest``
    fingerprints the benchmark facts behind the body; both are hidden lines at
    the top of ``body``, so a comment always carries the keys that identify it
    and the state it was rendered from. An empty ``facts_digest`` makes the
    upsert compare whole bodies instead (:attr:`CommentDraft.keyed_by_facts`).
    """

    repo: str
    number: int
    marker: str
    body: str
    score: float
    facts_digest: str = ""

    @property
    def target(self) -> str:
        """``owner/repo#123`` — how this comment is named in logs."""
        return f"{self.repo}#{self.number}"


@dataclass(frozen=True)
class CommentDraft:
    """One comment decided and reviewed, rendered only when it is written.

    The body draws on state the comments already posted in its lineage carry —
    the exact regression identities behind the headline, the rows retained from
    earlier versions, and the observation history — and only the publisher has
    read those. So :func:`build_comments` stops here, and :meth:`render` builds
    the body once the prior bodies are known. A dry run renders with none, which
    is exactly what a first comment says.

    ``facts_payload`` is the canonical set of benchmark facts the digest hashes
    (:func:`_facts_payload`), settled at build time from the plan, the review
    request and the published recipes. ``keyed_by_facts`` is off only for a
    preview, which iterates on wording against deliberately unchanged facts and
    must therefore be compared on its whole body
    (:func:`k4bench.blame.publish._upsert`).
    """

    repo: str
    number: int
    marker: str
    plan: CommentPlan
    attribution: Attribution | None
    facts_payload: dict[str, Any]
    #: The recipes published for this comment's rows (:class:`Reproducers`).
    reproduce: Reproducers | None = None
    dashboard_url: str | None = None
    min_score: float = _DEFAULT_MIN_SCORE
    keyed_by_facts: bool = True

    @property
    def target(self) -> str:
        """``owner/repo#123`` — how this comment is named in logs."""
        return f"{self.repo}#{self.number}"

    @property
    def score(self) -> float:
        return self.plan.top_score

    @property
    def observation(self) -> CommentObservation | None:
        """This version as the observation history records it, or ``None`` for
        a report with no night to date it by."""
        return _observation_for(self.plan, self.plan.rows, self.dashboard_url)

    def render(self, previous_bodies: Sequence[str] = ()) -> PRComment:
        """The comment as it will be written, given the comparable bodies
        already posted — the survivor of this lineage first, then any other
        comparable comment a converging window is absorbing."""
        return _render(self, LineageState.from_bodies(previous_bodies))


def _render(draft: CommentDraft, lineage: LineageState) -> PRComment:
    """One draft as a GitHub-flavoured Markdown comment, given its lineage.

    A single comment for the ``(pull request, window)``: the claim, the window,
    the model's reasoning and any qualifier on it, the table of the most likely
    rows, the association with each scope, the observation history and the
    competing pull requests.

    Everything that draws on earlier versions comes from structured hidden state
    (:class:`LineageState`), never from their Markdown:

    * the **headline counts and association table** describe the exact identity
      union across the lineage and tonight's report, using the newest published
      attribution for a repeated identity;
    * **retained rows** rejoin the table's selection pool beside tonight's
      confirmed rows, and a fresh bounded state marker is written for the next
      version;
    * the **facts digest** covers the table as it renders: which historical rows
      render, their frozen facts and their standing in tonight's report are
      deterministic inputs a reader can see — while a retained row's mere
      heartbeat (tonight's date) is not;
    * the **observation history** carries every material version forward.

    Archive links pin the report being rendered. Report dates stay outside the
    digest, so they alone never make an unchanged regression re-notify a pull
    request.
    """
    plan, attribution = draft.plan, draft.attribution
    by_likelihood = sorted(
        plan.rows, key=lambda row: _row_sort_key(row, attribution)
    )

    # Read before the table, so the line under it can count each report the
    # rows came from, and again by the history below — one merge, so the two
    # can never disagree about what a report contained.
    observations, omitted = _merged_observations(draft, lineage)

    # Any identity confirmed tonight is superseded by its current evidence —
    # including a current ``not_candidate`` or unscored standing, which must
    # not be papered over by an old score.
    current = {_row_identity(row) for row in plan.rows}
    # A lineage identity tonight's report places before this window, outside
    # this pull request's candidates, loses its old score the same way
    # (:mod:`.supersede`).
    moved = _moved_identities(plan, lineage, current)
    moved_attributed = sorted(
        identity for identity in moved
        if _was_attributed(identity, lineage, draft.min_score)
    )
    historical = sorted(
        (
            replace(row, likelihood=None, source="", state="not_candidate")
            if _retained_identity(row) in moved else row
            for row in lineage.retained if _retained_identity(row) not in current
        ),
        key=_retained_sort_key,
    )
    noun = _comment_noun(plan, historical, observations)
    rises_seen = _rises_seen(plan, historical, observations)

    # Tonight's rows overlay the lineage's: a reconfirmed identity keeps only
    # its newest published attribution. Directions an older marker did not
    # record are filled from the retained snapshots, which carry them.
    cumulative = withdrawn(
        _cumulative_with_directions(lineage.cumulative, historical), moved,
    )
    tonight = {
        _row_identity(row): CumulativeEntry(
            *_score_source(row, attribution), plan.report_night,
            _direction_of(row.verdict),
        )
        for row in plan.rows
    }
    cumulative.update(tonight)
    cumulative_marker = _cumulative_marker(cumulative)
    payload = dict(draft.facts_payload)
    if moved_attributed:
        payload["superseded"] = [list(identity) for identity in moved_attributed]
    if cumulative_marker:
        identities = sorted(cumulative)
        scopes = len({identity[:3] for identity in identities})
        reviewed = [entry.likelihood for entry in cumulative.values()
                    if entry.likelihood is not None and entry.source == "reviewer"]
        carried = [entry.likelihood for entry in cumulative.values()
                   if entry.likelihood is not None and entry.source == "ranker"]
        attributed_scopes = len({
            identity[:3] for identity, entry in cumulative.items()
            if entry.likelihood is not None and entry.likelihood >= draft.min_score
        })
        alert = _alert(
            plan, by_likelihood, attribution, min_score=draft.min_score,
            n_regressions=len(identities), n_scopes=scopes,
            scores=(reviewed, carried), n_attributed_scopes=attributed_scopes,
            noun=noun, directions=_attributed_directions(cumulative, draft.min_score),
            rises_seen=rises_seen,
        )
        association = _association_summary(
            cumulative, draft.min_score, draft.dashboard_url, noun=noun,
        )
        canonical = json.dumps(identities, separators=(",", ":"))
        payload["cumulative"] = {
            "digest": hashlib.sha256(canonical.encode()).hexdigest()[:16],
            "regressions": len(identities),
            "scopes": scopes,
        }
    else:
        # A union too large to carry costs this night's merge — the state is
        # still in the bodies it would have merged — so the headline falls back
        # to counting tonight's report alone.
        alert = _alert(
            plan, by_likelihood, attribution, min_score=draft.min_score,
            noun=noun, directions=_attributed_directions(tonight, draft.min_score),
            rises_seen=rises_seen,
        )
        association = _association_summary(
            tonight, draft.min_score, draft.dashboard_url, noun=noun,
        )

    region, shown_past = _details_region(
        plan, attribution, dashboard_url=draft.dashboard_url,
        historical=historical,
        retained_marker=_retained_marker(_retained_state(
            plan, attribution, historical, draft.reproduce,
        )),
        reproduce=draft.reproduce, observations=observations,
        min_score=draft.min_score, noun=noun,
        superseded=(
            superseded_note(len(moved_attributed), plan.window, noun)
            if moved_attributed else None
        ),
    )
    digest = _digest_from(payload, [_retained_fact(row) for row in shown_past])

    history = None
    if draft.observation is not None:
        # A history of one repeats the window and counts the comment already
        # states above it, and has nothing to compare them against; it earns its
        # place from the second material update on. The hidden markers are still
        # emitted — they are the lineage's observation state, and dropping them
        # would restart the history at one entry every night, permanently.
        history = (
            "\n".join(_observation_marker(item) for item in observations)
            if len(observations) < 2 and not omitted
            else _observation_history(observations, omitted=omitted, noun=noun)
        )

    body = "\n".join(
        part for part in (
            draft.marker,
            f"{_FACTS_MARKER_PREFIX}{digest} -->",
            cumulative_marker or None,
            "### 📊 Possible performance change traced to this pull request",
            "",
            alert,
            "",
            _window_line(plan),
            region,
            # Below the table, not above it: the assessment is the reasoning
            # behind the alert and belongs next to it, and this summary is the
            # drill-down on the alert's counts — which reads after the rows it
            # generalizes, beside the per-report line asking the same question.
            association,
            history,
            _others_section(plan, draft.dashboard_url),
            "",
            "---",
            "",
            # The contact line is this renderer's own, not part of the shared
            # disclosure. A machine-written accusation needs a human at the end
            # of it, and a named address reaches one whether or not anyone is
            # watching the thread. k4Bench's name carries the page describing
            # how this attribution is made rather than the repository root:
            # someone who doubts the claim wants the method, and a README makes
            # them go looking for it.
            f"<sub>🤖 {RANKING_DISCLOSURE} Posted automatically by "
            f"[k4Bench]({_METHOD_URL}) — questions or feedback: "
            f"[{_CONTACT_EMAIL}](mailto:{_CONTACT_EMAIL})</sub>",
        ) if part is not None
    )
    return PRComment(
        repo=draft.repo,
        number=draft.number,
        marker=draft.marker,
        body=body,
        score=draft.score,
        facts_digest=digest if draft.keyed_by_facts else "",
    )


def _details_region(
    plan: CommentPlan,
    attribution: Attribution | None,
    *,
    dashboard_url: str | None,
    historical: list[RetainedRow],
    retained_marker: str | None,
    reproduce: Reproducers | None = None,
    observations: Sequence[CommentObservation] = (),
    min_score: float = _DEFAULT_MIN_SCORE,
    noun: str = REGRESSION,
    superseded: str | None = None,
) -> tuple[str, list[RetainedRow]]:
    """The middle of a comment — assessment, regression table and its
    reference links — rendered from structured inputs alone. Returns the region
    and the historical rows it actually rendered, which are what the digest
    must cover."""
    by_likelihood = sorted(
        plan.rows, key=lambda row: _row_sort_key(row, attribution)
    )
    selection = _selected_rows(by_likelihood, historical, attribution, min_score)
    shown = selection.rows
    shown_current = [entry.current for entry in shown if entry.current is not None]
    shown_past = [entry.past for entry in shown if entry.past is not None]
    links = _row_links(plan, shown_current, dashboard_url)
    past_links = _past_links(shown_past, dashboard_url)
    definitions = dict(links)
    definitions.update(dict(past_links.values()))
    region = "\n".join(
        part for part in (
            retained_marker,
            _assessment(
                plan, by_likelihood, attribution, rendered=shown_current, noun=noun,
            ),
            _crowded_note(plan),
            _table(
                plan, by_likelihood, attribution,
                selection=selection, links=links, past_links=past_links,
                dashboard_url=dashboard_url, reproduce=reproduce,
                observations=observations, min_score=min_score, noun=noun,
            ),
            f"\n_{superseded}_" if superseded else None,
            _link_definitions(definitions),
        ) if part is not None
    )
    return region, shown_past


def _direction_of(verdict_or_row) -> str:
    """A verdict's (or retained row's) direction as its plain string value."""
    direction = getattr(verdict_or_row, "direction")
    return str(getattr(direction, "value", direction))


def _moved_identities(
    plan: CommentPlan, lineage: LineageState, current: set[tuple],
) -> set[tuple]:
    """The lineage identities not among tonight's rows that tonight's report
    places before this window, outside this pull request's candidates
    (:func:`~.supersede.moved_before`)."""
    pr = (plan.repo.lower(), plan.number)
    known = set(lineage.cumulative) | {
        _retained_identity(row) for row in lineage.retained
    }
    return {
        identity for identity in known - current
        if moved_before(plan.placements.get(identity), plan.window, pr)
    }


def _was_attributed(identity: tuple, lineage: LineageState, min_score: float) -> bool:
    """Whether the lineage last published *identity* at or above *min_score*."""
    entry = lineage.cumulative.get(identity)
    if entry is not None and entry.likelihood is not None:
        return entry.likelihood >= min_score
    return any(
        _retained_identity(row) == identity and row.state != "not_candidate"
        and row.likelihood is not None and row.likelihood >= min_score
        for row in lineage.retained
    )


def _comment_noun(
    plan: CommentPlan,
    historical: Sequence[RetainedRow],
    observations: Sequence[CommentObservation],
) -> str:
    """What this comment calls its steps, decided once over everything it
    describes — tonight's rows, the retained ones, and every report the
    lineage observed — so the alert, the table and the counts below it use
    one word (:func:`~.markdown._noun_for`)."""
    directions = [_direction_of(row.verdict) for row in plan.rows]
    directions += [row.direction for row in historical]
    for item in observations:
        directions += ["DOWN"] * bool(item.down) + ["NONE"] * bool(item.none)
    return _noun_for(directions)


def _rises_seen(
    plan: CommentPlan,
    historical: Sequence[RetainedRow],
    observations: Sequence[CommentObservation],
) -> bool:
    """Whether anything this comment describes went up — tonight's rows, the
    retained ones, or any report the lineage observed. When nothing did, an
    attributed step whose direction the marker never recorded cannot have been
    a rise either."""
    return (
        any(_direction_of(row.verdict) == "UP" for row in plan.rows)
        or any(row.direction == "UP" for row in historical)
        or any(item.up for item in observations)
    )


def _cumulative_with_directions(
    cumulative: dict[tuple, CumulativeEntry],
    historical: Sequence[RetainedRow],
) -> dict[tuple, CumulativeEntry]:
    """The lineage's cumulative entries, with a direction the marker did not
    record taken from the retained snapshot of the same identity."""
    known = {_retained_identity(row): row.direction for row in historical}
    return {
        identity: entry if entry.direction else entry._replace(
            direction=known.get(identity, ""),
        )
        for identity, entry in cumulative.items()
    }


def _attributed_directions(
    cumulative: dict[tuple, CumulativeEntry], min_score: float,
) -> Counter[str]:
    """The signs of the steps attributed at or above *min_score*, with
    ``"unknown"`` counting entries whose direction was never recorded."""
    return Counter(
        entry.direction or "unknown" for entry in cumulative.values()
        if entry.likelihood is not None and entry.likelihood >= min_score
    )


def _merged_observations(
    draft: CommentDraft, lineage: LineageState
) -> tuple[list[CommentObservation], int]:
    """This comment's material versions, newest first, and how many aged out.

    Every comparable body's observations carried forward and tonight's laid over
    them, keyed by report night and window: re-running a changed report for the
    same night replaces that night's entry, and a later material change appends
    one. Archive links are re-pointed on the way through, so observations
    recorded against an older link shape migrate on the next material edit.

    Absorbed lineages are read too, because convergence hands their *rows* to
    this comment (:func:`_merged_retained`) and a table drawing a row from a
    report the history never recorded leaves both the history and the line under
    the table describing a lineage this comment no longer has.

    Entries are keyed by **report night and window**, not by night alone. The
    publisher only converges windows that are containment-comparable with the
    current one, but two absorbed windows can still be siblings — ``(09-01,
    09-02]`` and ``(09-02, 09-03]`` are each contained by ``(09-01, 09-03]``
    and neither contains the other — and those describe *disjoint* sets of
    regressions observed on the same night. Keying by night alone would drop one
    of them, taking its rows' provenance with it.
    """
    observations = {
        key: replace(
            item, url=nightly_report_href(
                draft.dashboard_url or item.url, item.report_night,
            ),
        )
        for key, item in lineage.observations.items()
    }
    omitted = lineage.omitted
    observation = draft.observation
    if observation is not None:
        observations[_observation_key(observation)] = observation
    ordered = sorted(
        observations.values(),
        key=lambda item: (
            item.report_night, item.base_release or "", item.onset_release,
        ),
        reverse=True,
    )
    if len(ordered) > _MAX_OBSERVATIONS:
        omitted += len(ordered) - _MAX_OBSERVATIONS
        ordered = ordered[:_MAX_OBSERVATIONS]
    return ordered, omitted
