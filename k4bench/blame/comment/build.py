"""From selected plans to reviewed drafts.

:func:`build_comments` runs the optional cross-configuration review
(:mod:`k4bench.blame.attribute`) for each plan and publishes the reproducer
recipes its table links. The diff, description, run-record and recipe sources
arrive as *injected callables*, so this module stays free of network I/O.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

from k4bench.blame.attribute import (
    MAX_COMPETITORS,
    Attribution,
    AttributionRequest,
    Attributor,
    RegressionFact,
)
from k4bench.blame.comment.digest import (
    Reproducers,
    _facts_payload,
)
from k4bench.blame.comment.render import (
    _TARGET_TABLE_ROWS,
    CommentDraft,
    _likelihood,
    _onset_label,
    _row_sort_key,
    _selected_rows,
)
from k4bench.blame.comment.selection import (
    _DEFAULT_MIN_SCORE,
    CommentPlan,
    RegressionRow,
    _competitor,
    _row_identity,
    _sorted_others,
)
from k4bench.blame.comment.state import marker_for
from k4bench.regression.render import report_views_url
from k4bench.blame.evidence import history_from_verdict
from k4bench.blame.geometry import compact_dir, detector_touches, geometry_tree
from k4bench.blame.github import FilePatch, diff_sample
from k4bench.blame.history import MAX_COMMENT_ANALOGUES, HistoricalPR
from k4bench.blame.models import CandidatePR
from k4bench.blame.prompt import movement
from k4bench.blame.reproduce import RecipeFacts
from k4bench.blame.reproduce import facts_from as reproducer_facts_from
from k4bench.layout import stack_dir

_log = logging.getLogger(__name__)

#: How a caller supplies one pull request's diff — ``(repo, number) -> patch``,
#: empty when it could not be fetched. Injected rather than imported so this
#: module stays free of the network, and so the CLI can memoize a night's
#: fetches (one window's subject is another's competitor).
PatchFor = Callable[[str, int], str]

#: And how it supplies one's description — same shape, same best-effort contract,
#: kept a separate callable rather than widening :data:`PatchFor` so every
#: existing caller keeps working and a run without descriptions is simply a run
#: whose prompts carry none.
BodyFor = Callable[[str, int], str]

#: And how it supplies one's per-file hunks — ``(repo, number) -> hunks``, empty
#: when they could not be fetched — so the review can sample each diff with the
#: rows' own geometry first (:func:`~k4bench.blame.github.diff_sample`). Optional
#: like the others: without it, or with nothing returned, :data:`PatchFor`'s
#: generic sample stands.
FilesFor = Callable[[str, int], tuple[FilePatch, ...]]

#: How a caller supplies one exact run's immutable metadata —
#: ``(detector, platform, stack, sample, run_id) -> run_info``. Like the diff
#: seams above it is injected so this module remains free of network I/O.
RunInfoFor = Callable[[str, str, str, str, str], dict | None]

#: Publishes one row's reproducer recipe and returns the URL it can be read at,
#: or ``None`` when it could not be published. Called *before* the comment is
#: rendered, so the table only ever links a recipe that is already readable —
#: a comment must not carry a link that 404s while a reader is looking at it.
ReproducerUrlFor = Callable[[RecipeFacts], str | None]


def build_comments(
    plans: list[CommentPlan],
    *,
    attributor: Attributor | None = None,
    patch_for: PatchFor | None = None,
    body_for: BodyFor | None = None,
    files_for: FilesFor | None = None,
    run_info_for: RunInfoFor | None = None,
    reproducer_url_for: ReproducerUrlFor | None = None,
    dashboard_url: str | None = None,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> list[CommentDraft]:
    """Draft *plans*, reviewing each against the whole window if it can.

    With an *attributor* configured, every plan gets one cross-configuration
    review (:mod:`k4bench.blame.attribute`) whose likelihoods order and fill the
    table. What the review cannot do is *widen*: it never introduces a pull
    request selection did not already implicate, and the only outcome it adds is
    withdrawal — a plan is dropped when no row is left standing at or above
    *min_score*. Within a plan it is a genuine second opinion, and an individual
    row's likelihood may come back higher than the first pass's as well as
    lower: that pass scored the row without ever seeing the other
    configurations, which is the deficiency this one exists to correct.

    **A configured review that does not answer produces no comment at all.**
    Not a fallback rendered from the first-pass scores — nothing. Two reasons,
    and the second is the load-bearing one:

    * The first pass asks the weaker question. It is enough to *select* a pull
      request; the whole premise of this stage is that a claim posted into
      someone else's repository deserves the cross-configuration reading too.
      An endpoint that is down is not a reason to lower that bar.
    * It makes comment quality **monotonic**, which nothing else here can. A
      degraded comment posted tonight and a reviewed comment rendered tomorrow
      rest on the *same* benchmark facts, so the digest is the same and
      :mod:`k4bench.blame.publish` would refuse the edit — the degraded body
      would stand forever, however many later reviews succeeded. Skipping the
      night instead means a standing comment is simply left alone and the next
      working review posts the real thing. A comment can therefore only ever
      improve: no review, then reviewed, and never back again.

    With no attributor configured at all the comment renders from the
    per-configuration scores, a coherent mode in its own right — every comment
    in it rests on the same evidence as every other.

    "Left standing" is the operative phrase in the withdrawal gate: it reads
    each row's *effective* likelihood, the review's score where it gave one and
    the per-configuration score where it did not. A partial reply is an accepted
    outcome (:func:`~k4bench.blame.attribute._parse_attribution`), so measuring
    withdrawal on the review's scores alone would let one low answer about one
    row acquit a pull request the review never disputed on the others.

    Each draft is rendered when it is written (:meth:`CommentDraft.render`),
    once the publisher has read the comments already posted in its lineage.
    """
    comments = []
    for plan in plans:
        attribution, request = _review(
            plan, attributor=attributor, patch_for=patch_for, body_for=body_for,
            files_for=files_for,
        )
        if attributor is not None and attribution is None:
            _log.warning(
                "build_comments: %s — the cross-configuration review produced "
                "nothing usable; posting no comment tonight rather than a "
                "first-pass-only one a later review could never replace",
                plan.target,
            )
            continue
        # Measured on what the table will actually show, not on the review's
        # own scores: a partial reply leaves the rows it omitted at their
        # per-configuration likelihood (:func:`_likelihood`), and a row the
        # review never spoke about must not be able to withdraw a comment the
        # ranker put at 91% and the review left standing.
        effective_top = max(
            (
                likelihood for row in plan.rows
                if (likelihood := _likelihood(row, attribution)) is not None
            ),
            default=0.0,
        )
        if attribution is not None and effective_top < min_score:
            _log.info(
                "build_comments: %s withdrawn — the cross-configuration review "
                "left every regression under %g%% (highest %.0f%%)",
                plan.target, min_score, effective_top,
            )
            continue
        if attribution is not None and attribution.assessment is not None \
                and attribution.assessment.likely_noise:
            # The review saw every configuration's history — strictly more than
            # the first pass, which reads one configuration at a time — and
            # concluded the movements are most likely the series' own noise. It
            # may still have scored a row highly; those two readings together are
            # exactly the case where an accusation should not be posted, and the
            # review is the better-informed of the two. Withdrawal only ever
            # narrows the bot, which is the property this pass is allowed to
            # affect.
            _log.info(
                "build_comments: %s withdrawn — the cross-configuration review "
                "reads these movements as likely measurement noise: %s",
                plan.target, attribution.assessment.reason or "no reason given",
            )
            continue
        reproduce = _publish_reproducers(
            plan, attribution=attribution,
            run_info_for=run_info_for, reproducer_url_for=reproducer_url_for,
        )
        comments.append(CommentDraft(
            repo=plan.repo,
            number=plan.number,
            marker=marker_for(plan.base_release, plan.onset_release, plan.suite),
            plan=plan,
            attribution=attribution,
            facts_payload=_facts_payload(plan, request, reproduce),
            reproduce=reproduce,
            # Every link a comment carries is into a report view, so a suite
            # whose page has none gets no links at all.
            dashboard_url=report_views_url(dashboard_url, plan.suite),
            min_score=min_score,
        ))
    return comments


#: How many recipes one comment publishes. Twice the table's height, so the
#: deterministic set below almost always covers whichever rows the
#: likelihood-ranked table ends up showing, while a detector-removal sweep of
#: three hundred near-identical rows still uploads ten small files.
_MAX_RECIPES = 2 * _TARGET_TABLE_ROWS


def _recipe_rows(plan: CommentPlan) -> list[RegressionRow]:
    """The rows whose recipes are **hashed** — chosen without any model score.

    Ranked on benchmark facts alone: one row reserved per step onset, then the
    largest movements, ties broken by identity. See :class:`Reproducers` for
    why the hashed set has to be model-independent while the rendered one does
    not."""
    ordered = sorted(
        plan.rows,
        key=lambda row: (-movement(row.verdict.pct_change), *_row_identity(row)),
    )
    chosen: dict[tuple, RegressionRow] = {}
    seen: set[str] = set()
    for row in ordered:                      # one per onset, largest first
        onset = _onset_label(row)
        if onset not in seen and len(chosen) < _MAX_RECIPES:
            seen.add(onset)
            chosen[_row_identity(row)] = row
    for row in ordered:                      # then fill on movement
        if len(chosen) >= _MAX_RECIPES:
            break
        chosen.setdefault(_row_identity(row), row)
    return [chosen[key] for key in sorted(chosen)]


def _publish_reproducers(
    plan: CommentPlan,
    *,
    attribution: Attribution | None,
    run_info_for: RunInfoFor | None,
    reproducer_url_for: ReproducerUrlFor | None,
) -> Reproducers | None:
    """Build and publish this comment's recipes.

    Two sets are published: the rows the table will show, so every rendered row
    can link its own commands, and the deterministic :func:`_recipe_rows`, which
    is what the digest hashes. They overlap heavily; the union is what gets
    uploaded, and both are bounded, so a detector-removal sweep of three
    hundred near-identical rows still uploads a handful of small files.

    Everything happens *before* the comment is rendered, so a comment either
    carries links that already resolve or carries none. Each row fails on its
    own: a run record that cannot be read, or a publisher that refuses, costs
    that row's link and leaves the rest of the table alone."""
    if run_info_for is None or reproducer_url_for is None or not plan.rows:
        return None
    hashed = _recipe_rows(plan)
    hashed_ids = {_row_identity(row) for row in hashed}
    by_likelihood = sorted(
        plan.rows, key=lambda row: _row_sort_key(row, attribution)
    )
    rendered = [
        entry.current
        for entry in _selected_rows(by_likelihood, [], attribution)
        if entry.current is not None
    ]
    ordered: dict[tuple, RegressionRow] = {}
    for row in [*hashed, *rendered]:
        ordered.setdefault(_row_identity(row), row)

    urls: dict[tuple, str] = {}
    facts: list[RecipeFacts] = []
    for identity, row in ordered.items():
        recipe = _reproducer_for(plan, row, run_info_for=run_info_for)
        if recipe is None:
            continue
        try:
            url = reproducer_url_for(recipe)
        except Exception as exc:  # noqa: BLE001 — optional artifact, never fatal
            _log.info("build_comments: reproducer not published for %s %s (%s)",
                      plan.target, row.verdict.label, exc)
            continue
        if not url:
            _log.info(
                "build_comments: reproducer not published for %s %s — no link "
                "to give", plan.target, row.verdict.label,
            )
            continue
        urls[identity] = url
        if identity in hashed_ids:
            facts.append(recipe)
    if not urls:
        return None
    return Reproducers(
        urls=urls,
        hashed=tuple(sorted(hashed_ids & urls.keys())),
        facts=tuple(facts),
    )


def _reproducer_for(
    plan: CommentPlan,
    row: RegressionRow,
    *,
    run_info_for: RunInfoFor,
) -> RecipeFacts | None:
    """Best-effort reproducer facts for one row."""
    verdict = row.verdict
    if not verdict.last_accepted_run_id or not verdict.last_accepted_run_date \
            or not verdict.onset_run_id or not verdict.onset_run_date:
        return None
    try:
        base_info = run_info_for(
            verdict.base_detector, verdict.base_platform,
            stack_dir(verdict.last_accepted_run_date), verdict.sample,
            verdict.last_accepted_run_id,
        )
        onset_info = run_info_for(
            verdict.onset_run_detector, verdict.onset_run_platform,
            stack_dir(verdict.onset_run_date), verdict.sample,
            verdict.onset_run_id,
        )
        facts = reproducer_facts_from(row, base_info, onset_info)
    except Exception as exc:  # noqa: BLE001 — optional metadata must fail soft
        _log.info("build_comments: no reproducer for %s %s (%s)",
                  plan.target, verdict.label, exc)
        return None
    if facts is None:
        _log.info(
            "build_comments: no reproducer for %s %s — run records were "
            "missing or incompatible", plan.target, verdict.label,
        )
    return facts


def _review(
    plan: CommentPlan,
    *,
    attributor: Attributor | None,
    patch_for: PatchFor | None,
    body_for: BodyFor | None,
    files_for: FilesFor | None = None,
) -> tuple[Attribution | None, AttributionRequest | None]:
    """One plan's cross-configuration review and the request it was made from.

    Every failure — no diff source, a raising fetch, a raising or declining
    model — leaves the attribution ``None``. With an attributor configured that
    means *no comment tonight* (see :func:`build_comments`): a blocked comment
    is recoverable on the next working night, while a degraded one posted now
    would be frozen in place by its own digest.

    The *request* comes back alongside it because it records which evidence this
    night could actually assemble — a diff that GitHub refused is a materially
    weaker review, and the digest has to be able to see that
    (:func:`_facts_payload`)."""
    if attributor is None:
        return None, None
    fetch = patch_for or (lambda _repo, _number: "")
    body_fetch = body_for or (lambda _repo, _number: "")
    try:
        request = _attribution_request(plan, fetch, body_fetch, files_for)
    except Exception as exc:  # noqa: BLE001 — a diff fetch must not lose the comment
        _log.warning(
            "build_comments: %s — could not assemble the review request (%s); "
            "no review, so no comment tonight", plan.target, exc,
        )
        return None, None
    try:
        return attributor.attribute(request), request
    except Exception as exc:  # noqa: BLE001 — an adapter that raises is a decline
        _log.warning(
            "build_comments: %s — the cross-configuration review failed (%s); "
            "no review, so no comment tonight", plan.target, exc,
        )
        return None, request


class HistoricalEvidenceUnavailable(RuntimeError):
    """A historical analogue the first pass read could not be read again.

    Raised out of :func:`_attribution_request` so it lands in the existing
    fail-closed path (:func:`_review`): with a reviewer configured, no comment is
    posted that night. That is the same rule an unusable review already follows,
    and for the same reason — the first pass's score was reached *with* this
    code in front of it, so a review conducted without it is not the second
    opinion the comment claims to rest on, and a comment posted on the weaker
    review would freeze itself in place by its own digest.
    """


def _historical(
    plan: CommentPlan, fetch: PatchFor, body_fetch: BodyFor
) -> tuple[HistoricalPR, ...]:
    """The plan's historical references, re-fetched into full analogues.

    The sidecar deliberately stores no patch and no body (see
    :class:`~k4bench.blame.models.HistoricalRef`), so the text is fetched again
    through the same authenticated boundary a competitor's diff comes through —
    one call per pull request, memoized for the whole run by the caller.

    A reference that comes back with **neither** a diff nor a description
    raises. That is stricter than the best-effort rule the competitors follow,
    and deliberately so: a competitor with no diff still appears with its paths
    and the first pass's reason, and losing it costs the review one alternative
    it can weigh. A missing analogue costs the review the evidence the first
    pass's own score was built on, and there is no honest way to render that as
    a weaker version of the same review. The cost of the strictness is a comment
    silenced on a night GitHub would not answer — recoverable tomorrow, unlike a
    comment posted on a materially different evidence set.

    Both halves are tested because the fetch seam reports failure as ``""`` and
    cannot say which kind it was. A binary-only or pure-rename pull request
    genuinely has no textual hunk, and the *first* pass accepted it on its paths
    and prose; failing the re-fetch on the empty patch alone would suppress that
    window's comment every night, forever, over a change GitHub is answering
    about perfectly well. A reference that yields nothing at all is the
    unreadable one. (A binary pull request with an empty description is still
    indistinguishable from a failure, and still fails closed — the rarer of two
    rare cases, and the safe side of it.)"""
    if len(plan.historical_refs) > MAX_COMMENT_ANALOGUES:
        # Checked before a single fetch: a night this wide must cost nothing,
        # not a hundred round trips inside one shared timeout before refusing.
        raise HistoricalEvidenceUnavailable(
            f"{len(plan.historical_refs)} historical analogues across this "
            f"window's rank groups, past the {MAX_COMMENT_ANALOGUES} one review "
            f"can carry; dropping some would leave the two passes weighing "
            f"different evidence"
        )
    analogues = []
    for ref in plan.historical_refs:
        patch = fetch(ref.repo, ref.pr)
        body = body_fetch(ref.repo, ref.pr)
        if not patch and not body:
            raise HistoricalEvidenceUnavailable(
                f"nothing readable for the historical analogue "
                f"{ref.repo}#{ref.pr}, which the first pass read before scoring "
                f"this window"
            )
        analogues.append(HistoricalPR(
            boundary_id=ref.boundary_id,
            base_release=ref.base_release,
            onset_release=ref.onset_release,
            package=ref.package,
            repo=ref.repo,
            number=ref.pr,
            title=ref.title,
            files=ref.files,
            additions=ref.additions,
            deletions=ref.deletions,
            body=body,
            patch=patch,
        ))
    return tuple(analogues)


def _attribution_request(
    plan: CommentPlan,
    fetch: PatchFor,
    body_fetch: BodyFor,
    files_fetch: FilesFor | None = None,
) -> AttributionRequest:
    """The whole window, as the reviewing model is shown it.

    The analogues are resolved *first*, before a single other fetch. They carry
    the only requirement here that can refuse the whole review, so a window that
    is going to be withheld should be withheld before it spends a round trip on
    a diff nobody will read.

    The subject's and the competitors' diffs are sampled with the compact
    directories of every row's run group first, then their geometry trees
    (:func:`~k4bench.blame.github.diff_sample`), when *files_fetch* can supply
    the hunks; otherwise *fetch*'s generic sample stands. The analogues keep the
    generic sample: they are the first pass's evidence, read as it read them."""
    historical = _historical(plan, fetch, body_fetch)
    own_dirs = tuple(sorted({compact_dir(p) for p in plan.geometry_paths} - {""}))
    trees = tuple(sorted({geometry_tree(p) for p in plan.geometry_paths} - {""}))
    hunks_read: dict[tuple[str, int], tuple[FilePatch, ...]] = {}

    def hunks(repo: str, number: int) -> tuple[FilePatch, ...]:
        if (repo, number) not in hunks_read:
            hunks_read[(repo, number)] = (
                tuple(files_fetch(repo, number) or ()) if files_fetch is not None else ()
            )
        return hunks_read[(repo, number)]

    def sampled(repo: str, number: int) -> str:
        files = hunks(repo, number)
        if not files:
            return fetch(repo, number)
        return diff_sample(files, own_dirs=own_dirs, trees=trees)

    def touches(candidate: CandidatePR):
        return detector_touches(
            candidate.files, hunks(candidate.repo, candidate.number), plan.geometry,
        )

    return AttributionRequest(
        repo=plan.repo,
        number=plan.number,
        title=plan.subject.title,
        base_release=plan.base_release,
        onset_release=plan.onset_release,
        suite=plan.suite,
        files=plan.subject.files,
        patch=sampled(plan.repo, plan.number),
        body=body_fetch(plan.repo, plan.number),
        additions=plan.subject.additions,
        deletions=plan.subject.deletions,
        regressions=tuple(_fact(row) for row in plan.rows),
        outcomes=plan.outcomes,
        competitors=tuple(
            replace(
                _competitor(other, scope, sampled, body_fetch), touches=touches(other),
            )
            # Cut the field to what the prompt can actually carry *before*
            # fetching anything: the prompt keeps the strongest
            # `MAX_COMPETITORS` in this same order, so a window with a hundred
            # candidates would otherwise spend a hundred GitHub round trips —
            # inside one shared timeout — to show thirty.
            for other, scope in _sorted_others(plan)[:MAX_COMPETITORS]
        ),
        packages_by_platform=plan.packages_by_platform,
        external_changes=tuple(sorted(plan.external_changes)),
        unchanged_by_platform=dict(plan.unchanged),
        packages_unavailable_on=plan.packages_unavailable_on,
        platform_switches=tuple(sorted(plan.platform_switches)),
        detector_switches=tuple(sorted(plan.detector_switches)),
        # Resolved above, and allowed to raise: the review must see the same
        # historical evidence the first pass did, or it must not happen at all.
        historical=historical,
        sweeps=plan.sweeps,
        touches=touches(plan.subject),
    )


def _fact(row: RegressionRow) -> RegressionFact:
    v = row.verdict
    return RegressionFact(
        id=row.fact_id,
        detector=v.detector, platform=v.platform, sample=v.sample,
        label=v.label, metric=v.metric, metric_family=v.metric_family,
        sub_detector=v.sub_detector, direction=str(getattr(v.direction, "value", v.direction)),
        pct_change=v.pct_change, value=v.value,
        baseline_median=v.baseline_median, z_score=v.z_score,
        scope_score=row.scope_score, scope_reason=row.scope_reason,
        scope_state=row.scope_state,
        # The release-boundary package counts come from the sidecar entry that
        # examined this row (:attr:`RegressionRow.boundary_changes`): this pass
        # has no provenance access of its own, and a boundary the ranker
        # measured as "no tracked package changed" is the sharpest evidence a
        # movement is the series' own noise. Boundaries the sidecar has nothing
        # for stay unread, which is what the prompt then says about them.
        history=history_from_verdict(v, packages_changed=row.boundary_changes),
        regions=v.region_deltas,
        components=v.component_deltas,
    )
