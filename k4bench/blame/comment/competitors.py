"""The field a comment's claim is weighed against: the other candidate pull
requests in the window, the note when the ranking is close, and the package
diff to inspect them in.
"""

from __future__ import annotations

import math

from k4bench.blame.comment.markdown import (
    _cell,
    _count,
    _one_line,
    _pct,
)
from k4bench.blame.comment.plan import (
    CommentPlan,
    ScopeStanding,
)
from k4bench.blame.comment.selection import (
    _sorted_others,
)
from k4bench.blame.comment.state import (
    _verdict_identity,
)
from k4bench.blame.comment.table import (
    _SHOW_PLATFORM_COLUMN,
    _UNSCORED,
)
from k4bench.blame.models import (
    CandidatePR,
)
from k4bench.labels import compact_sample, pretty_platform
from k4bench.regression.render import (
    stack_changes_href,
)

#: Competing candidates listed in the collapsed field; the rest are counted.
#: Bounded for the same reason as the regression table
#: (:data:`~k4bench.blame.comment.table._TARGET_TABLE_ROWS`).
_MAX_OTHER_CANDIDATES = 5


#: Likelihood points between this PR and the closest other candidate at or
#: under which the ranking is called a weak preference in words. Wide enough to
#: catch a genuinely crowded field, narrow enough that an ordinary night — where
#: the ranker picked one PR out of the pack — says nothing extra.
_CROWDED_SPREAD = 10.0


def _others_section(plan: CommentPlan, dashboard_url: str | None = None) -> str:
    """The rest of the candidates scored across this window, with their
    likelihoods — the reader needs to see what else was in the frame to weigh
    the claim against this PR, including the case where nothing else was.

    Collapsed by default, but the summary carries the strongest competing score
    without being opened: how far ahead this PR sits — the difference between a
    ranking that picked it and one that barely preferred it — belongs in front of
    a reader who expands nothing. The table is capped at
    :data:`_MAX_OTHER_CANDIDATES`, with any surplus counted rather than pasted.

    The candidates are named, never linked (:func:`_pr_ref`), so the section
    closes with the window's package diff — where the complete field, and every
    package behind it, can be read without notifying anyone. It is offered, not
    claimed as their provenance: see :func:`_package_diff_line`."""
    others = _sorted_others(plan)
    diff = _package_diff_line(plan, dashboard_url)
    if not others:
        return "\n".join([
            "",
            "> [!NOTE]",
            "> This was the only pull request found across every tracked "
            "package that changed in this window."
            + (f" {diff}" if diff else ""),
        ])

    shown = others[:_MAX_OTHER_CANDIDATES]
    strongest = _closest_candidate(plan)
    headline = (
        f"highest {_pct(strongest.score)}"
        + _scored_on(plan, strongest, prefix=" on ")
        if strongest is not None
        else "none of them scored by the ranker"
    )
    lines = [
        "",
        "<details>",
        "<summary><b>📋 Other pull requests in this window</b> — "
        f"{_count(len(others), 'candidate')}, {headline}"
        "</summary>",
        "",
        "| Pull request | Highest likelihood | Scored on |",
        "|:---|---:|:---|",
        *(
            f"| {_pr_ref(c)} — {_cell(_one_line(c.title, 80))} | "
            f"{_candidate_score_cell(c)} | {_scored_on(plan, c) if c.ranked else ''} |"
            for c, _scope in shown
        ),
    ]
    if len(others) > len(shown):
        lines += ["", f"_…and {_count(len(others) - len(shown), 'more candidate')}._"]
    if diff:
        lines += ["", diff]
    lines += ["", "</details>"]
    return "\n".join(lines)


def _scored_on(plan: CommentPlan, candidate: CandidatePR, *, prefix: str = "") -> str:
    """Where a competitor's listed score was given — detector and sample, plus
    the platform when the window spans several — or nothing when unknown. A
    competitor is scored once per rank group and listed at its strongest, so
    the score only means something next to the scope that produced it."""
    scope = plan.others_scope.get((candidate.repo.lower(), candidate.number))
    if scope is None:
        return ""
    detector, platform, sample = scope
    text = f"{_cell(detector)} · {_cell(compact_sample(sample))}"
    if _SHOW_PLATFORM_COLUMN or len(plan.platforms) > 1:
        text += f" · {_cell(pretty_platform(platform))}"
    return prefix + text


def _package_diff_line(plan: CommentPlan, dashboard_url: str | None) -> str | None:
    """The window's package diff, one link per build platform that contributed
    a row — offered as somewhere to look, never as the candidates' provenance.

    The distinction is load-bearing. :func:`_record_others` folds in every
    entry's candidates with no window filter at all, while only an entry
    measuring *exactly* this comment's window describes this window's release
    diff (:func:`_confirmed_rows`): a row can enter on a narrower range of its
    own, and :attr:`CommentPlan.packages_unavailable_on` exists to name the
    platforms whose diff for this window was never read. So a candidate here can
    have been found in a range whose package set is not the one behind this
    link, and "every candidate came from" would assert a provenance the pipeline
    does not establish.

    Per platform because provenance is recorded that way: a plan is keyed by
    pull request and window and never by platform, so its rows can span several,
    while the release diff behind them is kept separately for each
    (:class:`CommentPlan`). Two platforms' diffs are two measurements.

    Within a platform the scope parameters only choose which regression ledger
    the Stack Changes view draws underneath the diff, so the lowest row identity
    is as good a landing place as any — picked by identity rather than by list
    order so the sentence is byte-stable across runs."""
    platforms = sorted({row.verdict.platform for row in plan.rows})
    links = []
    for platform in platforms:
        if any(
            row.verdict.base_platform != row.verdict.onset_run_platform
            for row in plan.rows if row.verdict.platform == platform
        ):
            # Stack Changes compares releases of one platform; migration
            # rows already link to their window in the Regressions view.
            continue
        lead = min(
            (row.verdict for row in plan.rows if row.verdict.platform == platform),
            key=_verdict_identity,
        )
        href = stack_changes_href(
            dashboard_url,
            detector=lead.detector,
            platform=platform,
            sample=lead.sample,
            base_release=plan.base_release,
            onset_release=plan.onset_release,
        )
        if not href:
            continue
        label = (
            "package diff for this window" if len(platforms) == 1
            else _cell(pretty_platform(platform))
        )
        links.append(f"[{label} ↗]({href})")
    if not links:
        return None
    if len(links) == 1 and len(platforms) == 1:
        return f"Inspect the {links[0]}."
    return (
        "Inspect this window's package diffs, one per platform: "
        f"{', '.join(links)}."
    )


def _pr_ref(candidate: CandidatePR) -> str:
    """A competing candidate named as ``owner/repo#123``, inert on purpose.

    These pull requests are *not* the ones being commented on — they are the
    field the ranking was made against — and GitHub turns any reference to them,
    a bare ``owner/repo#123`` or a link carrying their URL, into a cross-
    reference on their own timeline, notifying everyone subscribed there. A PR
    that was merely a candidate should not collect a notification every time
    another window implicates someone else, so the number is broken with a
    zero-width space: unchanged to a reader, unparsed by GitHub, and
    unclickable. Whoever wants the full field has the package-diff link this
    section closes with (:func:`_package_diff_line`)."""
    zwsp = "​"  # U+200B zero-width space
    return _cell(f"{candidate.repo}#{zwsp}{candidate.number}")


def _closest_candidate(plan: CommentPlan) -> CandidatePR | None:
    """The strongest competing candidate the first pass scored anywhere in the
    window — what the competing-field summary quotes, beside the scope it was
    scored on.

    :func:`_sorted_others` is judged-first, so the leading entry is that
    candidate when there is one. A field nobody scored has no closest
    candidate: an unscored rival is an unknown, not a near miss, and the
    summary may not quote a percentage no model produced."""
    others = _sorted_others(plan)
    top = others[0][0] if others else None
    return top if top is not None and top.ranked else None


def _leading_standing(plan: CommentPlan) -> ScopeStanding | None:
    """The rank group carrying this pull request's strongest first-pass score —
    the claim the comment leads with — and, among equally strong groups, the one
    where a rival came closest."""
    if not plan.standings:
        return None
    return max(
        plan.standings.values(),
        key=lambda standing: (
            standing.score,
            -(standing.margin if standing.margin is not None else math.inf),
        ),
    )


def _crowded_note(plan: CommentPlan) -> str | None:
    """Said out loud when the ranking does not clearly favour this PR — in
    words, rather than leaving the reader to subtract two numbers. Nothing is
    said when no competing candidate was scored in that ranking call: an
    unscored rival is not a near miss, it is an unknown, and no gap can be
    computed from it.

    Which way the preference runs is the whole point, so the note is
    direction-aware. A PR the ranker placed *behind* another candidate is told so
    however wide the gap is: that is the single most important qualifier on a
    comment accusing it. A PR that is ahead hears about it only when the lead is
    thin (``_CROWDED_SPREAD``) — a caveat printed on every comfortable night is
    wallpaper, and the score and the summary line already say what a comfortable
    lead looks like.

    It renders directly under the assessment, where the claim it qualifies is
    made — not down beside the competing field, where a reader has already
    finished reading the accusation.

    Both scores come from one ranking call: the rank group behind this pull
    request's strongest score (:func:`_leading_standing`) and the strongest
    other candidate *that* call scored. A competitor's score from another
    detector answers a different question — which change explains that
    detector's regressions — and says nothing about the preference here."""
    standing = _leading_standing(plan)
    if standing is None or standing.rival is None:
        return None
    mine_raw, closest = standing.score, standing.rival
    delta = mine_raw - closest.score
    if not math.isfinite(delta) or delta > _CROWDED_SPREAD:
        return None
    # Prose has to agree with the percentages sitting right above it, not with
    # the raw scores behind them: :func:`_pct` rounds each score independently,
    # so a sub-point raw gap can render as a one-point *displayed* gap and vice
    # versa. Rounding both scores the same way :func:`_pct` does, then
    # differencing, keeps the two in lockstep.
    mine = int(round(mine_raw))
    theirs = int(round(closest.score))
    display_delta = mine - theirs
    points = abs(display_delta)
    if display_delta < 0:
        return (
            f"\n_The closest other candidate scored {_count(points, 'point')} "
            "**higher** than this PR — the ranker's preference in this window "
            "runs against it, not for it._"
        )
    if points == 0:
        separation = "Nothing separates this PR from the closest other candidate"
    else:
        verb = "separates" if points == 1 else "separate"
        separation = (
            f"Only {_count(points, 'point')} {verb} this PR from the closest "
            "other candidate"
        )
    return (
        f"\n_{separation} — the ranker is expressing a weak preference here, "
        "not a clear pick._"
    )


def _candidate_score_cell(candidate: CandidatePR) -> str:
    """A competing candidate's likelihood, or that nobody gave it one."""
    return _pct(candidate.score) if candidate.ranked else _UNSCORED
