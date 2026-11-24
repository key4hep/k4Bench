"""The top of a comment: the alert stating what is attributed to the pull
request, the change window, and the model's assessment with its qualifiers.
"""

from __future__ import annotations

from collections import Counter

from k4bench.blame.attribute import Attribution
from k4bench.blame.comment.markdown import (
    _MAX_DESCRIPTION_CHARS,
    _MAX_SUMMARY_CHARS,
    REGRESSION,
    _count,
    _direction_detail,
    _one_line,
    _pct,
    _sentences,
)
from k4bench.blame.comment.plan import (
    CommentPlan,
    RegressionRow,
)
from k4bench.blame.comment.table import (
    _likelihood,
)
from k4bench.blame.models import (
    rank_group_key,
)
from k4bench.layout import SIM
from k4bench.regression.render import (
    SUITE_TITLES,
)


def _benchmarks(suite: str) -> str:
    """What a comment calls the benchmarks that measured its rows: the sim
    suite's as its comments always read, any other's by its name."""
    return "benchmarks" if suite == SIM else f"{SUITE_TITLES[suite].lower()} benchmarks"


def _alert(
    plan: CommentPlan,
    rows: list[RegressionRow],
    attribution: Attribution | None,
    *,
    min_score: float,
    n_regressions: int | None = None,
    n_scopes: int | None = None,
    scores: tuple[list[float], list[float]] | None = None,
    n_attributed_scopes: int | None = None,
    noun: str = REGRESSION,
    directions: Counter[str] | None = None,
) -> str:
    """The headline claim as a GitHub alert: the regressions the benchmarks
    confirmed in this pull request's change window that a model attributes to
    it at or above the threshold.

    *noun* is what the comment calls its steps (:func:`~.markdown._noun_for`).
    When it is not "regression", *directions* — the signs of the attributed
    steps — are spelled out beside the count, and an attribution with no
    increase among them is a note rather than a warning: nothing got slower or
    larger, and a warning box would say otherwise.

    It counts only those. The window's other regressions are not claims about
    this pull request — on a wide night they are hundreds of rows on detectors
    the change never reached — and leading with their total buries the one
    number the author is asking about. The association summary below counts
    them.

    The measurement and the estimate stay two statements in one sentence: what
    the nightly benchmarks confirmed, and which model attributes it here. The
    model is named by whoever produced the scores (:func:`_scored`): the
    cross-configuration reviewer, the first-pass ranker for rows the review
    did not answer, or both. When the comment stands only on first-pass scores
    while the review put everything it scored below the threshold, that
    disagreement is said in a second sentence — a reader must not take a
    ranker's claim for one the reviewer endorsed. The threshold is named rather
    than summarised as "certain": it is a configured number (``min_score``),
    the same one that decided this comment exists at all.

    With the lineage's identity union passed in (*scores*,
    *n_attributed_scopes*), the counts describe that union, with each
    reconfirmed identity at its newest published attribution. Should nothing
    reach the threshold, the alert states the window's total and the highest
    score instead."""
    cumulative = n_regressions is not None
    n_regressions = len(rows) if n_regressions is None else n_regressions
    n_scopes = len(plan.scopes) if n_scopes is None else n_scopes
    reviewed, carried = scores or _scored(rows, attribution)
    over_reviewed = [score for score in reviewed if score >= min_score]
    over_carried = [score for score in carried if score >= min_score]
    attributed = over_reviewed + over_carried
    where = (
        "in the reports covering this PR's change window" if cumulative
        else "in this PR's change window"
    )
    benchmarks = f"k4Bench's nightly {_benchmarks(plan.suite)}"
    if not attributed:
        measured = (
            f"{benchmarks} confirmed "
            f"{_regressions_text(n_regressions, n_scopes, noun=noun)} {where}."
        )
        everything = reviewed + carried
        if not everything:
            return f"> [!WARNING]\n> {measured}"
        return (
            f"> [!WARNING]\n> {measured} None of them is attributed to this PR "
            f"at {_pct(min_score)} or above (highest {_pct(max(everything))})."
        )
    if n_attributed_scopes is None:
        n_attributed_scopes = len({
            row.scope for row in rows
            if (likelihood := _likelihood(row, attribution)) is not None
            and likelihood >= min_score
        })
    if over_reviewed and over_carried:
        who = (
            f"the AI reviewer (or, for {noun}s it did not score, the "
            "first-pass AI ranker)"
        )
    elif over_reviewed:
        who = "the AI reviewer"
    elif reviewed:
        who = "the first-pass AI ranker"
    else:
        who = "the AI ranker"
    level = (
        f"at {_pct(attributed[0])}" if len(attributed) == 1
        else f"at {_pct(min_score)} or above (highest {_pct(max(attributed))})"
    )
    signed = (
        directions is not None and sum(directions.values()) == len(attributed)
    )
    detail = (
        _direction_detail(directions["UP"], directions["DOWN"], directions["NONE"])
        if signed and noun != REGRESSION else ""
    )
    what = _regressions_text(
        len(attributed), max(n_attributed_scopes, 1), noun=noun, detail=detail,
    )
    lines = [f"{benchmarks} confirmed {what} {where} that {who} attributes to this PR {level}."]
    if reviewed and not over_reviewed:
        lines.append(
            f"The AI reviewer scored {_count(len(reviewed), f'other {noun}')} "
            f"in this window and put none at {_pct(min_score)} or above "
            f"(highest {_pct(max(reviewed))})."
        )
    kind = "NOTE" if signed and not directions["UP"] else "WARNING"
    return f"> [!{kind}]\n> " + " ".join(lines)


def _regressions_text(
    n_regressions: int, n_scopes: int, *, noun: str = REGRESSION, detail: str = "",
) -> str:
    """``n`` regressions (or *noun*) and the detector/platform/sample scopes
    they span, with *detail* — their signs — in parentheses after the count.

    Rows count regressions; scopes count (detector, platform, sample) run
    groups. Both axes stay in the phrase: several metrics can regress in one
    scope, while the scope count states how broadly they reached. Not called
    "configuration": everywhere else that word means the sweep label, which is
    what the table's Config column holds."""
    counted = f"a {noun}" if n_regressions == 1 else _count(n_regressions, noun)
    if detail:
        counted += f" ({detail})"
    if n_regressions == 1:
        return counted
    if n_scopes == 1:
        return f"{counted} within one detector/platform/sample scope"
    return f"{counted} across {_count(n_scopes, 'detector/platform/sample scope')}"


def _scored(
    rows: list[RegressionRow], attribution: Attribution | None
) -> tuple[list[float], list[float]]:
    """``(what the review scored, what only the first pass scored)``.

    The table and the withdrawal gate read a row's *effective* likelihood
    (:func:`_likelihood`); the alert cannot, because a count is worthless without
    knowing whose judgement it holds. On a wide window the split is routine
    rather than rare — everything past
    :data:`~k4bench.blame.attribute._MAX_ATTRIBUTED_ROWS` is first-pass-only by
    construction.

    Rows with no likelihood at all (``not_candidate``, or unscored by both
    passes) appear in neither list — the alert claims nothing about them."""
    reviewed: list[float] = []
    carried: list[float] = []
    for row in rows:
        likelihood = _likelihood(row, attribution)
        if likelihood is None:
            continue
        if attribution is not None and row.fact_id in attribution.likelihoods:
            reviewed.append(likelihood)
        else:
            carried.append(likelihood)
    return reviewed, carried


def _window_line(plan: CommentPlan) -> str:
    """The change window as a single caption line — the Key4hep release dates
    that bound the step, shared by every row below. An open-ended window says so
    here, where the dates it is missing one of are."""
    if plan.base_release:
        window = f"`{plan.base_release}` → `{plan.onset_release}`"
    else:
        window = (
            f"≤ `{plan.onset_release}` — open-ended: no earlier settled "
            "measurement bounds it"
        )
    # The dates are Key4hep release dates, not calendar dates the benchmark ran:
    # said in the label so a reader does not read them as run days.
    return f"**Change window** (Key4hep releases): {window}"


def _assessment(
    plan: CommentPlan,
    rows: list[RegressionRow],
    attribution: Attribution | None,
    *,
    rendered: list[RegressionRow] | None = None,
    noun: str = REGRESSION,
) -> str | None:
    """The model's reasoning as a labelled blockquote — the label is where the
    comment openly says an AI made this call. *noun* is what the comment calls
    its steps.

    With a cross-configuration review, that is its summary: it saw every
    configuration that moved and every one that did not, so it is the account
    that can actually explain the pattern. It does not necessarily account for
    every row in the table, though — a very wide window offers the review only
    its largest movements, and a reply may answer only some of what it was
    offered — so a partial review says which rows it covered. A narrative reading
    "this PR does not fit the affected set" printed above rows still carrying an
    unrelated 91% must not look like it was talking about them.

    With no reviewer configured at all — the only way a comment is rendered
    without a review (:func:`build_comments`) — the comment quotes the
    per-configuration ranker's one-liner for its strongest row, and then it
    claims "the most likely cause" only when this PR outranks every other
    candidate — a comment can fire on any score above ``min_score``, and a PR the
    ranker placed second must not be told it came first. Nothing is rendered when
    neither model explained itself: an unexplained score is not comment-worthy
    prose, and it already stands in the table."""
    if attribution is not None:
        text = _sentences(attribution.summary, _MAX_SUMMARY_CHARS)
        if text:
            return (
                f"\n> 🤖 **The AI reviewer's assessment:** {text}"
                + _evidence_note(attribution)
                + _coverage_note(
                    rendered if rendered is not None else rows, attribution, noun=noun,
                )
            )
        return None
    lead = rows[0] if rows else None
    text = _one_line(lead.scope_reason, _MAX_DESCRIPTION_CHARS) if lead else ""
    if not text or lead.scope_score is None:
        return None
    # Only judged candidates can be outranked: an unscored one is not behind
    # this pull request, it is simply unknown, so it cannot support a claim to
    # first place either. Outranked within the lead row's own ranking call —
    # the only field its score was compared against.
    standing = plan.standings.get(rank_group_key(lead.verdict))
    outranks_all = standing is not None and not standing.unranked_rivals and (
        standing.rival is None or standing.rival.score < lead.scope_score
    )
    claim = "the most likely" if outranks_all else "a likely"
    return (
        f"\n> 🤖 **The AI ranker judged this PR {claim} cause of the "
        f"{noun}:** {text}"
    )


def _evidence_note(attribution: Attribution) -> str:
    """One line when the review could not judge whether the movements are real.

    ``likely_noise`` never reaches here — it withdraws the comment outright
    (:func:`build_comments`) — so the only case left is
    ``insufficient_evidence``: the benchmark history behind these steps is too
    short for the review to corroborate them. That does *not* overturn the
    detector's own confirmation, which is a two-strike statistical judgement over
    the measurements themselves and is why the comment exists at all. It does
    change how much weight a reader should give the paragraph above it, so it is
    said out loud rather than left for nobody to know.

    ``real_change`` prints nothing. A caveat that appears on every comment is one
    every reader learns to skip, which is how the one that matters gets skipped
    too."""
    assessment = attribution.assessment
    if assessment is None or assessment.verdict != "insufficient_evidence":
        return ""
    return (
        "\n>\n> <sub>The benchmark history behind these steps was too short for "
        "the review to judge whether they are a real change; the steps "
        "themselves are confirmed by the nightly detector.</sub>"
    )


def _coverage_note(
    rows: list[RegressionRow], attribution: Attribution, *, noun: str = REGRESSION,
) -> str:
    """What the assessment above does *not* cover, when it covers less than all.

    Measured over the *current* rows this comment renders, not over the whole
    window — and the wording says "current", because a table carrying rows
    retained from earlier versions shows more rows than this count covers and a
    bare "of the four shown" would not add up for the reader counting them. A
    retained row is outside the count for the same reason it is outside the
    review: it is not part of tonight's evidence. A row goes unreviewed either
    because the window carried more
    regressions than the prompt offers or because the reply simply skipped it,
    and it then keeps whatever the first pass left it with — a score, or one of
    the states that has none (:func:`_likelihood`). That only misleads a reader
    who can see the row: a caveat about rows nobody renders warns of a
    discrepancy nothing on the page shows, and on a wide window a single dropped
    row would print one every night. The wording avoids promising a score for
    those rows, since plenty of them have none, and stays agreement-free so one
    unreviewed row reads as well as twenty."""
    unreviewed = sum(1 for row in rows if row.fact_id not in attribution.likelihoods)
    if not unreviewed:
        return ""
    return (
        f"\n>\n> <sub>This assessment covers "
        f"{_count(len(rows) - unreviewed, noun)} of the {len(rows)} "
        f"current {noun}s shown; anything it did not answer keeps its "
        "first-pass state, and its score where there is one.</sub>"
    )
