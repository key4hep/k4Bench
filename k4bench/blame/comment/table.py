"""The regression table of a comment: which rows it shows, how each row reads,
and the line under it that counts what each report carried.

The table lists the regressions attributed to the pull request at or above the
configured threshold, most likely first (:func:`_selected_rows`); the rest of the
window is counted by the association summary (:mod:`.lineage`) and linked
through the dashboard.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from k4bench.blame.attribute import Attribution
from k4bench.blame.comment.digest import (
    Reproducers,
)
from k4bench.blame.comment.markdown import (
    REGRESSION,
    _capitalized,
    _cell,
    _count,
    _pct,
)
from k4bench.blame.comment.plan import (
    CommentPlan,
    RegressionRow,
    _row_identity,
)
from k4bench.blame.comment.selection import (
    _DEFAULT_MIN_SCORE,
)
from k4bench.blame.comment.state import (
    CommentObservation,
    RetainedRow,
    _report_counts,
    _retained_identity,
    _retained_sort_key,
    _retained_window,
)
from k4bench.blame.prompt import movement
from k4bench.labels import compact_sample, pretty_platform
from k4bench.regression.render import (
    nightly_report_href,
    regression_href,
    window_href,
)

#: Regression rows shown in the table. Every other row is still scored and
#: linked through the dashboard. A sweep can confirm three hundred
#: near-identical rows and a comment over GitHub's 65,536-character limit is
#: rejected outright, so this hard bound is part of the write contract rather
#: than only a presentation preference.
_TARGET_TABLE_ROWS = 5


_UNKNOWN_ONSET = "unknown"


#: Whether the regression table carries a **Platform** column.
#:
#: Off: the suite builds on one platform today, so the column would repeat one
#: slug down every row. Purely a rendering choice — platform remains a first-class
#: scope dimension everywhere else (row identity, grouping, outcomes, dashboard
#: links, the facts digest, package provenance, both prompts), and the renderer
#: below is written to show the column the moment this is flipped. It is a
#: constant rather than a count of the platforms actually present, so the table's
#: shape is a decision someone made, not an accident of one night's data.
_SHOW_PLATFORM_COLUMN = False


def _likelihood(row: RegressionRow, attribution: Attribution | None) -> float | None:
    """What this row is shown as, and ordered by — or ``None`` when nothing has
    been claimed about it.

    The review's score when it gave one; otherwise the per-configuration
    ranker's. A row the review omitted is not a zero — an unanswered row keeps
    the judgement that was already made about it — and a row neither pass judged
    has no likelihood at all. Rendering that as 0% would read as "the models
    cleared this one", which nobody said.

    A row the pull request is *not a candidate* for never carries a likelihood,
    not even one the review offered. That state is a deterministic fact — the
    candidate search in that scope was complete and this change is not in the
    commit range behind the regression — and it outranks a model's opinion about
    it. Letting a stray high score on such a row into the table would put a
    percentage next to a regression the pipeline knows this change cannot have
    shipped in, and (through the withdrawal gate below) would let it hold up a
    comment on its own.

    A row a joint review was offered and left unanswered
    (:attr:`~k4bench.blame.attribute.Attribution.withheld`) carries none either:
    the other pull requests in the window were judged on it jointly, and the
    first pass's score here would set a second reading of the same movement
    beside theirs."""
    if row.scope_state == "not_candidate":
        return None
    if attribution is None:
        return row.scope_score
    if row.fact_id in attribution.withheld:
        return None
    return attribution.likelihoods.get(row.fact_id, row.scope_score)


def _score_source(
    row: RegressionRow, attribution: Attribution | None
) -> tuple[float | None, str]:
    likelihood = _likelihood(row, attribution)
    if likelihood is None or not (
        math.isfinite(likelihood) and 0 <= likelihood <= 100
    ):
        return None, ""
    reviewed = attribution is not None and row.fact_id in attribution.likelihoods
    return likelihood, "reviewer" if reviewed else "ranker"


def _row_sort_key(row: RegressionRow, attribution: Attribution | None) -> tuple:
    """Most likely first, then the full detector before its removal
    configurations, then the largest movement, then identity — so the table is
    stable across nights and a re-render triggers no edit. Rows nobody scored
    sort last: they are evidence about the window, not claims about this pull
    request, and they must not head a table that reads top-down.

    The baseline leads its ties because a review scores a scope as a whole, so
    its rows share one likelihood, and the full detector is the configuration a
    reader measures the claim against — a removal configuration's larger
    percentage is often only a smaller denominator."""
    likelihood = _likelihood(row, attribution)
    return (
        likelihood is None,
        -(likelihood if likelihood is not None else 0.0),
        row.verdict.label != "baseline",
        -movement(row.verdict.pct_change),
        *_row_identity(row),
    )


@dataclass(frozen=True)
class _PoolRow:
    """One candidate table row — tonight's evidence or a retained snapshot —
    with the sort key both kinds share, so a single ranked pool orders them."""

    key: tuple
    current: RegressionRow | None = None
    past: RetainedRow | None = None


@dataclass(frozen=True)
class _Selection:
    """The rows a table shows, and the pool they were drawn from.

    ``attributed`` says whether the pool is the rows attributed to this pull
    request at or above the threshold (the ordinary case) or, when none is,
    every row of the window; ``pool`` is that pool's size."""

    rows: list[_PoolRow]
    attributed: bool
    pool: int


def _selected_rows(
    current: list[RegressionRow],
    historical: list[RetainedRow],
    attribution: Attribution | None,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> _Selection:
    """The rows the table shows, in ranked order.

    The pool is every currently confirmed row plus every retained historical
    row whose likelihood reaches *min_score* — the regressions this comment
    attributes to the pull request. A row scored below it, unscored, or one the
    pull request is not a candidate for is not shown: next to the attributed
    rows it reads as part of the claim, and the association summary counts it
    instead. Only when nothing reaches the threshold does the pool fall back to
    every row, so the table is never a header alone.

    The pool is ranked by the shared likelihood/movement/identity key.
    Selection then reserves, in order: the pool's **globally strongest row** —
    the alert quotes the strongest evidence, so the table must never hide it,
    however old its onset; the strongest *current* row, for the same reason
    when a historical row leads; one representative per current onset — an
    undated group before the dated ones, newest dated first — so a newer step
    cannot vanish behind five rows of an older one; and finally the strongest
    remaining rows, all inside the hard cap."""
    everything = [
        _PoolRow(key=_row_sort_key(row, attribution), current=row)
        for row in current
    ] + [
        _PoolRow(key=_retained_sort_key(row), past=row) for row in historical
    ]
    pool = [
        entry for entry in everything
        if (likelihood := _pool_likelihood(entry, attribution)) is not None
        and likelihood >= min_score
    ]
    attributed = bool(pool)
    if not attributed:
        pool = everything
    pool.sort(key=lambda entry: entry.key)
    current_pool = [entry for entry in pool if entry.current is not None]

    chosen: dict[tuple, _PoolRow] = {}

    def _reserve(entry: _PoolRow) -> None:
        if len(chosen) < _TARGET_TABLE_ROWS:
            chosen.setdefault(entry.key, entry)

    if pool:
        _reserve(pool[0])
    if current_pool:
        _reserve(current_pool[0])
    strongest_per_onset: dict[str, _PoolRow] = {}
    for entry in current_pool:
        strongest_per_onset.setdefault(_onset_label(entry.current), entry)
    covered = {
        _onset_label(entry.current)
        for entry in chosen.values()
        if entry.current is not None
    }
    dated = sorted(set(strongest_per_onset) - {_UNKNOWN_ONSET}, reverse=True)
    undated = [_UNKNOWN_ONSET] if _UNKNOWN_ONSET in strongest_per_onset else []
    for onset in undated + dated:
        if onset not in covered:
            _reserve(strongest_per_onset[onset])
    for entry in pool:
        _reserve(entry)
    shown = sorted(chosen.values(), key=lambda entry: entry.key)
    return _Selection(rows=shown, attributed=attributed, pool=len(pool))


def _pool_likelihood(entry: _PoolRow, attribution: Attribution | None) -> float | None:
    """The likelihood a pool row is shown with: a current row's effective one
    (:func:`_likelihood`), a retained row's last published one."""
    if entry.current is not None:
        return _likelihood(entry.current, attribution)
    if entry.past.state == "not_candidate":
        return None
    return entry.past.likelihood


def _onset_label(row: RegressionRow) -> str:
    """The one grouping key used for a row's step onset — what the table
    reserves a place for, so every step inside a containing window is
    represented."""
    return row.verdict.onset_run_date or _UNKNOWN_ONSET


def _row_window(row: RegressionRow) -> tuple[str, str]:
    """A row's *own* tightest change window, which can be narrower than the
    comment's: a metric that settled later carries a later last-accepted
    release, and the step is only bounded by the pair the metric itself
    measured."""
    v = row.verdict
    return (v.last_accepted_run_date or "", v.onset_run_date or "")


def _window_cell(base: str, onset: str) -> str:
    """One row's change window as the table shows it. An unbounded base is
    written as an upper bound rather than invented, matching
    :func:`_window_line`."""
    onset_text = f"`{_cell(onset or _UNKNOWN_ONSET)}`"
    return f"`{_cell(base)}` → {onset_text}" if base else f"≤ {onset_text}"


def _row_links(
    plan: CommentPlan, rows: list[RegressionRow], dashboard_url: str | None
) -> dict[str, str]:
    """``{fact id: href}`` for every row the table will render.

    Each row goes to its *own* regression pinned in the dashboard's Stack
    Changes view (:func:`~k4bench.regression.render.regression_href`), where the
    metric's trend, its onset and the window's package diff sit in one place —
    the reader's question is "did my change do this?", and that view is the one
    that answers it without a second click. A row whose verdict cannot be pinned
    (no onset identity) falls back to its configuration's Regressions view, which
    at least lands on the right window.

    The hrefs are ~400 characters each and a night can carry hundreds of rows;
    writing one inline per row is what pushes a wide night past GitHub's
    65,536-character comment limit, where a comment is rejected outright rather
    than truncated. Markdown *reference* links move each URL into a definition at
    the bottom of the body, and only rendered rows get one. The labels are the
    rows' own fact ids — already assigned in identity order — so a body is stable
    across nights and a re-render triggers no edit."""
    if not dashboard_url:
        return {}
    links = {}
    for row in rows:
        href = regression_href(
            dashboard_url,
            verdict=row.verdict,
            base_release=plan.base_release, onset_release=plan.onset_release,
        ) or window_href(
            dashboard_url,
            detector=row.verdict.detector, platform=row.verdict.platform,
            sample=row.verdict.sample,
            base_release=plan.base_release, onset_release=plan.onset_release,
            stack=row.stack,
        )
        if href:
            links[row.fact_id] = href
    return links


def _link_definitions(links: dict[str, str]) -> str | None:
    """The reference-link definitions :func:`_row_links` promised.

    Markdown renders these as nothing at all, so they sit at the end of the body
    where they interrupt no one."""
    if not links:
        return None
    return "\n" + "\n".join(
        f"[{label}]: {href}" for label, href in sorted(links.items())
    )


def _table_head(*, show_window: bool, show_reproduce: bool) -> list[str]:
    """The table header, including the optional platform, change-window and
    reproduce columns."""
    header = ["Metric", "Detector"]
    align = [":---", ":---"]
    if _SHOW_PLATFORM_COLUMN:
        header.append("Platform")
        align.append(":---")
    header += ["Sample", "Config"]
    align += [":---", ":---"]
    if show_window:
        header.append("Change window")
        align.append(":---")
    header.append("Change")
    align.append("---:")
    header.append("Attribution")
    align.append("---:")
    if show_reproduce:
        header.append("Reproduce")
        align.append(":---")
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(align) + "|",
    ]


def _row_line(
    row: RegressionRow,
    attribution: Attribution | None,
    links: dict[str, str],
    *,
    show_window: bool,
    recipe: str,
) -> str:
    """One regression as a table row — the same cells whichever ordering placed
    it, so both tables read identically row for row.

    The dashboard link hangs off the **metric** cell because that is what it
    opens, and only when :func:`_row_links` emitted a definition for this row.
    Metric and configuration keep their raw names: those are what the dashboard
    labels the series with."""
    v = row.verdict
    metric = (
        f"`{_cell(v.metric)}`"
        + (f" · {_cell(v.sub_detector)}" if v.sub_detector else "")
    )
    cells = [
        f"[{metric}][{row.fact_id}]" if row.fact_id in links else metric,
        _cell(v.detector),
    ]
    if _SHOW_PLATFORM_COLUMN:
        cells.append(_cell(pretty_platform(v.platform)))
    cells += [
        _cell(compact_sample(v.sample)),
        f"`{_cell(v.label)}`",
    ]
    if show_window:
        cells.append(_window_cell(*_row_window(row)))
    cells.append(_change_cell(v.pct_change))
    cells.append(_likelihood_cell(row, attribution))
    if recipe is not None:
        cells.append(_recipe_cell(recipe))
    return "| " + " | ".join(cells) + " |"


def _past_row_line(
    row: RetainedRow,
    past_links: dict[tuple, tuple[str, str]],
    *,
    show_window: bool,
    recipe: str | None,
) -> str:
    """One retained row with its last published movement and likelihood."""
    metric = (
        f"`{_cell(row.metric)}`"
        + (f" · {_cell(row.sub_detector)}" if row.sub_detector else "")
    )
    reference = past_links.get(_retained_identity(row))
    cells = [
        f"[{metric}][{reference[0]}]" if reference else metric,
        _cell(row.detector),
    ]
    if _SHOW_PLATFORM_COLUMN:
        cells.append(_cell(pretty_platform(row.platform)))
    cells += [
        _cell(compact_sample(row.sample)),
        f"`{_cell(row.label)}`",
    ]
    if show_window:
        cells.append(_window_cell(*_retained_window(row)))
    cells += [
        _change_cell(row.pct),
        _past_likelihood_cell(row),
    ]
    if recipe is not None:
        # The recipe published on the night this row was last confirmed. It is
        # carried in the snapshot rather than rebuilt, because only the night
        # that published it can vouch for the file being there.
        cells.append(_recipe_cell(recipe))
    return "| " + " | ".join(cells) + " |"


def _recipe_cell(url: str) -> str:
    """The link to this row's own published recipe.

    Empty when nothing was published for the row, rather than a link to
    commands for a different configuration than the one whose line the reader
    followed."""
    return f"[🔁 recipe ↗]({_cell(url)})" if url else ""


def _past_likelihood_cell(row: RetainedRow) -> str:
    """The likelihood recorded on the row's last published version — the same
    three shapes a current cell has, so an unscored or not-a-candidate snapshot
    can never resurface wearing a number. A score that drifted on a night the
    digest held unchanged was never written, so this is the last *published*
    number rather than the last one the ranker produced."""
    if row.state == "not_candidate":
        return _NOT_A_CANDIDATE
    if row.likelihood is None:
        return _UNSCORED
    return _pct(row.likelihood)


def _past_links(
    rows: list[RetainedRow], dashboard_url: str | None
) -> dict[tuple, tuple[str, str]]:
    """``{identity: (label, href)}`` for the retained rows the table renders.

    Reconstructed from the snapshot's validated structured fields through the
    shared link helper — never from a stored URL. The link pins the archived
    Regressions view for the row's own window, stack and last-reported night,
    which is where a regression that tonight's report no longer confirms can
    still be read. Both run ids are passed: they are what qualifies a
    same-release window, whose releases alone name several different windows
    and would land the reader on whichever the view ordered first
    (:func:`~k4bench.regression.render.window_token`). Labels are ``h1``,
    ``h2``, … in identity order, disjoint from the current rows' fact ids."""
    if not dashboard_url:
        return {}
    links: dict[tuple, tuple[str, str]] = {}
    ordered = sorted(rows, key=_retained_identity)
    for index, row in enumerate(ordered, start=1):
        href = window_href(
            dashboard_url,
            detector=row.detector,
            platform=row.platform,
            sample=row.sample,
            base_release=row.base_release,
            onset_release=row.onset_release,
            base_run=row.base_run or None,
            onset_run=row.onset_run or None,
            stack=row.stack or None,
            report_night=row.last_reported,
        )
        if href:
            links[_retained_identity(row)] = (f"h{index}", href)
    return links


def _table(
    plan: CommentPlan,
    rows: list[RegressionRow],
    attribution: Attribution | None,
    *,
    selection: _Selection,
    links: dict[str, str],
    past_links: dict[tuple, tuple[str, str]],
    dashboard_url: str | None,
    reproduce: Reproducers | None = None,
    observations: Sequence[CommentObservation] = (),
    min_score: float = _DEFAULT_MIN_SCORE,
    noun: str = REGRESSION,
) -> str:
    """The regressions attributed to this pull request, most likely first —
    tonight's confirmed ones and any retained rows that outrank them.

    Each row carries its **own tightest change window** rather than the
    comment's: a metric that settled later entered on a narrower range, and
    pinning the step to the pair that metric actually measured is what makes a
    row checkable.

    One table rather than one section per configuration: which configurations
    moved — and, read against the review's summary, which did not — is the
    substance of the claim, and a reader weighing it needs to see the pattern at
    once. Selection reserves the strongest evidence and one row per represented
    onset (:func:`_selected_rows`); a night wider than the cap says
    how many more there are in one line rather than pasting or folding them: a detector-removal sweep
    can confirm three hundred near-identical rows, which no one reads and GitHub
    will not accept — and a collapsed ``<details>`` block would not help, since
    collapsed Markdown still counts against the body limit. Every shown row
    links into the dashboard, which is where the complete set — and any
    re-sorting — lives.

    Only rows at or above the threshold are drawn (:func:`_selected_rows`): the
    window's other regressions are not claims about this pull request, and the
    association summary counts them. That can leave the window's largest
    movement off the table; the line below counts what each report carried and
    links all of it (:func:`_reports_line`).

    A retained historical row keeps the movement, likelihood and recipe last
    published for it; the cumulative alert uses the same newest-published
    rule."""
    shown = selection.rows
    urls = reproduce.urls if reproduce is not None else {}
    recipes = {
        id(entry): (
            urls.get(_row_identity(entry.current), "")
            if entry.current is not None
            else entry.past.recipe
        )
        for entry in shown
    }
    # The column exists only when at least one shown row has somewhere to send
    # the reader; an all-empty column is a header and nothing else.
    show_recipe = any(recipes.values())
    # Shown whenever a *rendered* row's own window is not simply the comment's
    # — a narrower base, a newer onset, or a retained row carrying a window of
    # its own. Decided over the rows that render rather than the whole plan: a
    # narrower window carried only by a row the cap cut is not on the page, and
    # a column repeating the header's window once per line is a column of
    # nothing.
    show_window = {
        _row_window(entry.current) if entry.current is not None
        else _retained_window(entry.past)
        for entry in shown
    } != {(plan.base_release or "", plan.onset_release)}
    lines = [
        "",
        # A bold caption, not a Markdown heading: it reads at the same size as the
        # window line above it (:func:`_window_line`), so the two captions the
        # reader scans first sit at one level rather than the table's shouting over
        # the window that scopes it.
        (
            f"📊 **{_capitalized(noun)}s attributed to this PR at "
            f"{_pct(min_score)} or above**"
            if selection.attributed else
            f"📊 **{_capitalized(noun)}s in this window, ranked by AI-based "
            "attribution likelihood**"
        ),
        "",
        *_table_head(show_window=show_window, show_reproduce=show_recipe),
    ]
    for entry in shown:
        recipe = recipes[id(entry)] if show_recipe else None
        if entry.current is not None:
            lines.append(_row_line(
                entry.current, attribution, links,
                show_window=show_window, recipe=recipe,
            ))
        else:
            lines.append(_past_row_line(
                entry.past, past_links, show_window=show_window, recipe=recipe,
            ))
    shown_current = sum(1 for entry in shown if entry.current is not None)
    reports = _reports_line(
        len(rows), len(shown), shown_current, observations,
        dashboard_url=dashboard_url,
        attributed=selection.pool if selection.attributed else None,
        noun=noun,
        min_score=min_score,
    )
    if reports:
        lines += ["", reports]
    return "\n".join(lines)


#: How many reports the line under the table names before it starts counting
#: them instead. Three sets of counts and links is what stays readable as one
#: sentence; a lineage longer than that is what the observation history is for.
_MAX_NAMED_REPORTS = 3


def _reports_line(
    n_rows: int,
    shown: int,
    shown_current: int,
    observations: Sequence[CommentObservation],
    *,
    dashboard_url: str | None,
    attributed: int | None = None,
    min_score: float = _DEFAULT_MIN_SCORE,
    noun: str = REGRESSION,
) -> str | None:
    """What each report in this lineage carried, as one line under the table.

    A wide night is not folded into a second copy of the table: a
    detector-removal sweep confirms three hundred near-identical rows, which no
    one reads whether or not they are behind a disclosure, and which GitHub will
    not accept in one comment anyway. So the table is a selection, and this line
    is what says so — per report, because the comment's evidence is assembled
    from several overlapping nightly snapshots and a single total would hide
    which one each count came from.

    Every date is its own link into that night's full report, where the complete
    set and every re-sorting of it live. Only the dates are linked — with the
    arrow that conventionally means "this opens somewhere else" — so the counts
    read as prose and the click target is the thing being opened.

    The rows counted as shown are every row the table drew, current and retained
    alike: the reader counts lines on the page, not the bookkeeping behind them.
    *attributed* is the size of the attributed pool the table drew from, or
    ``None`` when it fell back to every row of the window.
    """
    counts = _report_counts(observations)
    # Nothing to add when one report's regressions all reached the table: the
    # line would restate the rows immediately above it.
    if not counts or (len(counts) < 2 and shown_current >= n_rows):
        return None
    named = []
    for index, (night, regressions) in enumerate(counts[:_MAX_NAMED_REPORTS]):
        href = nightly_report_href(dashboard_url, night)
        where = f"[{night} report ↗]({href})" if href else f"{night} report"
        # The noun is carried once, by the leading clause; repeating it on every
        # report turns one sentence into a list of identical clauses.
        of_noun = f" of {noun}s" if index == 0 else ""
        if regressions is None:
            # Converged sibling windows whose overlap this state cannot resolve
            # (:func:`_report_counts`). The report is still named — its rows are
            # in the table above — but no number is invented for it.
            count = f"an unspecified number{of_noun}"
        elif index == 0:
            count = f"**{_count(regressions, noun)}**"
        else:
            count = f"**{regressions}**"
        named.append(f"{count} in the {where}")
    line = ", ".join(named)
    if len(counts) > _MAX_NAMED_REPORTS:
        line += (
            f", and {_count(len(counts) - _MAX_NAMED_REPORTS, 'earlier report')}"
        )
    if attributed is None:
        drawn = (
            "the most likely one is shown above" if shown == 1
            else f"the {shown} most likely are shown above"
        )
    else:
        which = f"attributed to this PR at {_pct(min_score)} or above"
        if shown < attributed:
            drawn = f"the strongest {shown} of the {attributed} {which} are shown above"
        elif shown == 1:
            drawn = f"the one {which} is shown above"
        else:
            drawn = f"the {shown} {which} are shown above"
    return f"{line} — {drawn}."


#: What a likelihood cell says when no model ever scored that pair. Short enough
#: for a table cell, and a phrase rather than a number: "0%" would claim a
#: judgement, and an empty cell would look like a rendering bug.
_UNSCORED = "_not scored_"


#: And what it says when the pipeline knows why there is no score — this pull
#: request is not in the commit range behind that regression at all. Stated
#: plainly because it is the one cell in the table that argues for the reader.
_NOT_A_CANDIDATE = "_not a candidate_"


def _likelihood_cell(row: RegressionRow, attribution: Attribution | None) -> str:
    """A row's attribution cell.

    A row nobody scored says so in words. Those rows are in the table because
    the window is what the comment is about — a regression this pull request was
    not even a candidate for is evidence a reader should see, and it must not
    arrive wearing a percentage."""
    if row.scope_state == "not_candidate":
        return _NOT_A_CANDIDATE
    likelihood = _likelihood(row, attribution)
    return _UNSCORED if likelihood is None else _pct(likelihood)


def _change_cell(pct_change: float | None) -> str:
    """A metric's step as a signed percentage with a direction marker.
    ``pct_change`` is a fraction on :class:`MetricVerdict`, matching the
    report's own formatting. The markers are monochrome glyphs, not coloured
    ones: they state the sign of the step and nothing more, because k4Bench
    cannot tell whether a step in either direction is welcome — a drop can be
    an optimization or work that is no longer being done
    (:class:`~k4bench.regression.models.Direction`). The gap between marker and
    number is non-breaking so a narrow column wraps the cell as a whole rather
    than stranding the marker on its own line.

    Emphasised with ``**``, so every caller must be somewhere GitHub renders
    Markdown — inside a raw ``<summary>`` the asterisks would reach the reader
    as asterisks."""
    if pct_change is None or not math.isfinite(pct_change):
        return "—"
    arrow = "▲" if pct_change >= 0 else "▼"
    return f"{arrow}&nbsp;**{pct_change:+.1%}**"
