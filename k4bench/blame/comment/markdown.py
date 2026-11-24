"""Formatting and quoting primitives for a comment's Markdown.

Pull-request titles, descriptions and model summaries are untrusted text quoted
into a comment the bot posts in someone else's repository. :func:`_one_line`,
:func:`_sentences` and :func:`_defang` are the only way such text enters a body:
flattened, clipped, and with every mention, cross-reference, link and tag made
inert. :func:`_cell` keeps any text inside its table column.
"""

from __future__ import annotations

import re

#: Longest model explanation quoted verbatim — the per-configuration ranker's
#: one-liner, or the cross-configuration review's short paragraph. Both contracts
#: ask for less than this; a model that ignores its contract must not be able to
#: paste an essay into someone's pull request.
_MAX_DESCRIPTION_CHARS = 400
_MAX_SUMMARY_CHARS = 700


def _pct(score: float) -> str:
    return f"{int(round(score))}%"


#: What a comment calls the steps it counts. "Regression" only when every one
#: of them went up; otherwise the neutral term, because a drop is no more an
#: improvement than a rise is a regression
#: (:class:`~k4bench.regression.models.Direction`), and calling a drop a
#: regression is simply wrong.
REGRESSION = "regression"
PERFORMANCE_CHANGE = "performance change"


def _noun_for(directions) -> str:
    """:data:`REGRESSION` when every direction in *directions* is ``UP``, else
    :data:`PERFORMANCE_CHANGE`."""
    return REGRESSION if all(d == "UP" for d in directions) else PERFORMANCE_CHANGE


def _capitalized(noun: str) -> str:
    return noun[:1].upper() + noun[1:]


#: The sign words :func:`_direction_detail` counts with, and their article.
_SIGNS = (("decrease", "a"), ("increase", "an"), ("unsigned change", "an"))


def _direction_detail(up: int, down: int, none: int = 0) -> str:
    """``a decrease`` / ``an increase`` / ``all decreases`` /
    ``221 decreases, 2 increases`` — the signs behind a count of performance
    changes, largest first."""
    parts = [
        (n, word, article)
        for n, (word, article) in zip((down, up, none), _SIGNS, strict=True) if n
    ]
    if sum(n for n, _, _ in parts) == 1:
        _, word, article = parts[0]
        return f"{article} {word}"
    if len(parts) == 1:
        return f"all {parts[0][1]}s"
    parts.sort(key=lambda part: -part[0])
    return ", ".join(_count(n, word) for n, word, _ in parts)


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


def _sentences(text: str, limit: int) -> str:
    """:func:`_one_line`, cut at the last sentence that fits rather than
    mid-sentence — the review's summary is quoted as its reasoning, and a
    clause broken off after a comma can say the opposite of the sentence it
    came from. Falls back to the plain clip when not even one sentence fits."""
    flat = _one_line(text, 10 * limit)
    if len(flat) <= limit:
        return flat
    ends = [
        m.end() for m in re.finditer(r"[.!?](?=\s)", flat[: limit - 1])
    ]
    if ends and ends[-1] >= limit // 3:
        return flat[: ends[-1]] + " …"
    return _one_line(text, limit)


def _defang(text: str) -> str:
    """Neutralise the active Markdown/HTML in quoted, externally-authored prose.

    A PR title or a model's summary is untrusted text pasted into a comment the
    bot posts in someone else's repository. Left as-is it could:

    * ``@login`` — ping a person on every nightly edit (the same ban the whole
      bot honours by never rendering an author with an ``@``);
    * ``#123`` — cross-reference an unrelated issue, notifying its subscribers;
      a title like "Revert #45" carries one for free (:func:`_pr_ref` applies the
      same rule to the references this module writes itself, and the review is
      asked to name alternatives in exactly that form);
    * ``<!-- … -->`` / ``<tag>`` — hide following content, or inject markup;
    * ``[text](url)`` / ``![alt](url)`` — put an arbitrary clickable destination,
      or a remote image, into a comment the bot signs its own name to;
    * ``https://...`` / ``www....`` — GitHub autolinks a bare URL, so the prose
      needs no Markdown at all to become a link. A pull-request URL autolinked
      this way also cross-references that PR's timeline, which is the very
      notification :func:`_pr_ref` goes out of its way not to send.

    A zero-width space at each sequence's join breaks what GitHub would act on
    while leaving the text visually unchanged: after the trigger character for
    the prefix forms, and between ``]`` and ``(`` for a link, whose two halves
    are what make it one. Emphasis and backticks are deliberately left alone —
    they restyle the quoted text but cannot carry a reader anywhere. Table pipes
    are left to :func:`_cell`, which the cell paths still apply on top of this."""
    zwsp = "​"  # U+200B zero-width space
    return (
        text.replace("@", "@" + zwsp)
        .replace("#", "#" + zwsp)
        .replace("<", "<" + zwsp)
        .replace("](", "]" + zwsp + "(")
        .replace("![", "!" + zwsp + "[")
        .replace("://", ":" + zwsp + "//")
        .replace("www.", "www" + zwsp + ".")
        .replace("WWW.", "WWW" + zwsp + ".")
    )


def _cell(text: str | None) -> str:
    """Text safe inside a Markdown table cell: a pipe would end the column."""
    return (text or "").replace("|", "\\|")
