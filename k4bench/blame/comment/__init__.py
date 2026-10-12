"""Decide which pull requests hear about a regression, and what they are told.

The nightly ranker already answers "which PR most likely caused this step"
(:mod:`k4bench.blame.rank`), but that answer only reaches people who read the
e-group mail or open the dashboard — never the author of the change. This package
turns a night's ``report.json`` + ``blame.json`` into a set of pull-request
comments; :mod:`k4bench.blame.publish` posts them.

The work happens in three steps, because they have different failure domains:

* :func:`select` (:mod:`.selection`) is **pure** — no network, no token, no
  clock — so the whole "who gets told what" decision is unit-testable, and the
  CLI can print exactly what would be posted (``--dry-run``) without touching
  GitHub.
* :func:`build_comments` (:mod:`.build`) reviews those selections into
  drafts, and is where the optional second model pass
  (:mod:`k4bench.blame.attribute`) and the diff fetch it needs arrive — as
  *injected callables*, the same seam :mod:`k4bench.blame.builder` uses for
  its ranker.
* :meth:`CommentDraft.render` (:mod:`.render`) writes the body once the
  publisher has read the comments already posted in the draft's lineage, whose
  hidden markers (:mod:`.state`) carry the state a new version builds on. The
  facts digest that decides whether a standing comment is edited is
  :mod:`.digest`'s, and untrusted prose is quoted through :mod:`.markdown`.

Commenting in someone else's repository is an outward-facing act on the strength
of a model's judgement, so the gates are deliberately narrow and all of them
must pass:

* the candidate's repository is on the **allowlist** — an empty allowlist means
  the bot is inert;
* the ranker actually **judged** the candidate — an unranked one has no opinion
  attached to it, and no threshold is low enough to be cleared by a missing
  judgement;
* the ranker's likelihood is at or above ``min_score`` (default 80);
* the pull request is **merged** — an open PR cannot have shipped in a release;
* the blame entry's candidate discovery and changed-file evidence were **complete**
  (:attr:`~k4bench.blame.models.BlameEntry.discovery_incomplete`) — naming one PR
  out of a knowingly partial set, or from partial path evidence, is exactly the
  overclaim the ranker itself refuses to make;
* the night is under the ``max_comments`` cap — a storm is a bug, not a night;
* and, when a cross-configuration review ran, it did not acquit the pull request
  outright (:func:`build_comments`'s withdrawal gate).

One comment covers one pull request and one change-window lineage — the reader's
question is "did my change do this?", asked once. A strictly expanding or
contracting window is a newer view of the same finding, while non-containing
windows remain distinct. :func:`marker_for` names the current window, and the
publisher migrates the nearest comparable marker. A containing window's bounded
detail table reserves the globally strongest row and one row for each represented
onset before filling the remaining places by attribution likelihood. Rows that stopped being
confirmed are not simply dropped: each material version carries a bounded,
validated snapshot of its strongest rows (:class:`RetainedRow`), and the next
version's table resurfaces the ones that still outrank tonight's evidence —
dated, annotated with their current report standing, and never rescored. A
collapsed observation history retains the newest material reports behind
immutable dashboard links. The dashboard remains the complete view.

A comment is written once and thereafter only *edited*, never retracted: when
the regression resolves, or the candidate drops below ``min_score``, tonight's
selection simply stops producing it and the comment already on the pull request
is left exactly as it stands. That is deliberate. It records what the benchmarks
saw at the time — which remains true after the metric recovers — and silently
rewriting or deleting a comment people may have replied to is worse than leaving
a dated one in place. Follow-ups belong in the thread.
"""

from k4bench.blame.comment.build import (
    BodyFor,
    FilesFor,
    HistoricalEvidenceUnavailable,
    PatchFor,
    ReproducerUrlFor,
    RunInfoFor,
    build_comments,
)
from k4bench.blame.comment.digest import Reproducers
from k4bench.blame.comment.render import CommentDraft, PRComment
from k4bench.blame.comment.selection import (
    CommentConfigError,
    CommentPlan,
    CommentPolicy,
    CommentStormError,
    RegressionRow,
    select,
)
from k4bench.blame.comment.state import (
    MARKER_VERSION,
    ChangeWindow,
    CommentObservation,
    LineageState,
    RetainedRow,
    facts_digest_of,
    marker_for,
    window_contains,
    window_from_marker,
)

__all__ = [
    "MARKER_VERSION",
    "BodyFor",
    "ChangeWindow",
    "CommentConfigError",
    "CommentDraft",
    "CommentObservation",
    "CommentPlan",
    "CommentPolicy",
    "CommentStormError",
    "FilesFor",
    "HistoricalEvidenceUnavailable",
    "LineageState",
    "PRComment",
    "PatchFor",
    "RegressionRow",
    "ReproducerUrlFor",
    "Reproducers",
    "RetainedRow",
    "RunInfoFor",
    "build_comments",
    "facts_digest_of",
    "marker_for",
    "select",
    "window_contains",
    "window_from_marker",
]
