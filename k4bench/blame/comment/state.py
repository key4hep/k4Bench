"""The hidden state a pull-request comment carries for its next version.

A comment is written once and afterwards only edited, so everything a later
version needs to know about the earlier ones travels inside the body as hidden
HTML lines: the window marker the upsert recognises, the facts digest, the exact
regression identities behind the headline, the observation history and the
retained rows. This module writes and reads those lines and nothing else.

A posted comment is anyone's to edit, so every decoder here treats its input as
hostile: bounded, strictly validated, and ignored whole when any check fails.
Rendering tonight's rows without history is safe; rendering forged or garbled
history is not.
"""

from __future__ import annotations

import json
import math
import zlib
from base64 import b64decode, urlsafe_b64encode
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from urllib.parse import urlsplit

from k4bench.blame.prompt import movement
from k4bench.layout import SIM, SUITES
from k4bench.regression.models import MetricVerdict
from k4bench.regression.render import window_token

#: Marker format version. Bumping it makes every existing comment invisible to
#: the upsert (a *new* comment is posted rather than the old one edited), so it
#: changes only when a body is no longer an in-place successor of the old one.
MARKER_VERSION = "v1"

#: Material versions retained in one comment. Each carries an archived report
#: link twice (visible and in its machine-readable marker), so an unbounded list
#: would eventually cross GitHub's body limit and make the lineage unwritable.
_MAX_OBSERVATIONS = 20
_OBSERVATION_PREFIX = "<!-- k4bench-blame-observation:v1 "
_OMITTED_OBSERVATIONS_PREFIX = "<!-- k4bench-blame-observations-omitted:v1 "

#: Exact cumulative regression identities and their latest published
#: attribution for this comment lineage. Report snapshots overlap heavily, so
#: adding their row counts would over-count; the marker carries the union itself
#: and lets a converging lineage merge all of its parents without scraping
#: Markdown.
#:
#: Versioned because the identities inside are keyed on the run label: a marker
#: written under a different label vocabulary names rows that no longer match
#: anything, and a comment is not ours to migrate. Bump the version whenever
#: that vocabulary changes — an unreadable marker costs one lineage's merged
#: state, where a misread one silently doubles every row it carries.
_CUMULATIVE_PREFIX = "<!-- k4bench-blame-cumulative:v2 "
_MAX_CUMULATIVE_IDENTITIES = 10_000
_MAX_CUMULATIVE_BYTES = 256_000

#: What the marker may take of GitHub's 65,536-byte comment body, measured on
#: the encoded line that is actually written. The two caps above bound the JSON
#: *before* compression, which is the guard the decoder needs against a hostile
#: marker but says little about the size of the resulting line — a body over the
#: limit is rejected outright by the API, so the written form has to be bounded
#: in its own right. A quarter of the budget leaves the table, the assessment
#: and the retained-row marker their share.
_MAX_CUMULATIVE_MARKER_BYTES = 16_384

#: Retained-row snapshots carried in one comment's single hidden state marker —
#: the strongest candidates for the table, current and historical alike. The cap
#: bounds the marker (and so the body) however wide a night is; anything past it
#: is dropped strongest-last, and the dashboard remains the complete view.
_MAX_RETAINED_ROWS = 20

#: Versioned for the same reason as :data:`_CUMULATIVE_PREFIX`: a retained row
#: is keyed on the run label, so a snapshot written under another vocabulary
#: renders beside tonight's rows as a separate regression rather than as their
#: own history.
_RETAINED_PREFIX = "<!-- k4bench-blame-retained:v2 "

#: Every hidden state line ends here; what lies between its prefix and this is
#: its payload (:func:`_marker_line`).
_MARKER_SUFFIX = " -->"


@dataclass(frozen=True)
class CommentObservation:
    """One materially different report snapshot retained in a PR comment."""

    report_night: str
    base_release: str | None
    onset_release: str
    regressions: int
    scopes: int
    up: int
    down: int
    none: int
    url: str | None = None


@dataclass(frozen=True)
class RetainedRow:
    """One regression row's snapshot as a comment's hidden state retains it.

    A row that stops being confirmed disappears from tonight's report, and with
    it from the plan — but a reader who was shown "``mean_time_s`` +36.7%, 88%"
    two nights ago deserves to keep seeing that finding, dated, rather than have
    it silently vanish under weaker rows. So every material comment version
    carries a bounded list of these snapshots in one versioned marker, and the
    next version's table can resurface the ones no longer confirmed.

    Every field is frozen at the row's **last published version**, which is the
    last night it was confirmed *and* the comment was materially edited. The
    two differ: :func:`k4bench.blame.publish._upsert` writes nothing when the
    facts digest is unchanged, and neither the report night nor a model score
    is hashed into that digest, so a night that reconfirms a row on identical
    evidence leaves this snapshot untouched. ``last_reported`` is therefore the
    newest night this row was *published* as confirmed, never the newest night
    it was confirmed — hence the name, and hence the column heading the table
    renders. ``likelihood`` and ``source`` carry the same caveat: they are what
    the review recorded on that published night, never rescored afterwards.

    ``state`` is the pull request's scope standing on that night
    (:data:`~k4bench.blame.attribute.ScopeCandidateState`), kept so a
    ``not_candidate`` row can never be resurrected wearing a percentage. The
    window, ``stack``, ``onset`` and the two run ids are exactly the fields a
    dashboard deep link needs, so no URL is ever stored — ``base_run`` and
    ``onset_run`` qualify a same-release window, which its releases alone
    cannot identify (see :func:`~k4bench.regression.render.window_token`).
    Decoded state is validated as strictly as the observation markers; a marker
    that fails any check is ignored whole and the current rows render on their
    own.
    """

    detector: str
    platform: str
    sample: str
    label: str
    metric: str
    sub_detector: str
    direction: str
    pct: float | None
    onset: str
    onset_run: str
    base_release: str | None
    base_run: str
    onset_release: str
    stack: str
    last_reported: str
    likelihood: float | None
    source: str
    state: str
    #: The row's own last-accepted release — the tight end of the window the
    #: table shows for it, which can be later than the comment's
    #: ``base_release``. Defaulted because markers written before the column
    #: existed carry no such field; those fall back to ``base_release``, the
    #: window they were rendered under.
    base: str = ""
    #: The recipe published for this row on the night it was last confirmed.
    #: Carried rather than rebuilt: the file name is derivable from the fields
    #: above, but only the night that actually published it can vouch for the
    #: file being there, and a link that 404s is worse than no link.
    recipe: str = ""


@dataclass(frozen=True)
class LineageState:
    """The hidden state carried by the comments already posted in one lineage,
    merged across every comparable body and read from structured markers only,
    never from presentation Markdown.

    * ``cumulative`` — every regression identity the lineage has published,
      with its newest published attribution;
    * ``observations`` — the material versions, keyed by report night *and*
      window, with the survivor winning an exact collision, plus the most any
      one lineage aged out (``omitted``);
    * ``retained`` — one snapshot per row identity, the newest confirmation
      winning.
    """

    cumulative: dict[tuple, tuple[float | None, str, str]] = field(default_factory=dict)
    observations: dict[tuple[str, ChangeWindow], CommentObservation] = field(
        default_factory=dict
    )
    omitted: int = 0
    retained: tuple[RetainedRow, ...] = ()

    @classmethod
    def from_bodies(cls, bodies: Sequence[str]) -> LineageState:
        cumulative: dict[tuple, tuple[float | None, str, str]] = {}
        for body in bodies:
            for identity, recorded in _decoded_cumulative(body).items():
                kept = cumulative.get(identity)
                if kept is None or recorded[2] > kept[2]:
                    cumulative[identity] = recorded
        # Absorbed lineages first, so the survivor wins an exact (night, window)
        # collision: that pair really is one finding recorded twice, and the
        # lineage this comment continues is the one to believe.
        observations: dict[tuple[str, ChangeWindow], CommentObservation] = {}
        omitted = 0
        for body in reversed(list(bodies)):
            recorded_observations, aged_out = _observations(body)
            # Not summed: each lineage counts what *it* aged out, and the merged
            # history is not the concatenation of them.
            omitted = max(omitted, aged_out)
            for item in recorded_observations:
                observations[_observation_key(item)] = item
        return cls(
            cumulative=cumulative,
            observations=observations,
            omitted=omitted,
            retained=tuple(_merged_retained(bodies)),
        )


def marker_for(
    base_release: str | None, onset_release: str | None, suite: str = SIM,
) -> str:
    """The hidden HTML key naming a comment's suite and current change window.

    A re-confirmation matches this exact key; a strictly expanding or contracting
    window of the same suite adopts and rewrites the nearest comparable key at
    publish time. Non-containing windows stay separate, and so do two suites'
    comments on one pull request: a reco step and a sim step attributed to one
    change are two findings, whatever their windows. The sim suite's key names
    no suite — every comment posted before there were others is its — and any
    other's says which it is. Reuses
    :func:`~k4bench.regression.render.window_token` so the key and the dashboard
    link it carries name the window identically.
    """
    named = "" if suite == SIM else f"suite={suite} "
    return (
        f"<!-- k4bench-blame-comment:{MARKER_VERSION} "
        f"{named}window={window_token(base_release, onset_release)} -->"
    )


ChangeWindow = tuple[str | None, str]


def window_contains(outer: ChangeWindow, inner: ChangeWindow) -> bool:
    """Whether half-open *outer* fully contains half-open *inner*.

    ``None`` is an unbounded base, so only another unbounded window can contain
    it. Release identifiers are ISO dates in report data and therefore sort in
    their chronological order as strings.
    """
    outer_base, outer_onset = outer
    inner_base, inner_onset = inner
    base_contains = outer_base is None or (
        inner_base is not None and outer_base <= inner_base
    )
    return base_contains and outer_onset >= inner_onset


def window_from_marker(marker: str, suite: str = SIM) -> ChangeWindow | None:
    """Parse one current-version marker of *suite* (:func:`marker_for`),
    rejecting anything not emitted here, and every other suite's."""
    named = "" if suite == SIM else f"suite={suite} "
    prefix = f"<!-- k4bench-blame-comment:{MARKER_VERSION} {named}window="
    suffix = " -->"
    if "\n" in marker or not marker.startswith(prefix) or not marker.endswith(suffix):
        return None
    token = marker[len(prefix):-len(suffix)]
    base, separator, onset = token.partition("..")
    if not separator or not onset or ".." in onset or any(c.isspace() for c in token):
        return None
    return (base or None, onset)


def parse_marker(marker: str) -> tuple[str, ChangeWindow] | None:
    """The suite and window one current-version marker names
    (:func:`marker_for`), or ``None`` for anything not emitted here."""
    for suite in SUITES:
        if (window := window_from_marker(marker, suite)) is not None:
            return suite, window
    return None


#: Prefix of the second hidden line — see :func:`_digest_from` and
#: :func:`facts_digest_of`.
_FACTS_MARKER_PREFIX = "<!-- k4bench-blame-facts:"


def facts_digest_of(body: str) -> str:
    """The facts digest carried by an already-posted comment, or ``""``.

    The read half of :func:`_digest_from`, used by
    :func:`k4bench.blame.publish._upsert` to decide whether a differing body
    represents a real change. A body carrying no readable digest line returns
    ``""``, and the caller falls back to comparing whole bodies."""
    for line in body.split("\n", 3)[:3]:
        line = line.strip()
        if line.startswith(_FACTS_MARKER_PREFIX) and line.endswith("-->"):
            return line[len(_FACTS_MARKER_PREFIX):-3].strip()
    return ""


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _marker_line(prefix: str, data: bytes) -> str:
    """One hidden state line carrying *data* as unpadded urlsafe base64."""
    encoded = urlsafe_b64encode(data).decode().rstrip("=")
    return f"{prefix}{encoded}{_MARKER_SUFFIX}"


def _marker_values(body: str, prefix: str) -> Iterator[str]:
    """What follows *prefix* on every hidden line of *body* that carries it."""
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix) and line.endswith(_MARKER_SUFFIX):
            yield line[len(prefix):-len(_MARKER_SUFFIX)]


def _marker_data(encoded: str) -> bytes | None:
    """The bytes :func:`_marker_line` wrote, or ``None`` for a value that is not
    urlsafe base64 — padded or not, since a comment is anyone's to edit."""
    try:
        return b64decode(
            encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True,
        )
    except ValueError:  # binascii.Error, or a value that is not ASCII
        return None


def _cumulative_marker(
    state: dict[tuple, tuple[float | None, str, str]]
) -> str:
    if len(state) > _MAX_CUMULATIVE_IDENTITIES:
        return ""
    payload = [
        [*identity, likelihood, source, last_reported]
        for identity, (likelihood, source, last_reported) in sorted(state.items())
    ]
    raw = _canonical_json(payload)
    if len(raw) > _MAX_CUMULATIVE_BYTES:
        return ""
    marker = _marker_line(_CUMULATIVE_PREFIX, zlib.compress(raw, level=9))
    # Compression normally makes this moot; when it does not, dropping the
    # marker costs one night's merge — the state is still carried by the bodies
    # this one merged — while writing it would cost the whole comment.
    if len(marker.encode()) > _MAX_CUMULATIVE_MARKER_BYTES:
        return ""
    return marker


def _decoded_cumulative(
    body: str,
) -> dict[tuple, tuple[float | None, str, str]]:
    """The identities a posted body carries — or nothing for a marker that fails
    a single check, read from the first cumulative line only. Decompression is
    bounded, so a hostile marker cannot expand past the cap the writer obeys."""
    encoded = next(_marker_values(body, _CUMULATIVE_PREFIX), None)
    compressed = _marker_data(encoded) if encoded is not None else None
    if compressed is None:
        return {}
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, _MAX_CUMULATIVE_BYTES + 1)
        if (
            len(raw) > _MAX_CUMULATIVE_BYTES
            or decoder.unconsumed_tail
            or decoder.unused_data
            or not decoder.eof
        ):
            return {}
        values = json.loads(raw)
    except (ValueError, zlib.error):
        return {}
    if not isinstance(values, list) or len(values) > _MAX_CUMULATIVE_IDENTITIES:
        return {}
    state = {}
    for value in values:
        if (
            not isinstance(value, list)
            or len(value) != 9
            or not all(
                isinstance(part, str) and len(part) <= 2048
                for part in value[:6]
            )
        ):
            return {}
        likelihood, source, last_reported = value[6:]
        if likelihood is not None and (
            isinstance(likelihood, bool)
            or not isinstance(likelihood, (int, float))
            or not math.isfinite(likelihood)
            or not 0 <= likelihood <= 100
        ):
            return {}
        if source not in ("", "reviewer", "ranker") or not isinstance(
            last_reported, str
        ) or len(last_reported) > 64:
            return {}
        state[tuple(value[:6])] = (likelihood, source, last_reported)
    return state


def _cumulative_identities(body: str) -> set[tuple]:
    return set(_decoded_cumulative(body))


def _observation_key(observation: CommentObservation) -> tuple[str, ChangeWindow]:
    """What makes two observations the same finding: night *and* window."""
    return (
        observation.report_night,
        (observation.base_release, observation.onset_release),
    )


def _report_counts(
    observations: Sequence[CommentObservation],
) -> list[tuple[str, int | None]]:
    """``(report night, regressions)`` newest first, ``None`` where unknowable.

    One night can carry several observations once lineages have converged, and
    what their counts mean together depends on how their windows sit:

    * a window contained in another contributes nothing of its own — its rows
      are a subset of the containing window's, which already counts them;
    * the remaining (maximal) windows, when pairwise **disjoint**, partition the
      night's regressions, so their counts add exactly;
    * two maximal windows can still *partially* overlap — ``(09-01, 09-03]`` and
      ``(09-02, 09-04]`` are incomparable, so nothing collapses them — and then
      the rows in the shared span are counted by both. Observations are
      aggregates, not identity sets, so that union is simply not recoverable
      from this state, and the night is reported **without** a count rather than
      with a plausible-looking wrong one.

    The alternative — recording every night each identity was seen, instead of
    the latest — would make this exact, at the cost of growing a marker that is
    already size-capped, for a case this narrow.
    """
    by_night: dict[str, list[CommentObservation]] = {}
    for item in observations:
        if _iso_date(item.report_night) and item.regressions > 0:
            by_night.setdefault(item.report_night, []).append(item)
    counts: list[tuple[str, int | None]] = []
    for night in sorted(by_night, reverse=True):
        windows = [
            ((item.base_release, item.onset_release), item.regressions)
            for item in by_night[night]
        ]
        maximal = [
            (window, regressions) for window, regressions in windows
            if not any(
                other != window and window_contains(other, window)
                for other, _ in windows
            )
        ]
        counts.append((night, _disjoint_total(maximal)))
    return counts


def _disjoint_total(
    windows: list[tuple[ChangeWindow, int]]
) -> int | None:
    """The sum of *windows*' counts, or ``None`` if any two of them overlap.

    Half-open ``(base, onset]``, so two windows are disjoint exactly when the
    later one starts at or after the earlier one ends. An unbounded base reaches
    back forever, so it can only ever be the first."""
    ordered = sorted(
        windows, key=lambda entry: (entry[0][0] is not None, entry[0][0] or "", entry[0][1])
    )
    total = 0
    previous_onset: str | None = None
    for (base, onset), regressions in ordered:
        if previous_onset is not None and (base is None or base < previous_onset):
            return None
        total += regressions
        previous_onset = onset
    return total


def _observation_marker(observation: CommentObservation) -> str:
    payload = {
        "report_night": observation.report_night,
        "base_release": observation.base_release,
        "onset_release": observation.onset_release,
        "regressions": observation.regressions,
        "scopes": observation.scopes,
        "up": observation.up,
        "down": observation.down,
        "none": observation.none,
        "url": observation.url,
    }
    return _marker_line(_OBSERVATION_PREFIX, _canonical_json(payload))


def _observations(body: str) -> tuple[list[CommentObservation], int]:
    """The valid observations a posted body carries, and how many it records as
    aged out. A marker that fails any check is skipped on its own."""
    omitted = max(
        (
            int(value) for value in _marker_values(body, _OMITTED_OBSERVATIONS_PREFIX)
            if len(value) <= 9 and value.isascii() and value.isdigit()
        ),
        default=0,
    )
    observations = []
    for encoded in _marker_values(body, _OBSERVATION_PREFIX):
        data = _marker_data(encoded)
        if data is None:
            continue
        try:
            observation = CommentObservation(**json.loads(data))
        except (TypeError, ValueError):
            continue
        if _valid_observation(observation):
            observations.append(observation)
    return observations, omitted


def _valid_observation(observation: CommentObservation) -> bool:
    counts = (
        observation.regressions,
        observation.scopes,
        observation.up,
        observation.down,
        observation.none,
    )
    return (
        _iso_date(observation.report_night)
        and _iso_date(observation.onset_release)
        and (
            observation.base_release is None
            or _iso_date(observation.base_release)
        )
        and all(
            not isinstance(value, bool) and isinstance(value, int) and value >= 0
            for value in counts
        )
        and observation.regressions > 0
        and 0 < observation.scopes <= observation.regressions
        and observation.up + observation.down + observation.none
        == observation.regressions
        and _safe_observation_url(observation.url)
    )


def _iso_date(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _safe_observation_url(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, str) or not value or len(value) > 4096:
        return False
    if any(char.isspace() or char in "<>()\\\"'" for char in value):
        return False
    try:
        split = urlsplit(value)
        return (
            split.scheme in {"http", "https"}
            and bool(split.netloc)
            and split.username is None
            and split.password is None
            and not split.fragment
        )
    except ValueError:
        return False


def _retained_marker(rows: list[RetainedRow]) -> str | None:
    """The whole retained state as one compact hidden line — one marker per
    comment, never one per row per report."""
    if not rows:
        return None
    return _marker_line(
        _RETAINED_PREFIX, _canonical_json({"rows": [asdict(row) for row in rows]}),
    )


def _merged_retained(bodies: Sequence[str]) -> list[RetainedRow]:
    """Every valid retained row across *bodies*, one per identity.

    When converging lineages carry the same identity, the snapshot with the
    newest confirmation is the truer record of what was last claimed."""
    merged: dict[tuple, RetainedRow] = {}
    for body in bodies:
        for row in _decoded_retained(body):
            key = _retained_identity(row)
            kept = merged.get(key)
            if kept is None or row.last_reported > kept.last_reported:
                merged[key] = row
    return list(merged.values())


def _decoded_retained(body: str) -> list[RetainedRow]:
    """The retained rows a posted body carries — or nothing for a marker that
    fails a single check, because rendering tonight's rows without history is
    safe and rendering forged or garbled history is not."""
    rows: list[RetainedRow] = []
    for encoded in _marker_values(body, _RETAINED_PREFIX):
        data = _marker_data(encoded)
        if data is None:
            continue
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        decoded = _validated_retained(payload)
        if decoded is not None:
            rows.extend(decoded)
    return rows


def _validated_retained(payload: object) -> list[RetainedRow] | None:
    """*payload* as retained rows, or ``None`` unless every check passes: the
    exact schema, the row-count cap, and every field bound of
    :func:`_valid_retained`."""
    if not isinstance(payload, dict) or set(payload) != {"rows"}:
        return None
    items = payload["rows"]
    if not isinstance(items, list) or len(items) > _MAX_RETAINED_ROWS:
        return None
    rows = []
    for item in items:
        if not isinstance(item, dict) or not all(
            isinstance(key, str) for key in item
        ):
            return None
        try:
            row = RetainedRow(**item)
        except TypeError:
            return None
        if not _valid_retained(row):
            return None
        rows.append(row)
    return rows


_RETAINED_DIRECTIONS = frozenset({"UP", "DOWN", "NONE"})
_RETAINED_STATES = frozenset({
    "ranked", "unranked", "not_candidate", "discovery_incomplete",
})
_RETAINED_SOURCES = frozenset({"reviewer", "ranker", ""})
_MAX_RETAINED_FIELD_CHARS = 200


def _bounded_name(value: object, *, required: bool = True) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= _MAX_RETAINED_FIELD_CHARS
        and (bool(value) or not required)
        and (not value or value.isprintable())
    )


def _valid_retained(row: RetainedRow) -> bool:
    names = (row.detector, row.platform, row.sample, row.label, row.metric)
    likelihood = row.likelihood
    return (
        all(_bounded_name(value) for value in names)
        and _bounded_name(row.sub_detector, required=False)
        # A run group with no recorded release is a link that falls back to the
        # unscoped window, not a reason to discard a whole marker of valid rows.
        and _bounded_name(row.stack, required=False)
        and _bounded_name(row.onset_run, required=False)
        and not any(char.isspace() for char in row.onset_run)
        and _bounded_name(row.base_run, required=False)
        and not any(char.isspace() for char in row.base_run)
        and _iso_date(row.last_reported)
        and _iso_date(row.onset_release)
        and (row.base_release is None or _iso_date(row.base_release))
        and (row.onset == "" or _iso_date(row.onset))
        and (row.base == "" or _iso_date(row.base))
        # A link the table renders, decoded from a comment anyone can edit:
        # bounded, and https or nothing at all.
        and isinstance(row.recipe, str)
        and len(row.recipe) <= 512
        and (row.recipe == "" or row.recipe.startswith("https://"))
        and not any(char.isspace() for char in row.recipe)
        and row.direction in _RETAINED_DIRECTIONS
        and row.state in _RETAINED_STATES
        and row.source in _RETAINED_SOURCES
        and (
            likelihood is None
            or (
                not isinstance(likelihood, bool)
                and isinstance(likelihood, int | float)
                and math.isfinite(likelihood)
                and 0 <= likelihood <= 100
            )
        )
        # A missing number and a missing producer must agree, and the state the
        # pipeline *knows* has no number can never be smuggled one.
        and (likelihood is None) == (row.source == "")
        and not (row.state == "not_candidate" and likelihood is not None)
        and (
            row.pct is None
            or (
                not isinstance(row.pct, bool)
                and isinstance(row.pct, int | float)
                and math.isfinite(row.pct)
            )
        )
    )


def _retained_identity(row: RetainedRow) -> tuple:
    return (
        row.detector, row.platform, row.sample, row.label, row.metric,
        row.sub_detector,
    )


def _retained_window(row: RetainedRow) -> tuple[str, str]:
    """A retained row's change window as it is shown. A snapshot taken before
    the column existed recorded no window of its own, so it falls back to the
    comment window it was published under — wider than the row's own, never
    narrower, and exactly what that reader was shown."""
    return (row.base or row.base_release or "", row.onset)


def _retained_sort_key(row: RetainedRow) -> tuple:
    """The same shape :func:`_row_sort_key` produces, so current and retained
    rows rank in one pool: likelihood first, the baseline configuration before
    the others, then movement, identity last."""
    likelihood = row.likelihood
    return (
        likelihood is None,
        -(likelihood if likelihood is not None else 0.0),
        row.label != "baseline",
        -movement(row.pct),
        *_retained_identity(row),
    )


def _verdict_identity(v: MetricVerdict) -> tuple:
    """The six-field identity every table, digest and retained-state structure
    keys a row on — one definition, so a current row and its retained snapshot
    can never disagree about which regression they are."""
    return (
        v.detector, v.platform, v.sample, v.label, v.metric, v.sub_detector or "",
    )
