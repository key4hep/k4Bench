"""Stack Changes tab — what moved in Key4hep between two nightlies.

Like the Regressions tab this view is cross-detector: a Key4hep release is one
stack, sourced identically by every detector benchmarked against it, so the
only axis that scopes it is the platform. Which detector supplied the answer is
an implementation detail — the tab reads whichever run it finds first.

The comparison is between **releases**, never run dates. The nightly build
lags: on a day with no new nightly the benchmark sources the newest one
available, so consecutive run dates routinely share one identical stack — most
of the run dates on EOS are not release dates at all. Comparing run dates would
manufacture changes that never happened.
"""

from __future__ import annotations

from urllib.parse import urlencode

import pandas as pd
import requests
import streamlit as st

from k4bench.provenance.diff import ADDED, CHANGED, REMOVED, diff_packages, unchanged_packages
from k4bench.regression.render import _pretty_sample, from_json
from remote_cache import (
    _cached_fetch_report,
    _cached_fetch_stack_packages,
    _cached_list_detectors,
    _cached_list_report_dates,
    _cached_list_stacks,
)
from tabs import _blame
from ui_chrome import seed_query_param
from ui_utils import _METRIC_LABELS

#: Releases are stored as ``key4hep-{YYYY-MM-DD}`` directories; the tab talks
#: in the bare nightly tag the rest of the dashboard shows on its axes.
_PREFIX = "key4hep-"

#: This tab's section name and the query-param names its two pickers seed from
#: (see :func:`_seed`). Kept here, beside the widgets that read them, so a tab
#: that deep-links *into* this view (the Regressions blame note, via
#: :func:`deep_link`) shares one source of truth: rename a param and its builder
#: moves with it, instead of silently breaking a literal in another module.
_TAB_NAME = "Stack Changes"
PARAM_FROM = "from"
PARAM_TO = "to"


def deep_link(
    *, detector: str, platform: str, head_release: str, base_release: str | None = None
) -> str:
    """A relative query string that opens this tab seeded to a release range.

    ``base_release`` → ``head_release`` become the two pickers. Omitting
    *base_release* (an open-ended blame window) leaves the older end at the
    tab's own default for the user to choose.

    *detector* is carried even though this tab is cross-detector: the sidebar
    resolves the platform list *from the selected detector*, so seeding the
    platform without a detector that offers it would be rejected and the
    comparison would open on the wrong platform.
    """
    params = {
        "tab": _TAB_NAME, "detector": detector, "platform": platform,
        PARAM_TO: head_release,
    }
    if base_release:
        params[PARAM_FROM] = base_release
    return "?" + urlencode(params)

#: Link colour carries meaning here: it marks the one *action* in a row. The
#: package and its two commits are identifiers, and a grid column has no
#: styling of its own, so linking them would draw the whole table blue and
#: leave the eye nothing to land on. The compare view spans both commits, so
#: nothing is lost by being the only link.
_STATUS_BADGE = {CHANGED: "🔄 changed", ADDED: "➕ added", REMOVED: "➖ removed"}

#: Why the dates offered here are not the dates benchmarks ran on. Lives in the
#: pickers' tooltips: it explains a gap in the list ("where is 2026-07-14?")
#: rather than the diff, so it is worth a look on demand and not a standing line
#: of prose above the answer.
_TAG_HELP = (
    "These are Key4hep release tags, not benchmark run dates: the nightly build "
    "does not publish every day, and a run then re-uses the newest release "
    "available."
)

_DOCS_URL = (
    "https://key4hep.github.io/k4Bench/user-guide/features/dashboard/#stack-changes-tab"
)


def _release(stack: str) -> str:
    return stack.removeprefix(_PREFIX)


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _seed(key: str, param: str, options: list[str], default: str) -> None:
    """Seed a picker from ``?param=`` if present, else from *default*.

    Both halves must go through ``session_state``: Streamlit rejects a widget
    that sets ``session_state[key]`` *and* passes ``index=`` in the same run
    (see :func:`ui_chrome.seed_query_param`), so the default cannot be an
    ``index=`` argument on the selectbox.
    """
    seed_query_param(key, param, options)
    if key not in st.session_state:
        st.session_state[key] = default


def _from_default_for(releases: list[str]) -> str:
    """The "From release" default: normally the second-newest, but when a deep
    link seeds only ``?to=`` (an open-ended blame window), the release one older
    than that ``to`` — so the pickers open on an older→newer range rather than
    tripping the reversed-range warning. *releases* is newest-first."""
    to_seed = st.query_params.get(PARAM_TO)
    if PARAM_FROM not in st.query_params and to_seed in releases:
        i = releases.index(to_seed)
        return releases[i + 1] if i + 1 < len(releases) else releases[i]
    return releases[1]


def _stacks_for_platform(data_url: str, platform: str) -> list[str]:
    """Every release benchmarked on *platform*, newest first.

    Unioned across detectors rather than read from one: detectors join and
    leave the matrix, so no single detector's history is the full set of
    releases.
    """
    stacks: set[str] = set()
    for detector in _cached_list_detectors(data_url):
        try:
            stacks.update(_cached_list_stacks(data_url, detector, platform))
        except requests.RequestException:
            # A detector never benchmarked on this platform has no such
            # directory to list. Only that is expected here — a broader catch
            # would bury a bug in the listing itself as "no releases".
            continue
    return sorted(stacks, reverse=True)


def _packages(data_url: str, platform: str, stack: str) -> dict | None:
    """The release's package map, from whichever detector has it recorded.

    Detectors are tried in turn because any one of them may have skipped this
    release, or may have run it before provenance capture existed.
    """
    for detector in _cached_list_detectors(data_url):
        packages = _cached_fetch_stack_packages(data_url, detector, platform, stack)
        if packages:
            return packages
    return None


def packages_for_release(data_url: str, platform: str, release: str) -> dict | None:
    """The package map for a bare release tag (e.g. ``2026-07-10``) on *platform*,
    from whichever detector recorded it. The public provenance lookup other tabs
    use, so they need neither the ``key4hep-`` prefix nor this module's
    internals."""
    return _packages(data_url, platform, _PREFIX + release)




def render(data_url: str, platform: str) -> None:
    st.caption(
        "Which Key4hep packages moved between two nightly releases — the upstream "
        f"changes a regression between them could be attributed to. "
        f"[Learn more →]({_DOCS_URL})"
    )

    stacks = _stacks_for_platform(data_url, platform)
    if len(stacks) < 2:
        st.info(
            f"Need at least two Key4hep releases on `{platform}` to compare; "
            f"found {len(stacks)}."
        )
        return

    releases = [_release(s) for s in stacks]
    # Default to the two most recent releases: "what came in last night?" is the
    # question this tab exists to answer. Defaulting both pickers to the newest
    # would open the tab on "pick two different releases" instead.
    col_from, col_to = st.columns(2)
    with col_from:
        # Default the baseline to the second-newest release ("what came in last
        # night?"), except when a deep link seeds only `to`: then default `from`
        # to the release just older than it, so an open-ended blame link
        # (onset far in the past) opens on a valid older→newer range instead of
        # a reversed one.
        from_default = _from_default_for(releases)
        _seed("stack_from", PARAM_FROM, releases, from_default)
        base_release = st.selectbox(
            "From release", releases, key="stack_from",
            help=f"The older nightly tag — the accepted baseline. {_TAG_HELP}",
        )
    with col_to:
        _seed("stack_to", PARAM_TO, releases, releases[0])
        head_release = st.selectbox(
            "To release", releases, key="stack_to",
            help=f"The newer nightly tag. {_TAG_HELP}",
        )
    st.query_params[PARAM_FROM] = base_release
    st.query_params[PARAM_TO] = head_release

    if base_release == head_release:
        st.info("Pick two different releases to compare.")
        return
    # Ordered by position in the newest-first list rather than by comparing the
    # tags, so this and _span read the release order the same single way.
    if releases.index(base_release) < releases.index(head_release):
        # Reversing the range would report every change with its sign flipped.
        st.warning(
            f"**{base_release}** is newer than **{head_release}** — swap them to read "
            "the diff in the direction time runs.",
            icon="⚠️",
        )
        return

    span = _span(releases, base_release, head_release)
    base = _packages(data_url, platform, _PREFIX + base_release)
    head = _packages(data_url, platform, _PREFIX + head_release)
    missing = [r for r, p in ((base_release, base), (head_release, head)) if not p]
    if missing:
        st.warning(
            f"No stack provenance recorded for {', '.join(f'**{m}**' for m in missing)}. "
            "Releases benchmarked before provenance capture, or whose stack had already "
            "aged off CVMFS when the history was backfilled, cannot be diffed — but any "
            "regressions with onset in this range are still listed below.",
            icon="❔",
        )
    else:
        _render_diff(base, head, base_release, head_release, span)
    # Independent of the package diff: the reverse view is built from the
    # regression reports, so it renders even when provenance is unavailable. A
    # non-empty span means the *selected range* spans more than one release
    # step (see :func:`_span`) — the only case where the diff is cumulative.
    _render_regressions_in_range(
        data_url, platform, base_release, head_release, multi_release=bool(span),
    )


#: Cap on reverse-view rows so a month-wide range can't produce an unbounded
#: table; the largest by |Δ| are kept.
_MAX_REGRESSIONS = 30


def _regressions_in_range(
    reports, platform: str, older_release: str, newer_release: str
) -> list:
    """Confirmed regressions across *reports* (raw report dicts) whose onset
    falls in ``(older_release, newer_release]``, deduped per distinct onset
    *run* and sorted worst-first with unknown magnitude last.

    Pure — no Streamlit, no network — so the selection, dedup and ordering are
    testable on their own. Dedup keys on the onset run, not just its release:
    several runs share a release, and a re-anchored series can confirm two
    separate steps whose onset runs differ but whose onset releases match.
    """
    hits, seen = [], set()
    for raw in reports:
        if not raw:
            continue
        for v in from_json(raw).regressions:
            if v.platform != platform or not _blame.onset_in_range(v, older_release, newer_release):
                continue
            # A same-release window (onset == baseline) means the stack did not
            # move across the step, so it cannot be an effect of any diff — drop
            # it rather than list it with a nonsensical X → X window.
            if _blame.classify(v) is _blame.WindowKind.SAME_STACK:
                continue
            key = (v.detector, v.sample, v.label, v.metric, v.sub_detector, v.onset_run_id)
            if key not in seen:
                seen.add(key)
                hits.append(v)
    # Unknown relative magnitude (a zero baseline, or an absolute-floor metric)
    # sorts last rather than as if it were 0 %.
    hits.sort(key=lambda v: (v.pct_change is None, -abs(v.pct_change or 0.0)))
    return hits


def _render_regressions_in_range(
    data_url: str, platform: str, base_release: str, head_release: str,
    *, multi_release: bool,
) -> None:
    """Reverse attribution: the confirmed regressions whose onset falls in this
    release range — candidate effects of the diff above. Same-release regressions
    are excluded (no stack moved for them).

    *multi_release* is whether the *selected* range spans more than one release
    step: only then is the diff cumulative, so only then do the per-regression
    blame-window column and its warning appear. It is not derived per regression
    — a regression's baseline can predate the selected base, which would trip a
    per-window check even for two neighbouring releases.
    """
    dates = _cached_list_report_dates(data_url)
    # A regression confirms no earlier than its onset release, so only reports
    # on/after the older end can carry one whose onset is in range. Fetch each
    # night individually so it is cached once and reused as the range changes.
    reports = [_cached_fetch_report(data_url, d) for d in dates if d >= base_release]
    hits = _regressions_in_range(reports, platform, base_release, head_release)

    st.markdown("##### Regressions this change may have caused")
    st.caption(
        "Confirmed regressions whose step first appeared inside this release range — "
        "candidate effects of the changes above."
    )
    if not hits:
        st.info("No confirmed regression has its onset in this release range.", icon="✅")
        return

    shown = hits[:_MAX_REGRESSIONS]
    cumulative = multi_release
    if cumulative:
        st.warning(
            "This is a **multi-release** range, so its package diff above is "
            "**cumulative**. Each regression below was caused by the changes in its own "
            "**blame window** (shown) — *not* necessarily by every package that differs "
            "across the whole range. Compare consecutive releases to pin a single step.",
            icon="⚠️",
        )

    def _row(v):
        row = {
            "": "🔴",
            "Detector": v.detector,
            "Sample": _pretty_sample(v.sample),
            "Config": v.label,
            "Metric": _METRIC_LABELS.get(v.metric, v.metric)
                      + (f" · {v.sub_detector}" if v.sub_detector else ""),
            "Δ vs baseline": None if v.pct_change is None else v.pct_change * 100,
        }
        if cumulative:
            row["Blame window"] = (f"{v.last_accepted_run_date} → {v.onset_run_date}"
                                   if v.last_accepted_run_date else f"up to {v.onset_run_date}")
        return row

    column_config = {
        "": st.column_config.TextColumn("", width="small"),
        "Δ vs baseline": st.column_config.NumberColumn(
            "Δ vs baseline", format="%+.1f%%",
            help="Size and direction of the confirmed step, either way. Blank when the "
                 "metric has no meaningful relative change — an absolute-floor metric, "
                 "or a zero baseline.",
        ),
    }
    if cumulative:
        column_config["Blame window"] = st.column_config.TextColumn(
            "Blame window",
            help="The release range this step actually entered in (last accepted → "
                 "onset) — a sub-range of the wider diff above.",
        )
    st.dataframe(
        pd.DataFrame([_row(v) for v in shown]),
        hide_index=True, width="stretch", column_config=column_config,
    )
    if len(hits) > _MAX_REGRESSIONS:
        st.caption(f"Showing the {_MAX_REGRESSIONS} largest of {len(hits)} by |Δ|.")


def _span(releases: list[str], base_release: str, head_release: str) -> str:
    """How far apart the two releases are, as a sentence.

    A month-wide range is a *cumulative* diff, not one night's change — easy to
    misread when the table looks the same either way. Counting the releases in
    between says so; counting days would not, since the nightly skips days.
    """
    n_between = releases.index(base_release) - releases.index(head_release)
    if n_between == 1:
        return ""  # consecutive: the two tags in the heading already say it
    days = (pd.Timestamp(head_release) - pd.Timestamp(base_release)).days
    return (
        f"**{n_between} releases apart** ({_plural(days, 'day')}) — this is the "
        f"cumulative change across all {n_between}, not one night's."
    )


def _render_summary(
    n_changed: int, n_same: int, base_release: str, head_release: str, span: str
) -> None:
    """At-a-glance header, mirroring the Regressions tab's verdict banner."""
    with st.container(border=True):
        st.markdown(f"##### Stack diff — {base_release} → {head_release}")
        if span:
            st.caption(span)
        cols = st.columns(3)
        cols[0].metric(
            "Packages changed", n_changed,
            help="Packages whose upstream commit differs between the two releases. "
                 "These are the only places an upstream cause could come from.",
        )
        cols[1].metric(
            "Unchanged", n_same,
            help="Built from the identical commit in both releases.",
        )
        cols[2].metric(
            "Tracked", n_changed + n_same,
            help="Packages Key4hep builds from git, and whose commit is therefore "
                 "recorded. Release-tarball dependencies have no upstream commit "
                 "and are not tracked.",
        )


def _render_diff(
    base: dict, head: dict, base_release: str, head_release: str, span: str
) -> None:
    changes = diff_packages(base, head)
    same = unchanged_packages(base, head)
    _render_summary(len(changes), len(same), base_release, head_release, span)

    if not changes:
        # Not "nothing found" but "these are the same stack" — which rules an
        # upstream commit out entirely, and is an answer rather than a blank.
        st.success(
            f"**These two releases are the identical stack.** All {len(same)} tracked "
            "packages sit at the same commit, so nothing upstream changed between them: "
            "a metric that moved between these releases moved for another reason — the "
            "host, the sample, or noise.",
            icon="✅",
        )
        return

    df = pd.DataFrame([
        {
            "": _STATUS_BADGE[c.status],
            "Package": c.name,
            "From": (c.base_commit or "—")[:12],
            "To": (c.head_commit or "—")[:12],
            "Compare": c.compare_url,
        }
        for c in changes
    ])
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "": st.column_config.TextColumn(
                "", width="small",
                help="🔄 moved to a new commit · ➕ entered the stack · ➖ left it",
            ),
            "Package": st.column_config.TextColumn("Package", width="medium"),
            "From": st.column_config.TextColumn(
                "From", help="The package's commit in the older release.",
            ),
            "To": st.column_config.TextColumn(
                "To", help="The package's commit in the newer release.",
            ),
            "Compare": st.column_config.LinkColumn(
                "Compare", display_text="↗ commits", width="small",
                help="Every commit in this package's range, on its forge. Absent for a "
                     "package that was added or removed (there is no range to compare), "
                     "or hosted on a forge whose URL layout we do not know.",
            ),
        },
    )
    if same:
        st.caption(
            f"{_plural(len(same), 'tracked package')} unchanged between these releases."
        )
