"""Which slice of the sidebar a section honours, in the words every page's
scope note uses.

A page declares, per section, the scope it shows *on entry*
(:attr:`suite_page.SuitePage.scopes`), out of the named scopes below.
:func:`ui_chrome.render_scope_note` renders it as the single muted line under
the section bar. Several sections carry a control that changes their scope — a
historical sub-view, a "Whole platform" toggle — and a static registry cannot
see those, so each such ``render()`` returns the scope it actually rendered
(one of the overrides below) and the page hands that to the note. Only the
section itself knows which branch it took.

No scope declares a *time* reference, on purpose: every section that has one
already prints it in its own words and in the right place, and a second copy in
the chrome line would either duplicate it or — on the tabs whose sub-views
switch between the latest run and the trend window — contradict it. Run Trends
and Machine Info print a "Data range"; Overview prints "Latest night · Data
range" and, off the newest night, "Historical view · report night";
Regressions prints the night on its picker and blame cards; Logs prints the
run's release, date and commit. The sidebar itself captions the trend window.
So the note carries the hierarchy and nothing else.

:data:`SHARED_SCOPES` holds the scopes of the sections every page that has them
draws alike (:data:`section_dispatch.SHARED_RENDERERS`).
"""

from __future__ import annotations

from dataclasses import dataclass

from k4bench.labels import pretty_release


class _Scoped:
    """Sentinel for "this section follows the sidebar here"; see :data:`SCOPED`."""

    __slots__ = ()

    def __repr__(self) -> str:  # keeps registry diffs and test failures readable
        return "SCOPED"


#: Marks a dimension the section is genuinely scoped by, so the scope note
#: prints the sidebar's current value for it.
SCOPED = _Scoped()


@dataclass(frozen=True)
class SectionScope:
    """What one section does with each of the four sidebar dimensions.

    Every field takes one of three kinds of value:

    * :data:`SCOPED` — the section is scoped by the sidebar's selection, and
      the note shows that selection.
    * a string — the section deliberately ignores the sidebar here, and the
      note says so in those words ("all detectors") rather than dropping the
      dimension, which would read as if the sidebar still applied.
    * ``None`` — the dimension is meaningless for this section, so the note
      omits it entirely.

    *run* is :data:`SCOPED` for a section that draws the sidebar's run itself,
    whose note then names the night it was measured beside its release; any
    other section shows a night of its own choosing, or none.
    """

    detector: "_Scoped | str | None"
    platform: "_Scoped | str | None"
    sample:   "_Scoped | str | None"
    release:  "_Scoped | str | None"
    run:      "_Scoped | None" = None


#: The Overview's: cross-detector by construction, it compares every detector
#: benchmarked for the selected platform/sample on a nightly report, and reports
#: are per-night rather than per-release, so the sidebar release does not apply.
ACROSS_DETECTORS = SectionScope(
    detector="all detectors", platform=SCOPED, sample=SCOPED, release=None,
)

#: The history of the sidebar triple across every release the trend window
#: covers — the sidebar release picks the single-run sections' run only. The
#: window is named, not dated: the run history is not unbounded, and the
#: sidebar's own caption carries the dates. Run Trends' scope on entry, and what
#: a historical sub-view of a single-run section returns, so the note stops
#: naming a release the plot below it does not single out.
TREND_WINDOW_SCOPE = SectionScope(
    detector=SCOPED, platform=SCOPED, sample=SCOPED,
    release="all releases in the trend window",
)

#: Regressions': the release selects which report nights are on offer and the
#: section's picker then chooses one of them, so the sidebar's run is not what
#: it shows. When the night on screen was benchmarked by a different release
#: (the fallback when EOS cannot tie any night to the sidebar's), the section
#: says which by returning :func:`regressions_derived_release`.
ONE_RELEASE = SectionScope(
    detector=SCOPED, platform=SCOPED, sample=SCOPED, release=SCOPED,
)

#: Stack Changes': two halves with different scopes. A Key4hep release is one
#: stack whatever benchmarked it, so the package diff is scoped by the platform
#: alone — said in the release dimension, because that pair is what produces
#: the diff. The regressions-in-range view below it *is* scoped to the
#: sidebar's detector and sample, until its "Whole platform" toggle widens it;
#: see :data:`STACK_CHANGES_PLATFORM_WIDE`.
RELEASE_PAIR = SectionScope(
    detector=SCOPED,
    platform=SCOPED,
    sample=SCOPED,
    release="platform-wide package diff for the release pair chosen below",
)

#: The sections built on the sidebar's own selection: all four dimensions come
#: straight from it, and they draw its run. Those with a historical sub-view
#: widen across the trend window's releases, which they report by returning
#: :data:`TREND_WINDOW_SCOPE`.
ONE_RUN = SectionScope(
    detector=SCOPED, platform=SCOPED, sample=SCOPED, release=SCOPED, run=SCOPED,
)

#: The scope of each section every page that has it draws alike.
SHARED_SCOPES: dict[str, SectionScope] = {
    "Overview": ACROSS_DETECTORS,
    "Regressions": ONE_RELEASE,
    "Stack Changes": RELEASE_PAIR,
    "Event Size": ONE_RUN,
    "Machine Info": ONE_RUN,
}


def regressions_derived_release(release: str) -> SectionScope:
    """Regressions when the night on screen belongs to a release other than the
    sidebar's: the note names the release the verdicts and attribution actually
    describe, since the sidebar's is not what was rendered."""
    return SectionScope(
        detector=SCOPED, platform=SCOPED, sample=SCOPED,
        release=f"{pretty_release(release)} — the selected report night's release",
    )

#: Stack Changes while its reverse view's "Whole platform" toggle is on: the
#: regressions listed there then span every detector and sample benchmarked on
#: the platform, so the two sidebar dimensions the section otherwise honours
#: stop applying. Returned only when that view actually rendered — the tab exits
#: early on a range it cannot compare, and the toggle's remembered state would
#: otherwise outlive the view it describes.
STACK_CHANGES_PLATFORM_WIDE = SectionScope(
    detector="all detectors",
    platform=SCOPED,
    sample="all samples",
    release=RELEASE_PAIR.release,
)
