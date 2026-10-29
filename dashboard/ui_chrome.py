"""Page chrome and small sidebar utilities for the dashboard entry point.

The page configuration, footers, the stale-selection helper — presentation and
plumbing that would otherwise clutter :func:`suite_page.render_page`.
"""
from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable, Mapping
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import ImageFont

from k4bench.analysis.loader import failed_config_mask
from k4bench.labels import pretty_platform, pretty_release, pretty_sample
from k4bench.regression.lineage import is_detector_replaced, is_replaced
from sections import SCOPED, SectionScope


def order_platforms(platforms: list[str]) -> list[str]:
    """Platforms with replaced ones last, name-ascending within each group.

    The sidebar's selectbox takes its default from the first entry, and the
    listing it is built from is only alphabetical — which after a migration
    means the *replaced* platform, whose tree stopped growing and which the
    nightly report no longer covers. Replaced platforms stay selectable; their
    history is still worth reading, it is just not where a reader should land.
    """
    return sorted(platforms, key=lambda p: (is_replaced(p), p))


def order_detectors(detectors: list[str], suite: str) -> list[str]:
    """*suite*'s detector configs with replaced versions last, name-ascending
    within each group — the same reasoning as :func:`order_platforms`:
    alphabetically, a retired ``…_v03`` would otherwise be the default ahead of
    its ``…_v04``."""
    return sorted(detectors, key=lambda d: (is_detector_replaced(d, suite), d))


def failed_labels(results: "pd.DataFrame") -> list[str]:
    """Labels whose returncode is non-zero, missing, or invalid."""
    if "label" not in results.columns:
        return []
    return sorted(results.loc[failed_config_mask(results), "label"].astype(str))


def render_run_status(results: "pd.DataFrame | None", run_meta: dict | None) -> None:
    """Top-of-page alert banners for the selected run.

    The source of truth for failure is each config's ``returncode`` (loaded into
    *results*); ``run_meta`` (from ``run_info.json``) supplies the CI run link and
    the ``machine_consistent`` flag. Per-config detail + logs live in the Logs tab
    (:func:`tabs.logs.render`).
    """
    if results is None or "label" not in results.columns:
        return
    meta = run_meta or {}

    failed = failed_labels(results)
    if failed:
        run_url = meta.get("github_run_url")
        link = f" · [CI run logs]({run_url})" if run_url else ""
        st.error(
            f"**{len(failed)} of {len(results)} config(s) failed** in this run: "
            f"{', '.join(f'`{c}`' for c in failed)}. See the **Logs** tab.{link}"
        )
    if meta.get("machine_consistent") is False:
        machines = meta.get("machines") or []
        st.warning(
            "This sweep ran across **more than one machine**, so absolute timing "
            "comparisons between configs may be noisy: "
            + ", ".join(f"`{m}`" for m in machines)
        )


def render_scope_note(
    section: str,
    *,
    detector: str | None = None,
    platform: str | None = None,
    sample: str | None = None,
    release: str | None = None,
    run: str | None = None,
    data_dir: str | None = None,
    override: "SectionScope | None" = None,
    slot: "st.delta_generator.DeltaGenerator | None" = None,
    scopes: Mapping[str, SectionScope],
) -> None:
    """One muted line under the section bar naming what *section* is showing.

    Answers "what am I looking at?" without the reader having to reconstruct it
    from four sidebar selectboxes — while staying chrome rather than content:
    an ``st.caption``, so it takes the theme's small muted text colour, on one
    line, with no box, background or icon.

    What it may claim comes entirely from the page's *scopes*
    (:attr:`suite_page.SuitePage.scopes`), never from the other arguments:
    the sections honour genuinely different slices of the sidebar, so a
    uniform breadcrumb would misdescribe several of them. A section *scopes*
    does not know renders nothing rather than a guess.
    *override* replaces that declaration for the one section whose scope a
    control inside it can change (Stack Changes' "Whole platform" toggle); the
    caller supplies it, because the chrome must not read a tab's widget keys.

    *slot* is a placeholder reserved at the note's position on the page — pass
    one to render the note *after* the section's body has run, which is the only
    order in which a state that section owns can be read without being a frame
    behind it.

    The note carries the hierarchy only. Time references are each tab's own —
    see the registry for where every one of them is printed — and repeating
    them here would duplicate or, on the tabs that switch between the latest
    run and the trend window, contradict them.

    Each dimension is rendered through :mod:`k4bench.labels`, the module that
    owns these identifiers' layouts, so the note reads in the same vocabulary
    as the tabs below it and the nightly mail: a sample is named by its physics
    rather than its directory, and the release drops the prefix every tag
    shares. The dimensions are therefore joined with an em dash — the labels
    use ``·`` internally, and reusing it here would flatten "AlmaLinux 9 · GCC
    14.2.0" into two dimensions that look like four.

    The release is the software that was measured, and *run* the night it
    was measured; given one, a section that draws the sidebar's run (its
    scope's ``run``) is noted with both, never one for the other.

    In local mode the hierarchy does not exist (there is no EOS tree to walk,
    just one run directory), so every dimension arrives as ``None`` and the
    note names *data_dir* instead.
    """
    scope = override or scopes.get(section)
    if scope is None:
        return
    target = slot if slot is not None else st

    parts: list[str] = []
    scoped_any = False
    for declared, value, pretty in (
        # A detector name is already the name a person uses for it.
        (scope.detector, detector, str),
        (scope.platform, platform, pretty_platform),
        (scope.sample,   sample,   pretty_sample),
        (scope.release,  release,  lambda stack: (
            f"Key4hep {pretty_release(stack)}, measured {run}"
            if run and scope.run is SCOPED else pretty_release(stack)
        )),
    ):
        if declared is None:
            continue
        if isinstance(declared, str):
            parts.append(declared)
            continue
        # SCOPED: show the sidebar's selection, or nothing when it is unset —
        # a scoped dimension with no value is local mode, handled below.
        if value:
            parts.append(pretty(str(value)))
            scoped_any = True

    if not scoped_any:
        if data_dir:
            target.caption(f"Showing local run directory `{data_dir}`")
        return

    target.caption("Showing " + " — ".join(parts))


# ── Project links ────────────────────────────────────────────────────────────
# Surfaced natively via st.set_page_config(menu_items=...) in configure_page (the
# "☰" menu's Get Help / Report a bug / About entries) rather than as sidebar cards,
# which got crowded once WebEOS data, GitHub, and docs all competed for space.
GITHUB_URL = "https://github.com/key4hep/k4Bench"
DOCS_URL = "https://key4hep.github.io/k4Bench/"

# ── Page configuration ───────────────────────────────────────────────────────
#: Global CSS tweaks: header, padding, plot spacing, metric wrapping and the
#: sticky footer.
_GLOBAL_CSS = """
<style>
/* Start the content just below the header (3.75rem, where the page links
   sit) instead of below Streamlit's much taller default top margin. */
.block-container, .stMainBlockContainer {
    padding-top: 3.75rem !important;
    padding-bottom: 1rem !important;
}
/* Collapse the blank gap Streamlit inserts below plotly iframes */
[data-testid="stPlotlyChart"] > div,
.stPlotlyChart > div {
    line-height: 0;
}
/* Remove extra bottom margin on plotly chart wrappers */
[data-testid="stPlotlyChart"],
.stPlotlyChart {
    margin-bottom: 0 !important;
}
/* Tighten the gap between the last element and the footer */
footer { margin-top: 0 !important; padding-top: 0 !important; }

/* Let st.metric values wrap instead of truncating long strings with an
   ellipsis (e.g. a full OS name "AlmaLinux 9.7 (Seafoam Ocelot)" or a
   kernel version). Keeps every metric the same size while showing the
   value in full. */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] p {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    overflow-wrap: anywhere;
}

/* The reco Memory findings' names on one line, cut short with an ellipsis
   when they do not fit (the badge's tooltip names them in full), so the
   cards of a row line up. */
[class*="st-key-reco_memory_inspect_"] button { max-width: 100%; }
[class*="st-key-reco_memory_inspect_"] button p {
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 600;
}

/* Sticky footer: keep the copyright pinned to the bottom of the viewport
   when the page is short, but let it flow normally (and scroll) when the
   content is taller than the window. Achieved with a full-height flex
   column whose footer element gets margin-top:auto. */
.stMainBlockContainer {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}
/* Let the root vertical block (one or two wrapper levels deep) grow to
   fill the container so the auto margin below has space to consume. */
.stMainBlockContainer > div:first-child,
.stMainBlockContainer > div:first-child > [data-testid="stVerticalBlock"] {
    flex: 1 0 auto;
    display: flex;
    flex-direction: column;
}
/* Push the footer's element container to the bottom of that free space. */
[data-testid="stElementContainer"]:has(.k4-footer) {
    margin-top: auto;
}
</style>
"""


def configure_page() -> None:
    """Page title, icon, layout and menu, plus the global CSS. Must be the
    first Streamlit call of a run."""
    st.set_page_config(
        page_title="k4Bench Dashboard",
        page_icon="⚡",
        layout="wide",
        menu_items={
            "Get Help": DOCS_URL,
            "Report a bug": f"{GITHUB_URL}/issues",
            "About": f"[GitHub repository]({GITHUB_URL}) · [Documentation]({DOCS_URL})",
        },
    )
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ── Non-experiment ("example") detectors ────────────────────────────────────
# Detector names (the EOS top-level directory, derived from the compact XML's
# basename — see Job.detector in benchmark_job.py) that come from a simulation
# toolkit's own reference/tutorial geometry rather than a maintained FCC/Key4hep
# experiment design. Flagged here so the sidebar can make that distinction
# obvious instead of the name silently looking like "just another detector".
EXAMPLE_DETECTORS: dict[str, tuple[str, str]] = {
    # detector name -> (toolkit label, source URL)
    "SiD": ("DD4hep", "https://github.com/AIDASoft/DD4hep/tree/master/DDDetectors"),
}


def render_example_detector_badge(detector: str) -> None:
    """Sidebar note flagging *detector* as a toolkit example, not a real experiment.

    A no-op for anything not in :data:`EXAMPLE_DETECTORS`.
    """
    entry = EXAMPLE_DETECTORS.get(detector)
    if not entry:
        return
    toolkit, source_url = entry
    st.caption(
        f"🧪 **{detector}** is a {toolkit} reference/example detector, not a "
        f"maintained FCC/Key4hep experiment geometry. [Source ↗]({source_url})"
    )


def resource_link_card(href: str, icon_html: str, label: str, text: str) -> str:
    """Return the HTML for a styled sidebar link card (currently used by WebEOS data).

    ``icon_html`` may be an emoji or an inline SVG.
    """
    return f"""
        <a href="{href}" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
          <div style="
            background: rgba(91,155,213,0.08);
            border: 1px solid rgba(91,155,213,0.28);
            border-radius: 8px;
            padding: 0.45rem 0.75rem;
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.55rem;
            transition: background 0.2s;
          ">
            <span style="font-size:1.1rem;line-height:1;display:flex;">{icon_html}</span>
            <div style="overflow:hidden;">
              <div style="
                font-size:0.63rem;
                text-transform:uppercase;
                letter-spacing:0.07em;
                color:#7a9fbf;
                font-weight:600;
                margin-bottom:0.1rem;
              ">{label}</div>
              <div style="
                font-size:0.70rem;
                color:#5b9bd5;
                font-weight:500;
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
                max-width:180px;
              ">{text} ↗</div>
            </div>
          </div>
        </a>
    """


def render_footer() -> None:
    """Render a CERN / FCC copyright footer at the bottom of the page."""
    year = date.today().year
    st.markdown(
        f"""
        <hr class="k4-footer" style="border:none;border-top:1px solid rgba(128,128,128,0.25);margin:0rem 0 0rem 0;">
        <div style="
            display:flex;
            justify-content:center;
            align-items:center;
            gap:1.2rem;
            padding:0.2rem 0 1.2rem 0;
            font-size:0.80rem;
            color:#9a9a9a;
            line-height:1.7;
            text-align:center;
        ">
            <span style="font-size:1.8rem;opacity:0.75;">⚛️</span>
            <div>
                <strong style="color:#c0c0c0;letter-spacing:0.02em;">© {year} CERN</strong>
                &nbsp;·&nbsp;
                For the benefit of the&nbsp;<a
                    href="https://fcc.web.cern.ch/"
                    target="_blank"
                    rel="noopener noreferrer"
                    style="color:#5b9bd5;text-decoration:none;font-weight:600;"
                >FCC project</a>
                <br>
                Created by <strong style="color:#c0c0c0;">Joshua Falco Beirer</strong>
                &nbsp;<span style="opacity:0.6;">(CERN)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer() -> None:
    """Render a compact attribution note at the bottom of the sidebar."""
    year = date.today().year
    st.markdown(
        f"""
        <hr style="border:none;border-top:1px solid rgba(128,128,128,0.2);margin:1.5rem 0 0.6rem 0;">
        <div style="font-size:0.72rem;color:#888;text-align:center;line-height:1.6;padding-bottom:0.4rem;">
            <strong style="color:#a0a0a0;">© {year} CERN</strong><br>
            For the benefit of the<br>
            <a href="https://fcc.web.cern.ch/" target="_blank" rel="noopener noreferrer"
               style="color:#5b9bd5;text-decoration:none;">FCC project</a><br>
            <span style="opacity:0.7;">J. F. Beirer (CERN)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def drop_stale_selection(key: str, options: list[str]) -> str | None:
    """Clear a keyed selectbox's stored value when it's no longer a valid option,
    returning the value dropped (``None`` when nothing was).

    The dependent dropdowns (Platform → Sample → Stack) rebuild their option lists
    whenever an upstream selection changes, so a value left in ``session_state``
    from the old options can be fed back into ``st.selectbox`` as an invalid
    selection. Popping it *before* the widget is created (the only point at which
    a widget-backed key may be mutated) lets the selectbox re-default cleanly to a
    valid option. A no-op when the stored value is still present in *options*.

    The dropped value is returned because a silent re-default is a scope change
    the reader did not ask for: switching detector can take the selected sample
    with it, and the cross-detector Overview is scoped by sample, so its roster
    changes for a reason nothing on screen would otherwise state (see
    :func:`render_scope_reset` for the notice).
    """
    if key in st.session_state and st.session_state[key] not in options:
        return st.session_state.pop(key)
    return None


def render_scope_reset(dropped: str | None, current: str, reason: str) -> None:
    """Note that a sidebar scope selector re-defaulted, and why.

    Rendered right under the selector whose value moved, on the one rerun where
    it moved. Kept to a caption rather than a warning: nothing is wrong, the
    previous selection simply does not exist in the new scope — but the tabs
    below are scoped by it, so the change has to be visible.
    """
    if dropped and dropped != current:
        st.caption(f"⚠️ `{dropped}` {reason} — showing `{current}`.")


def seed_query_param(key: str, param: str, options: list[str]) -> str | None:
    """Seed a keyed widget's initial value from ``?param=...`` in the page
    URL, e.g. a nightly regression email's "view in dashboard" link (see
    ``k4bench.regression.render.dashboard_link``).

    Streamlit's documented pattern for this: write ``session_state[key]``
    before the widget is created, and never pass ``index=``/``default=`` for
    that same key, since Streamlit only honors the constructor argument the
    first time a key is created anyway and rejects a call that sets
    session_state *and* passes a default in the same run. A no-op once *key*
    already has a value (from an earlier rerun or a widget interaction), so
    it only ever applies on that one first run — call this immediately
    before creating the widget with ``key=key``.

    Returns the URL's value when it is not among *options* — a stale or
    mistyped link the widget opens without — so the caller can say so;
    ``None`` otherwise.
    """
    if key not in st.session_state:
        wanted = st.query_params.get(param)
        if wanted in options:
            st.session_state[key] = wanted
        elif wanted:
            return wanted
    return None


def current_option(key: str, options: list[str]) -> str:
    """The option the keyed selectbox *key* shows on this run, read before it
    is drawn: its stored value, else the ``?key=`` its ``bind="query-params"``
    opens on, else the first of *options* — the widget's own order of
    precedence, for a caller that sizes the box to what it shows."""
    for value in (st.session_state.get(key), st.query_params.get(key)):
        if value in options:
            return value
    return options[0]


#: A selectbox's text is Streamlit's bundled Source Sans at 14 px, regular
#: weight; its padding and chevron take 50 px beside the text.
_SELECT_FONT_PX = 14
_SELECT_FONT_WEIGHT = 400
_SELECT_CHROME_PX = 50
#: Upper bound of that font's width per character, for when it cannot be read:
#: one em, which its widest capitals, M and W, stay within.
_SELECT_PX_PER_CHAR = _SELECT_FONT_PX
#: A character the font lacks — an emoji badge — is drawn from the system's
#: emoji font instead, at most this many ems wide.
_SELECT_FALLBACK_EM = 1.3
#: Unicode categories drawn with no width of their own: the variation selector
#: that turns ⚠ into an emoji, joiners, combining marks.
_ZERO_WIDTH = frozenset({"Mn", "Me", "Cf"})
#: What a label's help icon takes beside its text: the 16 px icon and a gap.
_SELECT_HELP_PX = 24


@cache
def _select_font() -> ImageFont.FreeTypeFont | None:
    """The font Streamlit draws a selectbox's text in, read from the copy it
    serves, or ``None`` when this Streamlit no longer ships it there."""
    media = Path(st.__file__).parent / "static" / "static" / "media"
    try:
        font = ImageFont.truetype(
            str(next(media.glob("SourceSansVF-Upright*.woff2"))), _SELECT_FONT_PX,
        )
        font.set_variation_by_axes([_SELECT_FONT_WEIGHT])
    except (StopIteration, OSError):
        return None
    return font


@cache
def _in_select_font(char: str) -> bool:
    """Whether :func:`_select_font` has *char*; one it lacks comes out as its
    missing-glyph box, a private-use character's outline and advance."""
    font = _select_font()
    if font is None:
        return unicodedata.category(char) != "So"
    missing = "\U000F0000"
    return (font.getbbox(char), font.getlength(char)) != (
        font.getbbox(missing), font.getlength(missing),
    )


def _run_width(run: str) -> float:
    """The width of *run*, characters all in the font, measured whole so its
    kerning counts."""
    font = _select_font()
    return _SELECT_PX_PER_CHAR * len(run) if font is None else font.getlength(run)


def _text_width(text: str) -> float:
    """*text*'s width in px as a selectbox or its label draws it: in the font
    where it has the glyphs, an emoji at the emoji font's width."""
    width, run = 0.0, ""
    for char in text:
        if unicodedata.category(char) in _ZERO_WIDTH:
            continue
        if _in_select_font(char):
            run += char
        else:
            width += _run_width(run) + _SELECT_FALLBACK_EM * _SELECT_FONT_PX
            run = ""
    return width + _run_width(run)


def select_width(shown: str, label: str = "", *, help: bool = False) -> int:
    """A selectbox width, in px, that shows *shown* — its selected option —
    whole, and its *label* above on one line, with the help icon if it has one.

    ``st.selectbox`` takes either the parent's width or a pixel count; it clips
    a selected option wider than its box and wraps a label wider than it. Both
    are measured in the font the browser draws them in, so the box is no wider
    than it has to be; without the font, a per-character bound keeps them
    whole at the cost of some width.
    """
    return math.ceil(max(
        _SELECT_CHROME_PX + _text_width(shown),
        _text_width(label) + (_SELECT_HELP_PX if help else 0),
    ))


def fitted_selectbox(
    label: str,
    options: list[str],
    *,
    key: str,
    format_func: Callable[[str], str] = str,
    help: str | None = None,
    **kwargs: Any,
) -> str | None:
    """``st.selectbox`` sized by :func:`select_width` to the option it shows
    and its label. A keyed selectbox keeps its value when its width changes,
    so the box follows its selection from one run to the next (see
    :func:`current_option`). Its open list is as wide as the box, so an option
    longer than the one shown is cut short there."""
    shown = format_func(current_option(key, options))
    return st.selectbox(
        label, options, key=key, format_func=format_func, help=help,
        width=select_width(shown, label, help=help is not None), **kwargs,
    )
