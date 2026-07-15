"""Tests for the Stack Changes tab.

Covers the release-vs-release contract the tab rests on (never run dates), the
identical-stack case that is the tab's most useful answer, and the app-level
registration — including the remote-only section slice, which is an off-by-one
away from exposing a tab that cannot work in local mode.

All remote calls are stubbed; nothing touches the network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"

PLAT = "x86_64-almalinux9-gcc14.2.0-opt"


def _load_module():
    if str(_DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(_DASHBOARD_DIR))
    spec = importlib.util.spec_from_file_location(
        "k4bench_dashboard_stack_changes", _DASHBOARD_DIR / "tabs" / "stack_changes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stack_changes = _load_module()


def _pkg(commit: str, url: str = "https://github.com/key4hep/k4geo.git") -> dict:
    return {"commit": commit, "version": "develop", "repo_url": url}


#: 07-08 and 07-09 are the identical stack (the nightly-lag case); k4geo moves
#: in 07-10, so the tab's default view (the two newest) shows a real diff.
_STACKS = {
    "key4hep-2026-07-08": {"k4geo": _pkg("a" * 40), "dd4hep": _pkg("b" * 40)},
    "key4hep-2026-07-09": {"k4geo": _pkg("a" * 40), "dd4hep": _pkg("b" * 40)},
    "key4hep-2026-07-10": {"k4geo": _pkg("c" * 40), "dd4hep": _pkg("b" * 40)},
}


def _app(dashboard_dir, stack_names, packages, from_release, to_release):
    """The tab, rendered standalone with every remote call stubbed.

    ``AppTest.from_function`` re-executes this source in its own script
    context, so it can close over nothing: the imports, the stubs, and the
    platform literal all have to live inside it. The stubs are set on the
    ``tabs.stack_changes`` module the script itself imports — patching any
    other instance of it would not be seen from here.
    """
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as _st

    from tabs import stack_changes as _tab

    _tab._cached_list_detectors = lambda url: ["IDEA"]
    _tab._cached_list_stacks = lambda url, detector, platform: stack_names
    _tab._cached_fetch_stack_packages = (
        lambda url, detector, platform, stack: packages.get(stack)
    )

    if from_release:
        _st.query_params["from"] = from_release
    if to_release:
        _st.query_params["to"] = to_release
    _tab.render("https://example.invalid", "x86_64-almalinux9-gcc14.2.0-opt")


def _run(stack_names=None, packages=None, from_release=None, to_release=None) -> AppTest:
    at = AppTest.from_function(
        _app,
        args=(
            str(_DASHBOARD_DIR),
            sorted(_STACKS, reverse=True) if stack_names is None else stack_names,
            _STACKS if packages is None else packages,
            from_release,
            to_release,
        ),
        default_timeout=30,
    )
    at.run()
    assert not at.exception, at.exception
    return at


# ── the release-vs-release contract ──────────────────────────────────────────

def test_release_strips_the_directory_prefix():
    # The tab talks in nightly tags; EOS stores them as key4hep-{date} dirs.
    assert stack_changes._release("key4hep-2026-07-10") == "2026-07-10"
    assert stack_changes._release("2026-07-10") == "2026-07-10"


def test_stacks_are_unioned_across_detectors(monkeypatch):
    # Detectors join and leave the matrix, so no single detector's history is
    # the full set of releases.
    per_detector = {"IDEA": ["key4hep-2026-07-10"], "SiD": ["key4hep-2026-07-09"]}
    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA", "SiD"])
    monkeypatch.setattr(
        stack_changes, "_cached_list_stacks",
        lambda url, detector, platform: per_detector[detector],
    )
    assert stack_changes._stacks_for_platform("u", PLAT) == [
        "key4hep-2026-07-10", "key4hep-2026-07-09",
    ]


def test_a_detector_without_the_platform_is_skipped(monkeypatch):
    def _stacks(url, detector, platform):
        if detector == "SiD":
            raise RuntimeError("no such platform")
        return ["key4hep-2026-07-10"]

    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA", "SiD"])
    monkeypatch.setattr(stack_changes, "_cached_list_stacks", _stacks)
    # One detector missing a platform must not blank the whole tab.
    assert stack_changes._stacks_for_platform("u", PLAT) == ["key4hep-2026-07-10"]


def test_packages_fall_back_to_another_detector(monkeypatch):
    # A detector may have skipped a release, or run it before provenance
    # capture; any other detector's run answers for the same stack.
    def _fetch(url, detector, platform, stack):
        return _STACKS[stack] if detector == "SiD" else None

    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA", "SiD"])
    monkeypatch.setattr(stack_changes, "_cached_fetch_stack_packages", _fetch)
    assert stack_changes._packages("u", PLAT, "key4hep-2026-07-10") == _STACKS["key4hep-2026-07-10"]


def test_packages_none_when_no_detector_has_it(monkeypatch):
    monkeypatch.setattr(stack_changes, "_cached_list_detectors", lambda url: ["IDEA"])
    monkeypatch.setattr(
        stack_changes, "_cached_fetch_stack_packages", lambda *a: None,
    )
    # Unknown must never read as "an empty stack".
    assert stack_changes._packages("u", PLAT, "key4hep-2026-07-10") is None


# ── how far apart the releases are ───────────────────────────────────────────

def test_consecutive_releases_say_nothing():
    # There is nothing to warn about, and a line restating the heading is noise.
    releases = ["2026-07-10", "2026-07-09", "2026-07-08"]
    assert stack_changes._span(releases, "2026-07-09", "2026-07-10") == ""


def test_a_wide_range_warns_that_the_diff_is_cumulative():
    # A month-wide diff looks identical to one night's in the table; without
    # this it reads as "these 21 packages changed last night".
    releases = [f"2026-07-{d:02d}" for d in range(10, 0, -1)]
    span = stack_changes._span(releases, "2026-07-01", "2026-07-10")
    assert "9 releases apart" in span and "cumulative" in span


# ── render ───────────────────────────────────────────────────────────────────

def test_defaults_to_the_two_newest_releases():
    # "What came in last night?" — defaulting both pickers to the newest would
    # open the tab on "pick two different releases" instead of an answer.
    at = _run()
    assert [s.value for s in at.selectbox] == ["2026-07-09", "2026-07-10"]
    assert at.dataframe, "the default view should show the diff, not a prompt"


def test_renders_the_diff_between_two_releases():
    at = _run(from_release="2026-07-09", to_release="2026-07-10")
    rendered = at.dataframe[0].value

    assert len(rendered) == 1, "only the moved package belongs here"
    row = rendered.iloc[0]
    # Identifiers are plain text; the compare view is the row's one action, and
    # it spans both commits so nothing is lost by being the only link.
    assert row["Package"] == "k4geo"
    assert row["From"] == "a" * 12
    assert row["To"] == "c" * 12
    assert row["Compare"] == (
        f"https://github.com/key4hep/k4geo/compare/{'a' * 40}...{'c' * 40}"
    )


def test_the_diff_reports_how_far_apart_the_releases_are():
    at = _run(from_release="2026-07-08", to_release="2026-07-10")
    captions = " ".join(c.value for c in at.caption)
    assert "2 releases apart" in captions and "cumulative" in captions


def test_the_branch_column_is_not_rendered():
    # Every package in every release sits on `develop`, so a branch column
    # would be one repeated value taking space from the SHAs.
    at = _run(from_release="2026-07-09", to_release="2026-07-10")
    assert "Branch" not in at.dataframe[0].value.columns


def test_packages_on_an_unknown_forge_still_render():
    # No compare view exists for a forge whose URL layout we do not know; the
    # package and its commits are still the answer to what moved.
    stacks = {
        "key4hep-2026-07-09": {"odd": {"commit": "a" * 40, "version": "develop",
                                       "repo_url": "https://git.example.com/a/b"}},
        "key4hep-2026-07-10": {"odd": {"commit": "c" * 40, "version": "develop",
                                       "repo_url": "https://git.example.com/a/b"}},
    }
    at = _run(stack_names=sorted(stacks, reverse=True), packages=stacks,
              from_release="2026-07-09", to_release="2026-07-10")
    row = at.dataframe[0].value.iloc[0]
    assert row["Package"] == "odd"
    assert row["From"] == "a" * 12
    assert row["Compare"] is None


def test_identical_releases_are_called_out_not_left_empty():
    # The nightly-lag case is the tab's most valuable answer: it rules an
    # upstream commit out entirely, rather than showing an empty table.
    at = _run(from_release="2026-07-08", to_release="2026-07-09")
    body = " ".join(s.value for s in at.success)
    assert "identical stack" in body and "nothing upstream changed" in body
    assert not at.dataframe


def test_reversed_range_is_refused_not_sign_flipped():
    at = _run(from_release="2026-07-10", to_release="2026-07-08")
    assert any("swap them" in w.value for w in at.warning)
    assert not at.dataframe


def test_needs_two_releases_to_compare():
    at = _run(stack_names=["key4hep-2026-07-10"])
    assert any("at least two" in i.value for i in at.info)


def test_missing_provenance_is_reported_not_diffed():
    # Releases benchmarked before provenance capture, or whose stack had aged
    # off CVMFS by backfill time, cannot be compared — say so rather than
    # diffing against nothing.
    at = _run(packages={})
    assert any("No stack provenance" in w.value for w in at.warning)
    assert not at.dataframe


# ── app registration ─────────────────────────────────────────────────────────

def _sections():
    """The section registry.

    Imported from ``sections.py`` rather than ``app.py`` on purpose: app.py
    ends in a bare ``main()``, so importing it would run the whole dashboard —
    and, if ``K4BENCH_DATA_URL`` happens to be set, fetch over the network from
    inside a unit test.
    """
    if str(_DASHBOARD_DIR) not in sys.path:
        sys.path.insert(0, str(_DASHBOARD_DIR))
    spec = importlib.util.spec_from_file_location(
        "k4bench_dashboard_sections", _DASHBOARD_DIR / "sections.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_stack_changes_is_registered_and_remote_only():
    sections = _sections()
    # It compares two releases off EOS, so it cannot work without a data_url.
    assert "Stack Changes" in sections.SECTION_NAMES
    assert "Stack Changes" in sections.REMOTE_ONLY


def test_every_remote_only_section_is_a_real_section():
    sections = _sections()
    # A typo would silently fail to hide a section rather than erroring.
    assert sections.REMOTE_ONLY <= set(sections.SECTION_NAMES)


def test_local_mode_keeps_exactly_the_sections_that_work_without_a_data_url():
    sections = _sections()
    assert sections.visible_sections(trends_enabled=False) == [
        "Region Timing", "Event Timing", "Event Memory", "Machine Info", "Logs",
    ]


def test_remote_mode_keeps_every_section_in_display_order():
    sections = _sections()
    assert sections.visible_sections(trends_enabled=True) == sections.SECTION_NAMES


def test_section_order_is_independent_of_data_requirements(monkeypatch):
    """Reordering the bar must not change which sections are hidden.

    Order is a presentation choice and remote-only is a fact about data
    sources; deriving one from the other would let a reorder strand a tab with
    nothing behind it.
    """
    sections = _sections()
    monkeypatch.setattr(sections, "SECTION_NAMES", list(reversed(sections.SECTION_NAMES)))
    assert set(sections.visible_sections(trends_enabled=False)) == {
        "Region Timing", "Event Timing", "Event Memory", "Machine Info", "Logs",
    }
