"""Pure session-state guards for context-dependent widget defaults."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import ui_utils  # noqa: E402


def test_scope_change_clears_a_still_valid_old_widget_value(monkeypatch):
    state = {"picker": "still-valid", "_picker_scope": ("old",)}
    monkeypatch.setattr(ui_utils.st, "session_state", state)

    ui_utils.reset_widget_on_scope("picker", ("new",))

    assert state == {"_picker_scope": ("new",)}


def test_first_scope_preserves_a_preseeded_value(monkeypatch):
    state = {"picker": "deep-linked"}
    monkeypatch.setattr(ui_utils.st, "session_state", state)

    ui_utils.reset_widget_on_scope("picker", ("first",))

    assert state["picker"] == "deep-linked"


def test_automatic_widget_can_reset_unscoped_placeholder_state(monkeypatch):
    state = {"palette": "Matplotlib"}
    monkeypatch.setattr(ui_utils.st, "session_state", state)

    ui_utils.reset_widget_on_scope("palette", 1, reset_unscoped=True)

    assert state == {"_palette_scope": 1}


def test_scope_change_also_clears_the_widgets_query_param(monkeypatch):
    # A widget that writes its value back to the URL has to lose the parameter
    # too: leaving it would let seed_query_param put the old value straight back
    # on the same run, and the reset would be a no-op whenever that value is
    # still a valid option in the new scope.
    state = {"picker": "2026-07-09", "_picker_scope": ("old",)}
    params = {"report": "2026-07-09"}
    monkeypatch.setattr(ui_utils.st, "session_state", state)
    monkeypatch.setattr(ui_utils.st, "query_params", params)

    ui_utils.reset_widget_on_scope("picker", ("new",), query_param="report")

    assert state == {"_picker_scope": ("new",)}
    assert params == {}


def test_scope_change_sets_a_bound_widget_to_its_default(monkeypatch):
    # A widget bound to the URL is reset by its value: dropped, it would be
    # seeded straight back from its parameter, which st.query_params cannot drop.
    state = {"picker": ["old pick"], "_picker_scope": ("old",)}
    monkeypatch.setattr(ui_utils.st, "session_state", state)

    ui_utils.reset_widget_on_scope("picker", ("new",), default=["new default"])

    assert state == {"picker": ["new default"], "_picker_scope": ("new",)}


def test_first_scope_keeps_the_deep_linked_query_param(monkeypatch):
    state = {}
    params = {"report": "2026-07-09"}
    monkeypatch.setattr(ui_utils.st, "session_state", state)
    monkeypatch.setattr(ui_utils.st, "query_params", params)

    ui_utils.reset_widget_on_scope("picker", ("first",), query_param="report")

    assert params == {"report": "2026-07-09"}



# ── drop_stale_selection / render_scope_reset ───────────────────────────────

def test_drop_stale_selection_reports_the_value_it_dropped(monkeypatch):
    # The caller needs the old value to say *what* moved: switching detector can
    # take the selected sample with it, silently rescoping the Overview tab.
    import ui_chrome

    state = {"sb_sample": "single_e_10GeV"}
    monkeypatch.setattr(ui_chrome.st, "session_state", state)

    dropped = ui_chrome.drop_stale_selection("sb_sample", ["single_mu_10GeV"])

    assert dropped == "single_e_10GeV"
    assert "sb_sample" not in state


def test_drop_stale_selection_is_silent_when_the_value_survives(monkeypatch):
    import ui_chrome

    state = {"sb_sample": "single_e_10GeV"}
    monkeypatch.setattr(ui_chrome.st, "session_state", state)

    assert ui_chrome.drop_stale_selection(
        "sb_sample", ["single_e_10GeV", "single_mu_10GeV"]
    ) is None
    assert state == {"sb_sample": "single_e_10GeV"}
    # Nothing dropped and nothing to say.
    assert ui_chrome.drop_stale_selection("sb_missing", ["a"]) is None


def test_scope_reset_notice_only_fires_on_a_real_change(monkeypatch):
    import ui_chrome

    said: list[str] = []
    monkeypatch.setattr(ui_chrome.st, "caption", said.append)

    ui_chrome.render_scope_reset(None, "b", "isn't benchmarked for CLD")
    ui_chrome.render_scope_reset("b", "b", "isn't benchmarked for CLD")
    assert said == []

    ui_chrome.render_scope_reset("a", "b", "isn't benchmarked for CLD")
    assert said == ["⚠️ `a` isn't benchmarked for CLD — showing `b`."]


# ── Sidebar platform order ───────────────────────────────────────────────────
#
# The selectbox has no explicit index, so the first entry is what a fresh visit
# lands on. After a migration the plain alphabetical listing puts the *replaced*
# platform there.

import ui_chrome  # noqa: E402

_OLD = "x86_64-almalinux9-gcc14.2.0-opt"
_NEW = "x86_64-el9-gcc16-opt"


def test_a_replaced_platform_does_not_become_the_default():
    assert ui_chrome.order_platforms([_OLD, _NEW]) == [_NEW, _OLD]


def test_platforms_nothing_replaced_stay_alphabetical():
    assert ui_chrome.order_platforms(["b-plat", "a-plat"]) == ["a-plat", "b-plat"]


def test_a_replaced_detector_version_does_not_become_the_default(monkeypatch):
    # A fixed lineage: automated bumps rewrite the real one.
    from k4bench.regression import lineage

    monkeypatch.setattr(
        lineage,
        "DETECTOR_SUCCESSORS",
        {"ALLEGRO_o1_v04": "ALLEGRO_o1_v03", "CLD_o2_v09": "CLD_o2_v08"},
    )
    monkeypatch.setattr(lineage, "RETIRED_DETECTORS", {})
    assert ui_chrome.order_detectors(
        ["ALLEGRO_o1_v03", "ALLEGRO_o1_v04", "ALLEGRO_o2_v01", "CLD_o2_v08", "CLD_o2_v09"], "sim",
    ) == ["ALLEGRO_o1_v04", "ALLEGRO_o2_v01", "CLD_o2_v09", "ALLEGRO_o1_v03", "CLD_o2_v08"]
    # The reco page's detectors are never replaced by a sim bump.
    assert ui_chrome.order_detectors(["CLD_o2_v09", "CLD_o2_v08"], "reco") == [
        "CLD_o2_v08", "CLD_o2_v09",
    ]


# ── Selectbox width ──────────────────────────────────────────────────────────
#
# st.selectbox clips a selected option wider than its box. Chromium draws
# "Mean event anonymous RSS (GB)" 193 px wide, beside 50 px of padding and
# chevron; "Mean event file-backed RSS (GB)" has more characters but is narrower.

_ANON, _FILE = "Mean event anonymous RSS (GB)", "Mean event file-backed RSS (GB)"


def test_select_width_measures_in_streamlits_own_font():
    # Without it every box falls back to the wider per-character bound.
    assert ui_chrome._select_font() is not None


def test_select_width_measures_the_label_rather_than_counting_it():
    assert ui_chrome.select_width(_ANON) >= 50 + 193
    assert ui_chrome.select_width(_FILE) < ui_chrome.select_width(_ANON)


def test_select_width_without_the_font_still_fits_the_label(monkeypatch):
    measured = ui_chrome.select_width(_ANON)
    monkeypatch.setattr(ui_chrome, "_select_font", lambda: None)
    assert ui_chrome.select_width(_ANON) >= measured


def test_select_width_counts_an_emoji_badge_at_the_emoji_fonts_width():
    # Source Sans has no emoji; the browser draws them from the system's emoji
    # font, and the variation selector that makes ⚠ one takes no width.
    night = "2026-10-29"
    assert ui_chrome.select_width(f"🔴 {night}") >= ui_chrome.select_width(f" {night}") + 18
    assert ui_chrome.select_width(f"⚠️ {night}") == ui_chrome.select_width(f"🔴 {night}")


def test_current_option_follows_the_widgets_own_precedence(monkeypatch):
    options = ["a", "b", "c"]
    query = {"pick": "b"}
    monkeypatch.setattr(ui_chrome.st, "query_params", query)
    monkeypatch.setattr(ui_chrome.st, "session_state", {"pick": "c"})
    assert ui_chrome.current_option("pick", options) == "c"
    monkeypatch.setattr(ui_chrome.st, "session_state", {"pick": "gone"})
    assert ui_chrome.current_option("pick", options) == "b"
    query["pick"] = "stale"
    assert ui_chrome.current_option("pick", options) == "a"


def test_select_width_keeps_the_label_and_its_help_icon_on_one_line():
    # The label row is as wide as the box, so a long label widens a box whose
    # option is short.
    label = "Report night · release 2026-10-26"
    bare = ui_chrome.select_width("🔴 2026-10-28")
    labelled = ui_chrome.select_width("🔴 2026-10-28", label)
    assert labelled > bare
    assert ui_chrome.select_width("🔴 2026-10-28", label, help=True) == labelled + 24
