"""The dashboard's pages: one per suite, each a :class:`suite_page.SuitePage`.

The registries each page is built from must agree with each other, the router
must open Simulation at ``/`` (where the nightly mail's ``?tab=`` deep links
point), every page must keep its section and sidebar state under its own keys,
and no page may present another suite's run as its own.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import sim_page  # noqa: E402
from reco import page as reco_page  # noqa: E402

PAGES = [sim_page.SIM_PAGE, reco_page.RECO_PAGE]

#: A nightly CLD reconstruction run (see its README).
RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"
RECO_LABELS = ["baseline", "variant_truth_tracking"]

_RESULTS_HEADER = (
    "label,returncode,n_events,wall_time_raw,wall_time_s,user_cpu_s,sys_cpu_s,"
    "peak_rss_mb,peak_vmem_mb,major_page_faults,voluntary_ctx_switches,"
    "involuntary_ctx_switches,output_size_mb,events_per_sec"
)


@pytest.fixture
def sim_run(tmp_path: Path) -> Path:
    """A local ``k4bench ddsim`` run directory with one successful ``baseline``."""
    (tmp_path / "baseline_results.csv").write_text(
        _RESULTS_HEADER + "\nbaseline,0,10,0:50.00,50.0,49.0,1.0,1400,3400,0,10,1,7.6,0.2\n"
    )
    (tmp_path / "baseline.log").write_text("Generating event 0\n")
    return tmp_path


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.suite)
def test_every_section_has_a_renderer_and_a_scope(page):
    assert set(page.renderers) == set(page.sections)
    assert set(page.scopes) == set(page.sections)
    assert page.remote_only <= set(page.sections)


def test_pages_walk_their_own_suite_under_their_own_keys():
    assert [(p.suite, p.title) for p in PAGES] == [
        ("sim", "Simulation"), ("reco", "Reconstruction"),
    ]
    assert len({p.key_prefix for p in PAGES}) == len(PAGES)
    # Simulation keeps the unprefixed keys its tests and links were written for.
    assert sim_page.SIM_PAGE.key_prefix == ""


def test_reco_page_has_no_report_backed_section():
    # Nothing judges the reco suite, so its trend window must not wait for reports.
    assert reco_page.RECO_PAGE.reports is False
    assert not {"Regressions", "Run Trends", "Stack Changes"} & set(reco_page.RECO_PAGE.sections)
    # Its Overview is its own, drawn from the runs rather than the nightly reports.
    assert reco_page.RECO_PAGE.renderers["Overview"] is not sim_page.SIM_PAGE.renderers["Overview"]


def test_only_the_reco_page_loads_component_costs():
    assert reco_page.RECO_PAGE.components is True
    assert sim_page.SIM_PAGE.components is False


def _local_env(monkeypatch, run_dir: Path, cache_dir: Path) -> None:
    monkeypatch.delenv("K4BENCH_DATA_URL", raising=False)
    monkeypatch.setenv("K4BENCH_DATA_DIR", str(run_dir))
    monkeypatch.setenv("K4BENCH_CACHE_DIR", str(cache_dir))


def _router(tab: str) -> AppTest:
    at = AppTest.from_file(str(_DASHBOARD_DIR / "app.py"), default_timeout=60)
    at.query_params["tab"] = tab
    at.run()
    assert not at.exception, at.exception
    return at


def test_the_router_opens_simulation_and_follows_its_deep_link(monkeypatch, sim_run, tmp_path):
    _local_env(monkeypatch, sim_run, tmp_path / "cache")
    at = _router("logs")
    assert at.session_state["active_section"] == "Logs"
    assert at.button_group[0].options == list(
        sim_page.SIM_PAGE.visible_sections(trends_enabled=False)
    )
    assert "reco_active_section" not in at.session_state


def test_simulation_refuses_a_local_reco_run(monkeypatch, tmp_path):
    _local_env(monkeypatch, RECO_RUN, tmp_path / "cache")
    at = _router("Event Timing")
    assert "holds a reconstruction run" in at.warning[0].value
    assert "Reconstruction page" in at.warning[0].value
    assert not at.button_group
    assert not at.get("plotly_chart")


def _reco_app(dashboard_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st

    from reco import page

    # "away" stands for the reader being on another page: nothing of this one
    # is drawn, and the page switch cleared the URL.
    if st.session_state.get("away"):
        st.query_params.clear()
    else:
        page.main()


def _reco(tab: str) -> AppTest:
    at = AppTest.from_function(_reco_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.query_params["tab"] = tab
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize(("tab", "series"), [
    # The baseline's whole event beside its slowest algorithm.
    ("Event Timing", {"Whole event", "MyConformalTracking"}),
    # The baseline against the variant, from their own event files.
    ("Event Memory", set(RECO_LABELS)),
])
def test_reco_page_draws_the_runs_per_event_views(monkeypatch, tmp_path, tab, series):
    _local_env(monkeypatch, RECO_RUN, tmp_path / "cache")
    at = _reco(tab)
    assert at.session_state["reco_active_section"] == tab
    charts = at.get("plotly_chart")
    assert charts
    traces = json.loads(charts[0].proto.spec)["data"]
    assert {t.get("name") for t in traces} >= series


@pytest.mark.parametrize("tab", ["Machine Info", "Logs"])
def test_reco_page_draws_its_run_wide_sections_under_its_own_keys(monkeypatch, tmp_path, tab):
    _local_env(monkeypatch, RECO_RUN, tmp_path / "cache")
    at = _reco(tab)
    assert at.session_state["reco_active_section"] == tab
    assert "active_section" not in at.session_state
    assert at.button_group[0].options == reco_page.RECO_PAGE.visible_sections(trends_enabled=False)
    assert at.text_input(key="reco_data_dir").value == str(RECO_RUN)
    assert not [w for w in at.warning if "holds a" in w.value]


def test_reco_page_returns_to_the_local_directory_a_reader_left(monkeypatch, tmp_path):
    # Not the default directory, so a reset cannot pass for a restore.
    other = tmp_path / "other_reco_run"
    shutil.copytree(RECO_RUN, other)
    _local_env(monkeypatch, RECO_RUN, tmp_path / "cache")
    at = _reco("Logs")
    at.text_input(key="reco_data_dir").set_value(str(other)).run()
    at.session_state["away"] = True
    at.run()
    at.session_state["away"] = False
    at.run()
    assert not at.exception, at.exception
    assert at.text_input(key="reco_data_dir").value == str(other)


def test_reconstruction_refuses_a_local_sim_run(monkeypatch, sim_run, tmp_path):
    _local_env(monkeypatch, sim_run, tmp_path / "cache")
    at = _reco("Event Timing")
    assert "holds a simulation run" in at.warning[0].value
    assert "Simulation page" in at.warning[0].value
    assert not at.button_group


def _section_app(dashboard_dir, names) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import streamlit as st

    import section_dispatch

    # "away" stands for the reader being on another page: the bar is not
    # drawn, and the page switch cleared the URL it mirrors itself in.
    if st.session_state.get("away"):
        st.query_params.clear()
    else:
        st.session_state["drawn"] = section_dispatch.pick_section(names, key="page_section")


def test_the_section_bar_reopens_the_section_a_reader_left():
    names = list(reco_page.RECO_PAGE.sections)
    at = AppTest.from_function(_section_app, args=(str(_DASHBOARD_DIR), names), default_timeout=30)
    at.run()
    at.button_group[0].set_value("Logs").run()
    at.session_state["away"] = True
    at.run()
    at.session_state["away"] = False
    at.run()
    assert at.session_state["drawn"] == "Logs"
