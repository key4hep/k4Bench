"""The dashboard's pages: one per suite, each a :class:`suite_page.SuitePage`.

The registries each page is built from must agree with each other, the router
must open Simulation at ``/`` (where the nightly mail's ``?tab=`` deep links
point), and every page must keep its section and sidebar state under its own
keys.
"""

from __future__ import annotations

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

_RESULTS_HEADER = (
    "label,returncode,n_events,wall_time_raw,wall_time_s,user_cpu_s,sys_cpu_s,"
    "peak_rss_mb,peak_vmem_mb,major_page_faults,voluntary_ctx_switches,"
    "involuntary_ctx_switches,output_size_mb,events_per_sec"
)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A local run directory with one successful ``baseline`` config."""
    (tmp_path / "baseline_results.csv").write_text(
        _RESULTS_HEADER + "\nbaseline,0,10,0:50.00,50.0,49.0,1.0,1400,3400,0,10,1,7.6,0.2\n"
    )
    (tmp_path / "baseline.log").write_text("ApplicationMgr       INFO Application Manager Terminated successfully\n")
    return tmp_path


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.suite)
def test_every_section_has_a_renderer_and_a_scope(page):
    assert set(page.renderers) == set(page.sections)
    assert set(page.scopes) == set(page.sections)
    assert page.remote_only <= set(page.sections)


def test_pages_walk_their_own_suite_under_their_own_keys():
    assert [p.suite for p in PAGES] == ["sim", "reco"]
    assert len({p.key_prefix for p in PAGES}) == len(PAGES)
    # Simulation keeps the unprefixed keys its tests and links were written for.
    assert sim_page.SIM_PAGE.key_prefix == ""


def test_reco_page_has_no_report_backed_section():
    # Nothing judges the reco suite, so its trend window must not wait for reports.
    assert reco_page.RECO_PAGE.reports is False
    assert not {"Overview", "Regressions", "Run Trends", "Stack Changes"} & set(
        reco_page.RECO_PAGE.sections
    )


def _local_env(monkeypatch, run_dir: Path) -> None:
    monkeypatch.delenv("K4BENCH_DATA_URL", raising=False)
    monkeypatch.setenv("K4BENCH_DATA_DIR", str(run_dir))
    monkeypatch.setenv("K4BENCH_CACHE_DIR", str(run_dir / "cache"))


def test_the_router_opens_simulation_and_follows_its_deep_link(monkeypatch, run_dir):
    _local_env(monkeypatch, run_dir)
    at = AppTest.from_file(str(_DASHBOARD_DIR / "app.py"), default_timeout=60)
    at.query_params["tab"] = "logs"
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["active_section"] == "Logs"
    assert at.button_group[0].options == list(
        sim_page.SIM_PAGE.visible_sections(trends_enabled=False)
    )
    assert "reco_active_section" not in at.session_state


def _reco_app(dashboard_dir) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    from reco import page

    page.main()


def test_reco_page_draws_its_shared_sections_under_its_own_keys(monkeypatch, run_dir):
    _local_env(monkeypatch, run_dir)
    at = AppTest.from_function(_reco_app, args=(str(_DASHBOARD_DIR),), default_timeout=60)
    at.query_params["tab"] = "Machine Info"
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["reco_active_section"] == "Machine Info"
    assert "active_section" not in at.session_state
    assert at.button_group[0].options == list(reco_page.RECO_PAGE.sections)
    assert at.text_input(key="reco_data_dir").value == str(run_dir)
