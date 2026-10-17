"""The Logs tab reads each suite's logs in their own format: Geant4's for
ddsim, Gaudi's and its components' for k4run."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from tabs import logs  # noqa: E402

RECO_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "reco_run"

#: A failed k4run job: Marlin errors that do not stop it, then the one that does.
_FAILED_K4RUN = [
    "ApplicationMgr     SUCCESS",
    'Evaluator        WARN  +++ Overwriting variable [setVariable error]',
    '[ ERROR "MyConformalTracking"]  could not add inner hit to track !!!',
    '[ WARNING "MyDDMarlinPandora_10ns"] track state at calorimeter is not set',
    "Refit                FATAL Standard std::exception is caught",
    "Refit                ERROR basic_string::_M_create",
    '[ ERROR "MyConformalTracking"]  could not add inner hit to track !!!',
    "EventLoopMgr        ERROR Execution of algorithm k4FWCore__Sequencer failed",
    "ApplicationMgr       INFO Application Manager Terminated with error code 1",
    "Command exited with non-zero status 1",
    "\tCommand being timed: \"k4run CLDReconstruction.py\"",
    "\tExit status: 1",
]


def test_gaudi_levels_count_and_prose_does_not():
    errors = logs._filter_by_severity(_FAILED_K4RUN, True, False, logs.GAUDI)
    warnings = logs._filter_by_severity(_FAILED_K4RUN, False, True, logs.GAUDI)
    assert '[ ERROR "MyConformalTracking"]  could not add inner hit to track !!!' in errors
    assert "Refit                FATAL Standard std::exception is caught" in errors
    # "error" in a WARN line's prose is a warning, not an error.
    assert not [ln for ln in errors if ln.startswith("Evaluator")]
    assert [ln for ln in warnings if ln.startswith("Evaluator")]
    assert '[ WARNING "MyDDMarlinPandora_10ns"] track state at calorimeter is not set' in warnings
    assert logs._GAUDI_ERROR_RE.search("Error in <TFile::TFile>: file does not exist")
    assert logs._GAUDI_ERROR_RE.search("[k4run - ERROR] k4run.main: no such option")


def test_geant4_logs_still_count_the_word_error():
    # The Geant4 format, every page's default, keeps its broad match.
    assert logs._filter_by_severity(_FAILED_K4RUN[:2], True, False) == [_FAILED_K4RUN[1]]


def test_a_gaudi_failure_is_explained_by_its_first_fatal_line_and_its_component():
    excerpt = logs._extract_error_excerpt(_FAILED_K4RUN, log_format=logs.GAUDI)
    # The FATAL line, not the last Marlin error after it.
    assert excerpt.splitlines()[0] == "ApplicationMgr     SUCCESS"
    assert "Refit                FATAL" in excerpt
    assert logs._cause_source(excerpt, logs.GAUDI) == "Refit"
    # The Geant4 format centres on the last error line instead.
    assert logs._cause_source(logs._extract_error_excerpt(_FAILED_K4RUN), logs.GEANT4) is None


def test_messages_are_counted_by_the_component_that_logged_them():
    sources = logs._sources(logs._log_body(_FAILED_K4RUN), logs.GAUDI)
    assert sources.to_dict("records") == [
        {"Component": "MyConformalTracking", "Errors": 2, "Warnings": 0},
        {"Component": "Refit", "Errors": 2, "Warnings": 0},
        {"Component": "EventLoopMgr", "Errors": 1, "Warnings": 0},
        {"Component": "Evaluator", "Errors": 0, "Warnings": 1},
        {"Component": "MyDDMarlinPandora_10ns", "Errors": 0, "Warnings": 1},
    ]


def _logs_app(dashboard_dir, results_csv, run_dir, gaudi) -> None:
    import sys as _sys
    if dashboard_dir not in _sys.path:
        _sys.path.insert(0, dashboard_dir)
    import pandas as _pd

    from tabs import logs as _logs

    _logs.render(
        _pd.read_csv(results_csv), run_dir, {},
        log_format=_logs.GAUDI if gaudi else _logs.GEANT4,
    )


def _render(run_dir: Path, gaudi: bool) -> AppTest:
    results = run_dir / "all_results.csv"
    pd.concat(
        [pd.read_csv(p) for p in sorted(run_dir.glob("*_results.csv"))], ignore_index=True,
    ).to_csv(results, index=False)
    at = AppTest.from_function(
        _logs_app, args=(str(_DASHBOARD_DIR), str(results), str(run_dir), gaudi),
        default_timeout=60,
    )
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.fixture
def failed_reco_run(tmp_path: Path) -> Path:
    """The 10-17 run, its variant failed on the job above."""
    run = tmp_path / "run"
    shutil.copytree(RECO_RUN, run)
    results = run / "variant_truth_tracking_results.csv"
    frame = pd.read_csv(results)
    frame["returncode"] = 1
    frame.to_csv(results, index=False)
    (run / "variant_truth_tracking.log").write_text("\n".join(_FAILED_K4RUN) + "\n")
    return run


def test_a_failed_k4run_job_names_the_component_that_ended_it(failed_reco_run):
    at = _render(failed_reco_run, gaudi=True)
    # The failed configuration opens first.
    assert "raised by `Refit`" in at.error[1].value
    assert "Refit                FATAL" in at.code[0].value
    assert not [m for m in at.markdown if "plugin messages" in m.value]
    assert [m for m in at.markdown if "Errors and warnings by component" in m.value]
    assert at.text_input(key="logs_joboptions_filter")


def test_job_options_are_offered_and_filterable(failed_reco_run):
    at = _render(failed_reco_run, gaudi=True)
    at.text_input(key="logs_joboptions_filter").set_value("truthtrack").run()
    # Case-insensitive, and only the matching lines: the algorithm's own
    # property and the sequence that runs it.
    shown = at.code[-1].value.splitlines()
    assert [line.split(" =")[0] for line in shown] == [
        "MyTruthTrackFinder.OutputLevel", "k4FWCore__Algs.Members",
    ]


def test_a_geant4_log_keeps_its_plugin_messages_and_offers_no_job_options(failed_reco_run):
    at = _render(failed_reco_run, gaudi=False)
    assert [m for m in at.markdown if "plugin messages" in m.value]
    assert not [m for m in at.markdown if "Errors and warnings by component" in m.value]
    assert "raised by" not in at.error[1].value
    with pytest.raises(KeyError):
        at.text_input(key="logs_joboptions_filter")
