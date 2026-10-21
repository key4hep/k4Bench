"""The page URL's vocabulary for a run and its history (:mod:`url_params`):
what a link names, and which values it refuses to read as one."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import url_params  # noqa: E402
from k4bench.layout import RECO, RunRef  # noqa: E402

REF = RunRef(RECO, "CLD_o2_v09", "x86_64-el9-gcc16-opt", "key4hep-2026-10-17",
             "p8_ee_Zbb_ecm91", "2026-10-21")


def test_a_run_is_named_by_every_level_of_its_path():
    assert url_params.run_params(REF) == {
        "detector": "CLD_o2_v09", "platform": "x86_64-el9-gcc16-opt",
        "sample": "p8_ee_Zbb_ecm91", "stack": "key4hep-2026-10-17", "run": "2026-10-21",
    }


@pytest.mark.parametrize(("value", "valid"), [
    ("2026-10-21", True),
    ("2026-02-30", False),
    ("2026-10-21 · latest", False),
    ("latest", False),
    ("", False),
    (None, False),
])
def test_a_run_id_is_a_real_date(value, valid):
    assert url_params.is_run_id(value) is valid


def test_a_release_range_round_trips():
    value = url_params.release_range("key4hep-2026-09-22", "key4hep-2026-10-17")
    assert value == "key4hep-2026-09-22..key4hep-2026-10-17"
    assert url_params.parse_release_range(value) == ("key4hep-2026-09-22", "key4hep-2026-10-17")


@pytest.mark.parametrize("value", [
    None, "", "key4hep-2026-09-22", "..key4hep-2026-10-17", "key4hep-2026-09-22..",
    "a..b..c",
])
def test_a_malformed_release_range_names_no_window(value):
    assert url_params.parse_release_range(value) is None


def test_the_history_modes_default_to_the_latest_run():
    assert list(url_params.HISTORY_MODES)[0] == url_params.HISTORY_LATEST
    assert set(url_params.HISTORY_MODES) == {"latest", "all"}


def test_every_parameter_has_its_own_name():
    names = [
        url_params.TAB, url_params.DETECTOR, url_params.PLATFORM, url_params.SAMPLE,
        url_params.STACK, url_params.RUN, url_params.RELEASES, url_params.HISTORY,
        url_params.TIMING, url_params.COMPONENT,
    ]
    assert len(set(names)) == len(names)
    # Every one is documented in the module's table.
    assert all(f"``{name}``" in url_params.__doc__ for name in names)
