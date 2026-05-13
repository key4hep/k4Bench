"""Integration test: real ddsim simulation via the dd4bench CLI.

Invokes the ``dd4bench`` entry point as a subprocess against ALLEGRO_o1_v03
and verifies that the process exits cleanly and that the CSV output contains
sensible benchmark metrics.

Requires the Key4hep environment with ddsim available and $K4GEO set.
Run with: pytest -m integration
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


K4GEO = os.environ.get("K4GEO", "")
ALLEGRO_XML = (
    Path(K4GEO) / "FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml"
    if K4GEO
    else None
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not K4GEO
        or not (ALLEGRO_XML is not None and ALLEGRO_XML.exists())
        or shutil.which("ddsim") is None
        or shutil.which("dd4bench") is None,
        reason="$K4GEO not set, ALLEGRO XML not found, or ddsim/dd4bench not in PATH",
    ),
]

_N_EVENTS = 100


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cli_run():
    """Run dd4bench against ALLEGRO_o1_v03; yield (CompletedProcess, csv_row)."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        result = subprocess.run(
            [
                "dd4bench",
                "--verbose",
                "--events", str(_N_EVENTS),
                "--xml", str(ALLEGRO_XML),
                "--output-dir", str(output_dir),
                "--ddsim-args",
                (
                    "--enableGun "
                    "--gun.distribution uniform "
                    "--gun.energy '10*GeV' "
                    "--gun.particle e- "
                    "--random.enableEventSeed "
                    "--random.seed 42"
                ),
            ],
            capture_output=True,
            text=True,
        )

        row: dict[str, str] = {}
        csv_path = output_dir / "results.csv"
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                row = rows[0]

        yield result, row


@pytest.fixture(scope="module")
def cli_result(cli_run):
    result, _ = cli_run
    return result


@pytest.fixture(scope="module")
def csv_row(cli_run):
    _, row = cli_run
    return row


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cli_exits_zero(cli_result):
    """dd4bench must exit with code 0."""
    assert cli_result.returncode == 0, (
        f"dd4bench exited {cli_result.returncode}\n"
        f"stdout:\n{cli_result.stdout}\n"
        f"stderr:\n{cli_result.stderr}"
    )


def test_csv_written(csv_row):
    """A non-empty CSV row must be produced."""
    assert csv_row, "CSV file is missing or empty"


def test_csv_event_count(csv_row):
    """CSV records the correct event count."""
    assert int(csv_row["n_events"]) == _N_EVENTS


def test_csv_returncode_zero(csv_row):
    """ddsim return code captured in CSV is 0."""
    assert int(csv_row["returncode"]) == 0


def test_csv_wall_time_positive(csv_row):
    """Wall-clock time was parsed and is positive."""
    assert float(csv_row["wall_time_s"]) > 0


def test_csv_cpu_times_non_negative(csv_row):
    """User and system CPU times are non-negative."""
    assert float(csv_row["user_cpu_s"]) >= 0
    assert float(csv_row["sys_cpu_s"]) >= 0


def test_csv_peak_rss_reasonable(csv_row):
    """Peak RSS is >100 MB — a plausible lower bound for a full ALLEGRO load."""
    rss = float(csv_row["peak_rss_mb"])
    assert rss > 100, f"Peak RSS {rss:.1f} MB looks too small for ALLEGRO"


def test_csv_output_file_nonempty(csv_row):
    """The EDM4hep ROOT output file was written and has non-zero size."""
    assert float(csv_row["output_size_mb"]) > 0


def test_csv_events_per_sec_positive(csv_row):
    """Throughput metric is positive."""
    assert float(csv_row["events_per_sec"]) > 0
