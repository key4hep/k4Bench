"""Integration test: real ddsim simulation against ALLEGRO_o1_v03.

Requires the Key4hep environment with ddsim available and $K4GEO set.
Run with: pytest -m integration
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from dd4bench.benchmark.ddsim import BenchmarkConfig, SweepMode, run_sweep


K4GEO = os.environ.get("K4GEO", "")
ALLEGRO_XML = (
    Path(K4GEO) / "FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml"
    if K4GEO
    else None
)

_missing_env = not K4GEO or ALLEGRO_XML is None or not ALLEGRO_XML.exists()


@pytest.fixture(scope="module")
def simulation_result():
    """Run a real 100-event ddsim simulation and return the single RunResult."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config = BenchmarkConfig(
            xml_path=ALLEGRO_XML,
            n_events=100,
            output_file=tmp_path / "out.edm4hep.root",
            log_dir=tmp_path / "logs",
            mode=SweepMode.BASELINE,
            extra_args=[
                "--enableGun",
                "--gun.distribution", "uniform",
                "--gun.energy", "10*GeV",
                "--gun.particle", "e-",
                "--random.enableEventSeed",
                "--random.seed", "42",
            ],
            verbose=True,
        )
        results = run_sweep(config)
        assert len(results) == 1
        yield results[0]


@pytest.mark.integration
@pytest.mark.skipif(_missing_env, reason="$K4GEO not set or ALLEGRO XML not found")
def test_simulation_exits_zero(simulation_result):
    """ddsim must exit cleanly."""
    assert simulation_result.succeeded, (
        f"ddsim failed with returncode={simulation_result.returncode}"
    )


@pytest.mark.integration
@pytest.mark.skipif(_missing_env, reason="$K4GEO not set or ALLEGRO XML not found")
def test_simulation_event_count(simulation_result):
    """RunResult records the correct event count."""
    assert simulation_result.n_events == 100


@pytest.mark.integration
@pytest.mark.skipif(_missing_env, reason="$K4GEO not set or ALLEGRO XML not found")
def test_simulation_wall_time_positive(simulation_result):
    """/usr/bin/time wall-clock time was parsed and is positive."""
    assert simulation_result.wall_time_s is not None
    assert simulation_result.wall_time_s > 0


@pytest.mark.integration
@pytest.mark.skipif(_missing_env, reason="$K4GEO not set or ALLEGRO XML not found")
def test_simulation_cpu_times_positive(simulation_result):
    """User and system CPU times were parsed and are non-negative."""
    assert simulation_result.user_cpu_s is not None
    assert simulation_result.user_cpu_s >= 0
    assert simulation_result.sys_cpu_s is not None
    assert simulation_result.sys_cpu_s >= 0


@pytest.mark.integration
@pytest.mark.skipif(_missing_env, reason="$K4GEO not set or ALLEGRO XML not found")
def test_simulation_peak_rss_reasonable(simulation_result):
    """Peak RSS is present and plausibly large (>100 MB) for a full detector."""
    assert simulation_result.peak_rss_mb is not None
    assert simulation_result.peak_rss_mb > 100, (
        f"Peak RSS {simulation_result.peak_rss_mb:.1f} MB looks too small for ALLEGRO"
    )


@pytest.mark.integration
@pytest.mark.skipif(_missing_env, reason="$K4GEO not set or ALLEGRO XML not found")
def test_simulation_output_file_nonempty(simulation_result):
    """The EDM4hep ROOT output file was written and is non-empty."""
    assert simulation_result.output_size_mb is not None
    assert simulation_result.output_size_mb > 0, (
        f"Output file size is {simulation_result.output_size_mb} MB — expected >0"
    )


@pytest.mark.integration
@pytest.mark.skipif(_missing_env, reason="$K4GEO not set or ALLEGRO XML not found")
def test_simulation_events_per_sec_positive(simulation_result):
    """Throughput metric is computed and strictly positive."""
    assert simulation_result.events_per_sec is not None
    assert simulation_result.events_per_sec > 0


@pytest.mark.integration
@pytest.mark.skipif(_missing_env, reason="$K4GEO not set or ALLEGRO XML not found")
def test_simulation_cpu_efficiency_reasonable(simulation_result):
    """CPU efficiency (CPU/wall) is between 0 and a few cores worth."""
    eff = simulation_result.cpu_efficiency
    assert eff is not None
    assert 0 < eff < 64, f"CPU efficiency {eff:.2f} is outside the expected range"
