"""Unit tests for k4bench.runner.parser.

All tests are pure: no subprocesses, no filesystem access, no ddsim.
The fixture file (tests/fixtures/time_output.txt) is a realistic capture
of /usr/bin/time -v output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from k4bench.runner.parser import _wall_to_seconds, parse_time_output

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _fixture_text(filename: str) -> str:
    return (FIXTURES / filename).read_text()


# ---------------------------------------------------------------------------
# _wall_to_seconds — unit tests
# ---------------------------------------------------------------------------

class TestWallToSeconds:
    def test_minutes_and_seconds(self):
        assert _wall_to_seconds("0:16.23") == pytest.approx(16.23)

    def test_minutes_non_zero(self):
        assert _wall_to_seconds("2:05.50") == pytest.approx(125.50)

    def test_hours_minutes_seconds(self):
        assert _wall_to_seconds("1:02:45.10") == pytest.approx(3765.10)

    def test_hours_zero(self):
        assert _wall_to_seconds("0:00:00.00") == pytest.approx(0.0)

    def test_invalid_returns_none(self):
        assert _wall_to_seconds("not-a-time") is None

    def test_empty_string_returns_none(self):
        assert _wall_to_seconds("") is None

    def test_single_segment_returns_none(self):
        # Only one segment — not a valid wall-clock string.
        assert _wall_to_seconds("123.45") is None


# ---------------------------------------------------------------------------
# parse_time_output — happy path with real fixture
# ---------------------------------------------------------------------------

class TestParseTimeOutputHappyPath:
    """Parse the realistic fixture and assert exact expected values."""

    @pytest.fixture(scope="class")
    def metrics(self):
        return parse_time_output(_fixture_text("time_output.txt"))

    def test_wall_time_raw(self, metrics):
        assert metrics["wall_time_raw"] == "0:16.23"

    def test_wall_time_s(self, metrics):
        assert metrics["wall_time_s"] == pytest.approx(16.23)

    def test_user_cpu_s(self, metrics):
        assert metrics["user_cpu_s"] == pytest.approx(47.32)

    def test_sys_cpu_s(self, metrics):
        assert metrics["sys_cpu_s"] == pytest.approx(3.18)

    def test_peak_rss_mb(self, metrics):
        # 2 145 320 kB ÷ 1024 = 2095.039… MB
        assert metrics["peak_rss_mb"] == pytest.approx(2145320 / 1024)

    def test_major_page_faults(self, metrics):
        assert metrics["major_page_faults"] == 4

    def test_voluntary_ctx_switches(self, metrics):
        assert metrics["voluntary_ctx_switches"] == 28341

    def test_involuntary_ctx_switches(self, metrics):
        assert metrics["involuntary_ctx_switches"] == 9812


# ---------------------------------------------------------------------------
# parse_time_output — edge cases
# ---------------------------------------------------------------------------

class TestParseTimeOutputEdgeCases:
    def test_empty_string_returns_all_none(self):
        metrics = parse_time_output("")
        assert all(v is None for v in metrics.values())

    def test_returns_all_expected_keys(self):
        metrics = parse_time_output("")
        expected_keys = {
            "wall_time_raw",
            "wall_time_s",
            "user_cpu_s",
            "sys_cpu_s",
            "peak_rss_mb",
            "major_page_faults",
            "voluntary_ctx_switches",
            "involuntary_ctx_switches",
        }
        assert set(metrics.keys()) == expected_keys

    def test_partial_output_populates_present_fields(self):
        partial = (
            "\tUser time (seconds): 12.34\n"
            "\tSystem time (seconds): 0.56\n"
        )
        metrics = parse_time_output(partial)
        assert metrics["user_cpu_s"] == pytest.approx(12.34)
        assert metrics["sys_cpu_s"] == pytest.approx(0.56)
        assert metrics["wall_time_raw"] is None
        assert metrics["peak_rss_mb"] is None

    def test_non_numeric_rss_yields_none(self):
        bad = "\tMaximum resident set size (kbytes): UNKNOWN\n"
        metrics = parse_time_output(bad)
        assert metrics["peak_rss_mb"] is None

    def test_hours_wall_time(self):
        long_run = (
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:45.10\n"
        )
        metrics = parse_time_output(long_run)
        assert metrics["wall_time_raw"] == "1:02:45.10"
        assert metrics["wall_time_s"] == pytest.approx(3765.10)

    def test_involuntary_not_confused_with_voluntary(self):
        """Involuntary line must not clobber voluntary count."""
        both = (
            "\tVoluntary context switches: 100\n"
            "\tInvoluntary context switches: 200\n"
        )
        metrics = parse_time_output(both)
        assert metrics["voluntary_ctx_switches"] == 100
        assert metrics["involuntary_ctx_switches"] == 200

    def test_extra_unknown_lines_are_ignored(self):
        """Lines not matching any known key must not raise exceptions."""
        noise = (
            "\tSome Future Metric: 9999\n"
            "\tUser time (seconds): 5.00\n"
        )
        metrics = parse_time_output(noise)
        assert metrics["user_cpu_s"] == pytest.approx(5.00)
