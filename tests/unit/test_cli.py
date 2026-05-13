"""Unit tests for dd4bench.cli.

Tests cover argument parsing and config building only — no ddsim is run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dd4bench.benchmark.ddsim import SweepMode
from dd4bench.cli import _build_config, _build_parser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PARSER = _build_parser()
XML_A = Path("geometry_a.xml")
XML_B = Path("geometry_b.xml")


def _parse(args: list[str]):
    return PARSER.parse_args(args)


def _config(args: list[str]):
    return _build_config(_parse(args))


# ---------------------------------------------------------------------------
# Geometry argument (--xml / --compare are mutually exclusive)
# ---------------------------------------------------------------------------


class TestGeometryArgs:
    def test_xml_required_without_compare(self):
        with pytest.raises(SystemExit):
            _parse([])

    def test_xml_and_compare_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            _parse(["--xml", str(XML_A), "--compare", str(XML_A), str(XML_B)])

    def test_xml_sets_xml_path(self):
        config = _config(["--xml", str(XML_A)])
        assert config.xml_path == XML_A

    def test_compare_sets_both_paths(self):
        config = _config(["--compare", str(XML_A), str(XML_B)])
        assert config.xml_path == XML_A
        assert config.xml_path_b == XML_B


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------


class TestSweepMode:
    def test_default_mode_is_baseline(self):
        config = _config(["--xml", str(XML_A)])
        assert config.mode == SweepMode.BASELINE

    def test_sweep_flag_sets_full_mode(self):
        config = _config(["--xml", str(XML_A), "--sweep"])
        assert config.mode == SweepMode.FULL

    def test_include_only_sets_mode(self):
        config = _config(["--xml", str(XML_A), "--include-only", "EcalBarrel"])
        assert config.mode == SweepMode.INCLUDE_ONLY

    def test_exclude_only_sets_mode(self):
        config = _config(["--xml", str(XML_A), "--exclude-only", "EcalBarrel"])
        assert config.mode == SweepMode.EXCLUDE_ONLY

    def test_compare_sets_compare_mode(self):
        config = _config(["--compare", str(XML_A), str(XML_B)])
        assert config.mode == SweepMode.COMPARE

    def test_include_only_and_exclude_only_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            _parse([
                "--xml", str(XML_A),
                "--include-only", "EcalBarrel",
                "--exclude-only", "HcalBarrel",
            ])

    def test_include_only_detector_names(self):
        config = _config([
            "--xml", str(XML_A),
            "--include-only", "EcalBarrel", "HcalBarrel",
        ])
        assert config.detector_names == ["EcalBarrel", "HcalBarrel"]

    def test_exclude_only_detector_names(self):
        config = _config([
            "--xml", str(XML_A),
            "--exclude-only", "InnerTracker", "OuterTracker",
        ])
        assert config.detector_names == ["InnerTracker", "OuterTracker"]


# ---------------------------------------------------------------------------
# ddsim-args parsing
# ---------------------------------------------------------------------------


class TestDdsimArgs:
    def test_empty_ddsim_args_gives_empty_list(self):
        config = _config(["--xml", str(XML_A)])
        assert config.extra_args == []

    def test_ddsim_args_split_correctly(self):
        config = _config([
            "--xml", str(XML_A),
            "--ddsim-args=--runType=batch --enableGun",
        ])
        assert config.extra_args == ["--runType=batch", "--enableGun"]

    def test_ddsim_args_with_quoted_value(self):
        config = _config([
            "--xml", str(XML_A),
            "--ddsim-args=--gun.particle e-",
        ])
        assert "--gun.particle" in config.extra_args
        assert "e-" in config.extra_args

    def test_ddsim_args_present_in_config(self):
        config = _config([
            "--xml", str(XML_A),
            "--ddsim-args=--runType=batch",
        ])
        assert "--runType=batch" in config.extra_args


# ---------------------------------------------------------------------------
# Output options
# ---------------------------------------------------------------------------


class TestOutputOptions:
    def test_default_output_dir(self):
        config = _config(["--xml", str(XML_A)])
        assert config.log_dir == Path("logs")

    def test_custom_output_dir(self):
        config = _config(["--xml", str(XML_A), "--output-dir", "/tmp/bench"])
        assert config.log_dir == Path("/tmp/bench")

    def test_default_output_file(self):
        config = _config(["--xml", str(XML_A)])
        assert config.output_file == Path("/tmp/dd4bench_out.edm4hep.root")

    def test_custom_output_file(self):
        config = _config([
            "--xml", str(XML_A),
            "--output-file", "/tmp/custom.root",
        ])
        assert config.output_file == Path("/tmp/custom.root")

    def test_default_events(self):
        config = _config(["--xml", str(XML_A)])
        assert config.n_events == 2

    def test_custom_events(self):
        config = _config(["--xml", str(XML_A), "--events", "10"])
        assert config.n_events == 10


# ---------------------------------------------------------------------------
# CSV / pickle args (parsed but not executed here)
# ---------------------------------------------------------------------------


class TestCsvPickleArgs:
    def test_default_csv(self):
        args = _parse(["--xml", str(XML_A)])
        assert args.csv == "results.csv"

    def test_custom_csv(self):
        args = _parse(["--xml", str(XML_A), "--csv", "my_results.csv"])
        assert args.csv == "my_results.csv"

    def test_pickle_default_is_none(self):
        args = _parse(["--xml", str(XML_A)])
        assert args.pickle is None

    def test_pickle_custom(self):
        args = _parse(["--xml", str(XML_A), "--pickle", "results.pkl"])
        assert args.pickle == "results.pkl"
