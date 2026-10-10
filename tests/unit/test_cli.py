"""Unit tests for k4bench.cli.

Tests cover argument parsing and config building only — no ddsim is run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import k4bench.benchmark.ddsim as ddsim_module
from k4bench.benchmark.ddsim import SweepMode
from k4bench.cli import _build_config, _build_parser, main
from k4bench.results.model import RunResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PARSER = _build_parser()
XML_A = Path("geometry_a.xml")
MINIMAL_XML = Path(__file__).parent.parent / "fixtures" / "minimal_geometry" / "minimal.xml"


def _parse(args: list[str]):
    return PARSER.parse_args(args)


def _config(args: list[str]):
    return _build_config(_parse(args))


# ---------------------------------------------------------------------------
# Geometry argument
# ---------------------------------------------------------------------------


class TestGeometryArgs:
    def test_xml_required(self):
        with pytest.raises(SystemExit):
            _parse([])

    def test_xml_sets_xml_path(self):
        config = _config(["--xml", str(XML_A)])
        assert config.xml_path == XML_A


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

    def test_sweep_detectors_sets_full_mode_with_names(self):
        config = _config([
            "--xml", str(XML_A),
            "--sweep-detectors", "EcalBarrel", "HcalBarrel",
        ])
        assert config.mode == SweepMode.FULL
        assert config.detector_names == ["EcalBarrel", "HcalBarrel"]

    def test_sweep_and_sweep_detectors_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            _parse([
                "--xml", str(XML_A),
                "--sweep",
                "--sweep-detectors", "EcalBarrel",
            ])

    def test_include_only_sets_mode(self):
        config = _config(["--xml", str(XML_A), "--include-only", "EcalBarrel"])
        assert config.mode == SweepMode.INCLUDE_ONLY

    def test_exclude_only_sets_mode(self):
        config = _config(["--xml", str(XML_A), "--exclude-only", "EcalBarrel"])
        assert config.mode == SweepMode.EXCLUDE_ONLY

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
        # default is derived in main(), not _build_config(); arg stays None here
        args = _parse(["--xml", str(XML_A)])
        assert args.output_dir is None

    def test_custom_output_dir(self):
        config = _config(["--xml", str(XML_A), "--output-dir", "/tmp/bench"])
        assert config.log_dir == Path("/tmp/bench")

    def test_default_output_file(self):
        config = _config(["--xml", str(XML_A)])
        assert config.output_file == Path("/tmp/k4bench_out.edm4hep.root")

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
# Pickle args (parsed but not executed here)
# ---------------------------------------------------------------------------


class TestPickleArgs:
    def test_pickle_default_is_none(self):
        args = _parse(["--xml", str(XML_A)])
        assert args.pickle is None

    def test_pickle_custom(self):
        args = _parse(["--xml", str(XML_A), "--pickle", "results.pkl"])
        assert args.pickle == "results.pkl"


# ---------------------------------------------------------------------------
# --list-detectors
# ---------------------------------------------------------------------------


class TestListDetectorsArg:
    def test_default_is_false(self):
        args = _parse(["--xml", str(XML_A)])
        assert args.list_detectors is False

    def test_flag_sets_true(self):
        args = _parse(["--xml", str(XML_A), "--list-detectors"])
        assert args.list_detectors is True

    def test_does_not_require_sweep_mode(self):
        # --list-detectors needs only --xml; no ddsim-args required either.
        args = _parse(["--xml", str(XML_A), "--list-detectors"])
        assert args.list_detectors is True


class TestListDetectorsMain:
    def test_prints_detector_names_and_returns_zero(self, capsys):
        rc = main(["--xml", str(MINIMAL_XML), "--list-detectors"])
        out = capsys.readouterr().out
        assert rc == 0
        assert set(out.split()) == {
            "InnerTracker", "OuterTracker", "EcalBarrel", "HcalBarrel",
        }

    def test_no_simulation_is_run(self, capsys):
        # A real run would need ddsim on PATH; reaching rc == 0 here proves
        # main() returned before invoking run_sweep().
        rc = main(["--xml", str(MINIMAL_XML), "--list-detectors", "--sweep"])
        assert rc == 0

    def test_empty_geometry_returns_one(self, tmp_path, capsys):
        xml = tmp_path / "empty.xml"
        xml.write_text('<?xml version="1.0"?><lccdd></lccdd>')
        rc = main(["--xml", str(xml), "--list-detectors"])
        assert rc == 1
        assert "No subdetectors found" in capsys.readouterr().err


def test_strict_index_failure_makes_full_sweep_exit_nonzero(
    tmp_path,
    monkeypatch,
):
    class BrokenIndex:
        @classmethod
        def load(cls, path, *, strict):
            raise RuntimeError("strict indexing failed")

    def successful_baseline(**kwargs):
        return RunResult(
            label=kwargs["label"],
            returncode=0,
            n_events=2,
        )

    monkeypatch.setattr(ddsim_module, "GeometryIndex", BrokenIndex)
    monkeypatch.setattr(ddsim_module, "run_ddsim", successful_baseline)

    rc = main(
        [
            "--xml",
            str(MINIMAL_XML),
            "--sweep",
            "--output-dir",
            str(tmp_path / "results"),
        ]
    )

    assert rc == 1


# ---------------------------------------------------------------------------
# k4bench k4run
# ---------------------------------------------------------------------------


class TestK4runCommand:
    """``k4bench k4run`` builds a K4runConfig; the benchmark itself is mocked."""

    @pytest.fixture
    def stage(self, tmp_path):
        (tmp_path / "Reco.py").write_text("")
        return tmp_path

    def _main(self, monkeypatch, *argv: str) -> tuple[int, list]:
        import k4bench.cli as cli

        seen = []

        def fake_benchmark(config):
            seen.append(config)
            return [RunResult(label="baseline", returncode=0, n_events=config.n_events)]

        monkeypatch.setattr(cli, "run_k4run_benchmark", fake_benchmark)
        return main(["k4run", *argv]), seen

    def test_builds_the_config_from_the_flags(self, monkeypatch, stage, tmp_path):
        rc, (config,) = self._main(
            monkeypatch, "Reco.py",
            "--stage-dir", str(stage),
            "--events", "30",
            "--k4run-args=--inputFiles /data/sim.root --cms 91",
            "--variant", "truth_tracking=--truthTracking",
            "--variant", "native=--native --trackingOnly",
            "--output-dir", str(tmp_path / "out"),
        )
        assert rc == 0
        assert config.options == ["Reco.py"]
        assert config.stage_dir == stage
        assert config.n_events == 30
        assert config.extra_args == ["--inputFiles", "/data/sim.root", "--cms", "91"]
        assert config.variants == {
            "truth_tracking": ["--truthTracking"],
            "native": ["--native", "--trackingOnly"],
        }
        assert (tmp_path / "out" / "baseline_results.csv").is_file()

    def test_default_output_dir_is_named_after_the_options_file(self, monkeypatch, stage):
        monkeypatch.chdir(stage)
        _, (config,) = self._main(monkeypatch, "Reco.py", "--stage-dir", str(stage))
        assert config.log_dir == Path("logs") / "Reco"

    @pytest.mark.parametrize("spec", ["no_equals_sign", "=--args"])
    def test_malformed_variant_is_an_error(self, monkeypatch, stage, spec, capsys):
        rc, seen = self._main(monkeypatch, "Reco.py", "--stage-dir", str(stage), "--variant", spec)
        assert rc == 1 and not seen
        assert "--variant" in capsys.readouterr().err

    def test_repeated_variant_is_an_error(self, monkeypatch, stage, capsys):
        rc, seen = self._main(
            monkeypatch, "Reco.py", "--stage-dir", str(stage),
            "--variant", "a=--x", "--variant", "a=--y",
        )
        assert rc == 1 and not seen
        assert "given twice" in capsys.readouterr().err

    def test_failed_run_exits_one(self, monkeypatch, stage, tmp_path):
        import k4bench.cli as cli

        monkeypatch.setattr(
            cli, "run_k4run_benchmark",
            lambda config: [RunResult(label="baseline", returncode=2, n_events=1)],
        )
        argv = ["k4run", "Reco.py", "--stage-dir", str(stage), "--output-dir", str(tmp_path / "o")]
        assert main(argv) == 1

    def test_ddsim_command_line_is_unaffected(self):
        # Without the k4run command, --xml is still required.
        with pytest.raises(SystemExit):
            main(["--events", "2"])
