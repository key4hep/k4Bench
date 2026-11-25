"""Tests for the EOS upload in nightly_benchmark.sh's section 9."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/nightly_benchmark.sh"
EOS_ROOT = "/eos/test/k4bench"
RUN_DIR = "runs/sim/DET/DET_o1/PLAT/key4hep-2026-11-25/single_e-_10GeV/2026-11-25"

# Stands in for the script above section 9 and for the XRootD clients: EOS is a
# local directory, root://HOST//path its file below it. Every call is logged,
# so the order of what changed on EOS can be checked.
PRELUDE = """\
set -euo pipefail
voms-proxy-init() { :; }
xrdfs() {
    local cmd="$2"; shift 2
    echo "xrdfs ${cmd} $*" >> "${CALLS}"
    case "${cmd}" in
        mkdir) mkdir -p "${EOS_LOCAL}${2}" ;;
        ls)
            [[ -z "${LS_FAILS:-}" ]] || return 1
            local p
            for p in "${EOS_LOCAL}$1"/*; do
                if [[ -e "${p}" ]]; then echo "${p#"${EOS_LOCAL}"}"; fi
            done
            ;;
        rm) rm "${EOS_LOCAL}$1" ;;
        *) return 2 ;;
    esac
}
xrdcp() {
    [[ "$1" == --force ]] || return 2
    local src="$2" dest="${3#root://"${EOS_FQDN}"/}"
    echo "xrdcp $(basename "${src}") ${dest##*/}" >> "${CALLS}"
    [[ "$(basename "${src}")" != "${XRDCP_FAILS:-}" ]] || return 1
    cp "${src}" "${EOS_LOCAL}${dest%%\\?*}"
}
"""


def _write(directory: Path, files: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content)


def _contents(directory: Path) -> dict[str, str]:
    return {p.name: p.read_text() for p in directory.iterdir()} if directory.is_dir() else {}


class Upload:
    """Section 9 run against one local output directory and a local EOS."""

    def __init__(self, tmp_path: Path):
        self.output = tmp_path / "logs/benchmark"
        self.eos = tmp_path / "eos"
        self.run = self.eos / EOS_ROOT.lstrip("/") / RUN_DIR
        self.calls_log = tmp_path / "calls"

    def __call__(self, bench_rc: int = 0, **env: str) -> subprocess.CompletedProcess:
        script = SCRIPT.read_text()
        section = script[script.index("# ── 9."):]
        return subprocess.run(
            ["bash", "-c", PRELUDE + section],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "EOS_FQDN": "eos.example",
                "EOS_ROOT": EOS_ROOT,
                "RUN_DIR": RUN_DIR,
                "OUTPUT_DIR": str(self.output),
                "BENCH_RC": str(bench_rc),
                "X509_USER_CERT": "cert.pem",
                "X509_USER_KEY": "key.pem",
                "EOS_LOCAL": str(self.eos),
                "CALLS": str(self.calls_log),
                **env,
            },
        )

    def calls(self) -> list[str]:
        return self.calls_log.read_text().splitlines()

    def changes(self) -> list[str]:
        """What changed EOS, in order: every call but a listing or the mkdir."""
        return [c for c in self.calls() if not c.startswith(("xrdfs ls", "xrdfs mkdir"))]


NEW = {
    "baseline_results.csv": "new results",
    "baseline.log": "new log",
    "machine_info.json": "new machine",
    "run_info.json": "new run info",
}
EARLIER = {
    "baseline_results.csv": "old results",
    "baseline.log": "old log",
    "no_ECal_results.csv": "old, a config this attempt did not finish",
    "no_ECal.log": "old log of it",
    "run_info.json": "old run info",
}


@pytest.fixture
def upload(tmp_path: Path) -> Upload:
    up = Upload(tmp_path)
    _write(up.output, NEW)
    return up


def test_a_first_upload_puts_run_info_last(upload: Upload):
    proc = upload()
    assert proc.returncode == 0, proc.stderr
    assert _contents(upload.run) == NEW
    changes = upload.changes()
    assert all(c.startswith("xrdcp") for c in changes)
    assert changes[-1] == "xrdcp run_info.json run_info.json?eos.atomic=1"
    # Every file lands atomically.
    assert all(c.endswith("?eos.atomic=1") for c in changes)


def test_a_rerun_replaces_the_earlier_attempt_whole(upload: Upload):
    _write(upload.run, EARLIER)
    proc = upload()
    assert proc.returncode == 0, proc.stderr
    # Exactly this attempt's files: what only the earlier one wrote is gone.
    assert _contents(upload.run) == NEW
    changes = upload.changes()
    # run_info.json is taken down before anything else changes, and put back last.
    assert changes[0] == f"xrdfs rm {EOS_ROOT}/{RUN_DIR}/run_info.json"
    assert changes[-1] == "xrdcp run_info.json run_info.json?eos.atomic=1"
    assert sorted(c for c in changes[1:] if c.startswith("xrdfs rm")) == sorted(
        f"xrdfs rm {EOS_ROOT}/{RUN_DIR}/{name}" for name in ("no_ECal.log", "no_ECal_results.csv")
    )


def test_a_failed_copy_leaves_the_run_without_run_info(upload: Upload):
    _write(upload.run, EARLIER)
    proc = upload(XRDCP_FAILS="machine_info.json")
    assert proc.returncode != 0
    assert "Failed to upload" in proc.stderr
    # Readers see an upload that has not finished, never the earlier run's record.
    assert "run_info.json" not in _contents(upload.run)
    assert not any(c.startswith("xrdcp run_info.json") for c in upload.calls())


def test_a_listing_that_fails_changes_nothing(upload: Upload):
    _write(upload.run, EARLIER)
    proc = upload(LS_FAILS="1")
    assert proc.returncode != 0
    assert "Failed to list" in proc.stderr
    assert _contents(upload.run) == EARLIER
    assert upload.changes() == []


def test_a_failed_benchmark_is_uploaded_then_fails_the_job(upload: Upload):
    proc = upload(bench_rc=3)
    assert proc.returncode == 3
    assert _contents(upload.run) == NEW
    assert "results uploaded regardless" in proc.stderr
