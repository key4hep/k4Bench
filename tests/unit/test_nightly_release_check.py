"""Tests for the release check in nightly_benchmark.sh's Key4hep section."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/nightly_benchmark.sh"
REQUESTED = "2026-09-03"
FRESH = "Thu Sep  3 01:25:04 2026"
STALE = "Thu Aug 27 01:25:04 2026"

# Stands in for the script above section 3. `sleep` advances the clock the
# deadline reads instead of waiting, and can publish a view the way a lagging
# CVMFS client eventually serves the new one.
PRELUDE = """\
set -euo pipefail
sleep() {
    echo "$1" >> "${SLEEP_LOG}"
    SECONDS=$(( SECONDS + $1 ))
    if [[ -n "${SLEEP_PUBLISHES:-}" ]]; then cp "${SLEEP_PUBLISHES}" "${K4H_STACK_SETUP}"; fi
}
"""


def _view(path: Path, generated: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#    Generated: {generated}\n"
        f"echo 'sourced {generated}'\n"
        "export KEY4HEP_STACK=test-stack\n"
    )
    return path


def _run(setup: Path, sleep_log: Path, **env: str) -> subprocess.CompletedProcess:
    script = SCRIPT.read_text()
    start = script.index("# ── 3.")
    section = script[start : script.index("# ── 4.", start)]
    return subprocess.run(
        ["bash", "-c", PRELUDE + section],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "K4BENCH_ROOT": str(ROOT),
            "K4H_STACK_SETUP": str(setup),
            "K4H_RELEASE_REQUESTED": REQUESTED,
            "SLEEP_LOG": str(sleep_log),
            "GITHUB_STEP_SUMMARY": "",
            **env,
        },
    )


def _waits(sleep_log: Path) -> int:
    return len(sleep_log.read_text().splitlines()) if sleep_log.exists() else 0


def test_stale_slot_is_reread_until_the_requested_release_arrives(tmp_path: Path):
    setup = _view(tmp_path / "test-platform/setup.sh", STALE)
    fresh = _view(tmp_path / "fresh.sh", FRESH)
    proc = _run(setup, tmp_path / "sleeps", SLEEP_PUBLISHES=str(fresh))
    assert proc.returncode == 0, proc.stderr
    assert _waits(tmp_path / "sleeps") == 1
    assert f"sourced {STALE}" not in proc.stdout
    assert f"sourced {FRESH}" in proc.stdout
    assert f"Release : key4hep-{REQUESTED}" in proc.stdout


def test_slot_that_never_refreshes_fails_without_being_sourced(tmp_path: Path):
    setup = _view(tmp_path / "test-platform/setup.sh", STALE)
    proc = _run(setup, tmp_path / "sleeps")
    assert proc.returncode != 0
    # 6 min of 30 s waits, then the release check refuses the stale view.
    assert _waits(tmp_path / "sleeps") == 12
    assert f"requested Key4hep release {REQUESTED}" in proc.stderr
    assert "still provides 2026-08-27" in proc.stderr
    assert "sourced" not in proc.stdout
