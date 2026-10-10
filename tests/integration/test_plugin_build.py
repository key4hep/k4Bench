"""Integration tests for the k4Bench timing plugin build.

These tests call build.sh directly so that any change to the C++ source
triggers a real recompile — bypassing the early-return in ensure_plugin_built()
that would otherwise skip the build when a stale .so already exists.

Require the Key4hep environment (CMake, DD4hep headers, C++ compiler).
"""

from __future__ import annotations

import subprocess

import pytest

from k4bench.plugin.runtime import (
    _find_plugin_root,
    auditor_options_file,
    find_auditor_dir,
    find_plugin_lib_dir,
    setup_auditor_environment,
)

# Every Gaudi environment puts gaudirun.py on PATH; GAUDI_PLUGIN_PATH is not
# set by all of them.
requires_gaudi = pytest.mark.skipif(
    __import__("shutil").which("gaudirun.py") is None,
    reason="the k4BenchAuditor is built only in a Gaudi environment",
)


@pytest.fixture(scope="module")
def built_plugin():
    """Run build.sh and return (returncode, stdout, stderr)."""
    build_sh = _find_plugin_root() / "build.sh"
    result = subprocess.run(
        ["bash", str(build_sh)],
        capture_output=True,
        text=True,
    )
    return result


@pytest.mark.integration
def test_plugin_build_succeeds(built_plugin):
    """build.sh exits 0 against the current C++ source."""
    assert built_plugin.returncode == 0, (
        f"Plugin build failed:\n{built_plugin.stdout}\n{built_plugin.stderr}"
    )


@pytest.mark.integration
def test_auditor_target_without_gaudi_fails_and_builds_nothing():
    """Without Gaudi the auditor target says so instead of reporting itself up
    to date, and leaves the DDG4 plugins alone, built or not."""
    import os

    path = os.pathsep.join(
        d
        for d in os.environ.get("PATH", "").split(os.pathsep)
        if not os.path.exists(os.path.join(d, "gaudirun.py"))
    )
    env = {k: v for k, v in os.environ.items() if k != "GAUDI_PLUGIN_PATH"}
    env["PATH"] = path
    result = subprocess.run(
        ["bash", str(_find_plugin_root() / "build.sh"), "auditor"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "No Gaudi environment" in result.stderr
    assert "Building" not in result.stdout


@pytest.mark.integration
def test_build_sh_refuses_an_unknown_target():
    result = subprocess.run(
        ["bash", str(_find_plugin_root() / "build.sh"), "everything"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Usage" in result.stderr


@pytest.mark.integration
def test_plugin_library_is_found_after_build(built_plugin):
    """find_plugin_lib_dir locates libk4BenchTimingAction.so after a successful build."""
    assert built_plugin.returncode == 0, "Skipped: build already failed"
    lib_dir = find_plugin_lib_dir()
    assert any(lib_dir.glob("libk4BenchTimingAction.so*")), (
        f"No libk4BenchTimingAction.so* in {lib_dir}"
    )


@pytest.mark.integration
def test_plugin_library_is_a_regular_file(built_plugin):
    """The located .so is a real file, not a directory or dangling symlink."""
    assert built_plugin.returncode == 0, "Skipped: build already failed"
    lib_dir = find_plugin_lib_dir()
    so_files = list(lib_dir.glob("libk4BenchTimingAction.so*"))
    assert so_files and all(f.is_file() for f in so_files)


@pytest.mark.integration
@pytest.mark.parametrize("n_events", [0, 3])
def test_event_plugin_writes_memory_json_without_simulation(tmp_path, n_events):
    """Exercise the real callbacks/writer with minimal DDG4 interfaces.

    No geometry or ddsim run is needed. The real-library tests above verify
    compatibility with DDG4; this harness checks /proc reads and JSON syntax.
    """
    import json
    import os
    import shutil

    from k4bench.plugin.schema import EVENT_SCHEMA_VERSION

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    headers = {
        "G4Event.hh": """
#pragma once
class G4Event {
  int id;
public:
  explicit G4Event(int value) : id(value) {}
  int GetEventID() const { return id; }
};
""",
        "DDG4/Geant4Context.h": """
#pragma once
namespace dd4hep { namespace sim { struct Geant4Context {}; } }
""",
        "DDG4/Geant4EventAction.h": """
#pragma once
#include <string>
#include <G4Event.hh>
#include <DDG4/Geant4Context.h>
namespace dd4hep { namespace sim {
class Geant4EventAction {
public:
  Geant4EventAction(Geant4Context*, const std::string&) {}
  virtual ~Geant4EventAction() = default;
  virtual void begin(const G4Event*) {}
  virtual void end(const G4Event*) {}
};
} }
""",
        "DD4hep/Printout.h": """
#pragma once
namespace dd4hep {
enum { INFO, ERROR };
template <typename... Args> void printout(Args...) {}
}
""",
        "DDG4/Factories.h": "#define DECLARE_GEANT4ACTION(name)\n",
    }
    for name, content in headers.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    source = tmp_path / "sanity.cpp"
    source.write_text("""
#include "k4BenchTimingAction.cpp"
#include <sys/mman.h>
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) return 3;
  const int n_events = std::atoi(argv[1]);
  // A transient virtual mapping before event processing must survive in VmPeak.
  constexpr std::size_t bytes = 128 * 1024 * 1024;
  void* mapping = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED) return 1;
  const long peak = k4bench::read_vmpeak_kb();
  if (munmap(mapping, bytes) != 0) return 2;
  {
    dd4hep::sim::k4BenchTimingAction action(nullptr, "sanity");
    for (int i = 0; i < n_events; ++i) {
      G4Event event(i);
      action.begin(&event);
      action.end(&event);
    }
    // Shutdown during a begun event must not leave unequal array lengths.
    G4Event unfinished(n_events);
    action.begin(&unfinished);
  }
  std::cout << peak;
}
""")
    executable = tmp_path / "sanity"
    env = {**os.environ, "CCACHE_DISABLE": "1"}
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(tmp_path),
            "-I",
            str(_find_plugin_root()),
            str(source),
            "-o",
            str(executable),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    output = tmp_path / "sanity_events.json"
    result = subprocess.run(
        [str(executable), str(n_events)],
        env={**env, "K4BENCH_EVENT_JSON": str(output)},
        check=True,
        capture_output=True,
        text=True,
    )
    raw = json.loads(output.read_text())
    assert next(iter(raw)) == "schema_version"
    assert raw.pop("schema_version") == EVENT_SCHEMA_VERSION
    sampled_peak_kb = int(result.stdout)
    assert sampled_peak_kb >= 128 * 1024
    assert raw.pop("peak_vmem_mb") >= sampled_peak_kb / 1024 - 0.001
    assert set(raw) == {
        "event_numbers",
        "event_times_s",
        "event_rss_begin_mb",
        "event_rss_end_mb",
        "event_rss_anon_begin_mb",
        "event_rss_anon_end_mb",
        "event_rss_file_end_mb",
    }
    assert {len(values) for values in raw.values()} == {n_events}
    assert raw["event_numbers"] == list(range(n_events))
    assert all(value >= 0 for values in raw.values() for value in values)


@pytest.mark.integration
@requires_gaudi
def test_auditor_is_found_after_build(built_plugin):
    """build.sh builds the Gaudi auditor next to its .components manifest."""
    assert built_plugin.returncode == 0, "Skipped: build already failed"
    plugin_dir = find_auditor_dir()
    assert any(plugin_dir.glob("libk4BenchAuditor.so*"))
    assert "k4BenchAuditor" in "".join(p.read_text() for p in plugin_dir.glob("*.components"))


def _run_audited_job(tmp_path, label, options):
    """Run the job *options* under the auditor with gaudirun.py and return its
    component timing and event table."""
    import os

    from k4bench.analysis.loader import load_component_timing, load_event_timing

    job = tmp_path / f"{label}.py"
    job.write_text(options)
    env = dict(os.environ)
    assert setup_auditor_environment(
        env=env,
        components_json_path=tmp_path / f"{label}_components.json",
        event_json_path=tmp_path / f"{label}_events.json",
        joboptions_path=tmp_path / f"{label}_joboptions.opts",
    )
    result = subprocess.run(
        ["gaudirun.py", str(job), str(auditor_options_file())],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return load_component_timing(tmp_path)[label], load_event_timing(tmp_path)[label]


@pytest.mark.integration
@requires_gaudi
def test_auditor_measures_a_real_gaudi_job(built_plugin, tmp_path):
    """A sequencer running one algorithm that sleeps 1 s per event: the auditor
    must find both, charge the sleep to the sleeper as wall but not CPU time,
    and write an event file the ddsim readers accept."""
    import json

    from k4bench.plugin.schema import EVENT_SCHEMA_VERSION

    assert built_plugin.returncode == 0, "Skipped: build already failed"
    timing, events = _run_audited_job(
        tmp_path,
        "nap",
        "from Configurables import ApplicationMgr, Gaudi__Sequencer, GaudiTesting__SleepyAlg\n"
        "nap = GaudiTesting__SleepyAlg('Nap', SleepTime=1)\n"
        "ApplicationMgr(EvtSel='NONE', EvtMax=2, TopAlg=[Gaudi__Sequencer('Top', Members=[nap])])\n",
    )

    assert timing.components.loc["Nap", "parent"] == "Top"
    assert timing.components.loc["Nap", "category"] == "algorithm"
    assert timing.components.loc["Nap", "library"].endswith(".so")
    assert timing.components.loc["Nap", "execute_calls"] == 2
    wall, cpu = timing.execute["wall_s"], timing.execute["cpu_s"]
    assert wall.index.tolist() == [0, 1]
    assert ((wall["Nap"] > 0.99) & (wall["Nap"] < 1.5)).all()
    assert (cpu["Nap"] < 0.1).all()
    assert (wall["Top"] < 0.1).all(), "a sequencer's self time must exclude its children"
    assert "initialize" in timing.lifecycle

    raw_events = json.loads((tmp_path / "nap_events.json").read_text())
    assert next(iter(raw_events)) == "schema_version"
    assert raw_events["schema_version"] == EVENT_SCHEMA_VERSION
    assert events["event_number"].tolist() == [0, 1]
    # With a single top-level call, self costs add up to the event's time.
    assert wall.fillna(0).sum(axis=1).to_numpy() == pytest.approx(
        events["event_time_s"].to_numpy(), abs=1e-3
    )
    assert (events["rss_anon_end_mb"] > 0).all()

    assert "Nap.SleepTime" in (tmp_path / "nap_joboptions.opts").read_text()


@pytest.mark.integration
@requires_gaudi
def test_auditor_spans_an_event_over_several_top_level_algorithms(built_plugin, tmp_path):
    """Two top-level algorithms that each sleep 1 s per event: both have no
    parent, and the event runs from the first one's start to the second one's
    end, exceeding their summed self costs only by the gap between them."""
    assert built_plugin.returncode == 0, "Skipped: build already failed"
    timing, events = _run_audited_job(
        tmp_path,
        "pair",
        "from Configurables import ApplicationMgr, GaudiTesting__SleepyAlg\n"
        "first = GaudiTesting__SleepyAlg('First', SleepTime=1)\n"
        "second = GaudiTesting__SleepyAlg('Second', SleepTime=1)\n"
        "ApplicationMgr(EvtSel='NONE', EvtMax=2, TopAlg=[first, second])\n",
    )

    assert timing.components.loc[["First", "Second"], "parent"].isna().all()
    wall = timing.execute["wall_s"][["First", "Second"]]
    assert ((wall > 0.99) & (wall < 1.5)).all().all()
    assert events["event_number"].tolist() == [0, 1]
    event_time = events["event_time_s"].to_numpy()
    assert (event_time > 1.98).all(), "the event must cover both top-level calls"
    gap = event_time - timing.execute["wall_s"].fillna(0).sum(axis=1).to_numpy()
    assert ((gap > -1e-3) & (gap < 0.05)).all()


@pytest.mark.integration
@requires_gaudi
def test_auditor_leaves_a_filtered_algorithm_missing(built_plugin, tmp_path):
    """A sequencer whose filter passes every other event, followed by an
    algorithm that sleeps 1 s: the sleeper runs only in the passed events, and
    in the others its cost is missing rather than zero."""
    assert built_plugin.returncode == 0, "Skipped: build already failed"
    timing, events = _run_audited_job(
        tmp_path,
        "filtered",
        "from Configurables import ApplicationMgr, Gaudi__Sequencer\n"
        "from Configurables import GaudiTesting__OddEventsFilter, GaudiTesting__SleepyAlg\n"
        "members = [GaudiTesting__OddEventsFilter('Odd'), GaudiTesting__SleepyAlg('Nap', SleepTime=1)]\n"
        "ApplicationMgr(EvtSel='NONE', EvtMax=4, TopAlg=[Gaudi__Sequencer('Top', Members=members)])\n",
    )

    assert timing.components.loc["Odd", "parent"] == "Top"
    assert timing.components.loc["Nap", "parent"] == "Top"
    assert timing.components.loc["Odd", "execute_calls"] == 4
    assert timing.components.loc["Nap", "execute_calls"] == 2
    wall = timing.execute["wall_s"]
    assert wall.index.tolist() == [0, 1, 2, 3]
    assert wall["Odd"].notna().all()
    ran = wall["Nap"].notna()
    assert ran.tolist() in ([True, False] * 2, [False, True] * 2)
    assert ((wall["Nap"][ran] > 0.99) & (wall["Nap"][ran] < 1.5)).all()
    top = timing.inclusive("wall_s")["Top"]
    assert (top[ran] > 0.99).all()
    assert (top[~ran] < 0.1).all()
    # A single top-level call per event, so self costs add up to the event's
    # time whether the sleeper ran or not.
    assert wall.fillna(0).sum(axis=1).to_numpy() == pytest.approx(
        events["event_time_s"].to_numpy(), abs=1e-3
    )
