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
    assert (plugin_dir / "libk4BenchAllocCounter.so").is_file()


# Drives the preloaded counter from Python. The structures mirror
# k4BenchAllocCounter.h field by field.
_PROBE_PRELUDE = """\
import ctypes, json

class Counters(ctypes.Structure):
    _fields_ = [("allocations", ctypes.c_uint64), ("allocated_bytes", ctypes.c_uint64),
                ("freed_bytes", ctypes.c_uint64), ("peak_live_bytes", ctypes.c_int64),
                ("largest_allocation_bytes", ctypes.c_uint64), ("paused", ctypes.c_bool)]

class OwnerTable(ctypes.Structure):
    _fields_ = [("entries", ctypes.c_uint64), ("peak_entries", ctypes.c_uint64),
                ("mapped_bytes", ctypes.c_uint64), ("owners", ctypes.c_uint32),
                ("failed", ctypes.c_bool)]

process = ctypes.CDLL(None)  # the preloaded malloc and free, not libc's own
process.k4bench_thread_alloc_counters.restype = ctypes.POINTER(Counters)
for name in ("malloc", "__libc_malloc"):
    getattr(process, name).restype = ctypes.c_void_p
    getattr(process, name).argtypes = [ctypes.c_size_t]
process.free.argtypes = [ctypes.c_void_p]
process.realloc.restype = ctypes.c_void_p
process.realloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
process.malloc_usable_size.restype = ctypes.c_size_t
process.malloc_usable_size.argtypes = [ctypes.c_void_p]
process.k4bench_exchange_alloc_owner.restype = ctypes.c_uint32
process.k4bench_exchange_alloc_owner.argtypes = [ctypes.c_uint32]
process.k4bench_read_owners.restype = ctypes.c_size_t
process.k4bench_read_owners.argtypes = [ctypes.POINTER(ctypes.c_int64)] * 2 + [ctypes.c_size_t]
process.k4bench_owner_table.restype = ctypes.c_bool
process.k4bench_owner_table.argtypes = [ctypes.POINTER(OwnerTable)]
counters = process.k4bench_thread_alloc_counters().contents
"""

_ALLOC_PROBE = _PROBE_PRELUDE + """\

def counted(action):
    before = (counters.allocations, counters.allocated_bytes, counters.freed_bytes)
    result = action()
    after = (counters.allocations, counters.allocated_bytes, counters.freed_bytes)
    return result, [b - a for a, b in zip(before, after)]

def live():
    return counters.allocated_bytes - counters.freed_bytes

block, allocated = counted(lambda: process.malloc(1 << 20))
_, freed = counted(lambda: process.free(block))

# A block's heap memory is what glibc can hand out of it plus malloc's header:
# 8 bytes, or 16 for a block mapped on its own.
headers = []
for size in (1, 24, 100, 1000, 100_000, 1 << 20, 64 << 20):
    block, (_, heap_bytes, _) = counted(lambda: process.malloc(size))
    headers.append(heap_bytes - process.malloc_usable_size(block))
    process.free(block)

# Resized and freed again, a block gives back exactly what it took. Inside a
# function, so Python allocates nothing of its own meanwhile.
def resize_and_free():
    process.free(process.realloc(process.realloc(process.malloc(100), 5000), 200_000))

_, (_, taken, given_back) = counted(resize_and_free)
balance = taken - given_back

# Shrunk in place, a heap block never held its old and new size at once.
def shrink_in_place():
    block = process.malloc(64 << 10)
    start = counters.peak_live_bytes = live()
    resized = process.realloc(block, 32 << 10)
    peak = counters.peak_live_bytes - start
    process.free(resized)
    return resized == block, peak

in_place, in_place_peak = shrink_in_place()

# Grown, a block mapped on its own (64 MiB, above glibc's largest mmap
# threshold) is moved by mremap and never held at both sizes either.
def grow_mapped():
    block = process.malloc(64 << 20)
    start = counters.peak_live_bytes = live()
    process.free(process.realloc(block, 128 << 20))
    return counters.peak_live_bytes - start

mapped_peak = grow_mapped()

# A window like the auditor's around one call: 1 MiB held at once, then freed,
# then a small block.
start = counters.peak_live_bytes = live()
counters.largest_allocation_bytes = 0
process.free(process.malloc(1 << 20))
process.free(process.malloc(64))
peak = counters.peak_live_bytes - start
kept = live() - start
largest = counters.largest_allocation_bytes

counters.paused = True
_, paused = counted(lambda: process.free(process.malloc(1 << 20)))
print(json.dumps({"allocated": allocated, "freed": freed, "headers": headers,
                  "balance": balance, "in_place": in_place, "in_place_peak": in_place_peak,
                  "mapped_peak": mapped_peak, "peak": peak, "kept": kept, "largest": largest,
                  "paused": paused}))
"""


def _probe(script: str, owners: bool) -> dict:
    """What *script* prints, run under the preloaded counter, which records
    heap owners when *owners* is set."""
    import json
    import os
    import sys

    env = {**os.environ, "LD_PRELOAD": str(find_auditor_dir() / "libk4BenchAllocCounter.so")}
    env.pop("K4BENCH_ALLOC_OWNERS", None)
    if owners:
        env["K4BENCH_ALLOC_OWNERS"] = "1"
    # A call back into malloc from inside the counter's table would deadlock
    # on its lock, so a hang fails the test instead of stalling it.
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout)


@pytest.mark.integration
@requires_gaudi
@pytest.mark.parametrize("owners", [False, True], ids=["counting", "with-owners"])
def test_alloc_counter_counts_the_calling_threads_heap_blocks(built_plugin, owners):
    """Preloaded, the counter adds every block malloc hands out and free takes
    back to the calling thread's counters, as the heap memory glibc's own
    header says the block takes, keeps the most the thread held at once and
    its largest block since they were last reset, and counts nothing while
    paused, whether it records heap owners or not."""
    assert built_plugin.returncode == 0, "Skipped: build already failed"
    deltas = _probe(_ALLOC_PROBE, owners)

    allocations, allocated_bytes, freed_bytes = deltas["allocated"]
    assert allocations >= 1
    assert allocated_bytes >= 1 << 20
    assert freed_bytes == 0
    assert deltas["freed"][2] >= 1 << 20
    assert set(deltas["headers"]) <= {8, 16}, deltas["headers"]
    assert deltas["balance"] == 0
    assert deltas["in_place"]
    assert deltas["in_place_peak"] < 1 << 10
    assert deltas["mapped_peak"] < 65 << 20
    assert deltas["peak"] >= 1 << 20
    assert deltas["kept"] < 1 << 20
    assert deltas["largest"] >= 1 << 20
    assert deltas["paused"] == [0, 0, 0]


_OWNER_PROBE = _PROBE_PRELUDE + """\
import os, threading

A, B, C = 5, 6, 7
held = (ctypes.c_int64 * 8)()
peak = (ctypes.c_int64 * 8)()
table = OwnerTable()
malloc, free, realloc = process.malloc, process.free, process.realloc

def owners():
    process.k4bench_read_owners(held, peak, len(held))
    return list(held)

def charged(owner, action):
    \"\"\"*action*'s result, with what it allocates charged to *owner*.\"\"\"
    previous = process.k4bench_exchange_alloc_owner(owner)
    try:
        return action()
    finally:
        process.k4bench_exchange_alloc_owner(previous)

def change(action):
    \"\"\"*action*'s result, and what it changes in each owner's held bytes.\"\"\"
    before = owners()
    result = action()
    return result, [b - a for a, b in zip(before, owners())]

out = {"enabled": process.k4bench_owner_table(ctypes.byref(table)), "owners": table.owners}

# A's block is A's until freed, whoever frees it.
block, out["a_allocates"] = change(lambda: charged(A, lambda: malloc(1 << 20)))
_, out["b_frees"] = change(lambda: charged(B, lambda: free(block)))

# Resized by B, A's block stays A's: grown in the heap, moved to a block glibc
# maps on its own, then freed.
block, out["a_allocates_small"] = change(lambda: charged(A, lambda: malloc(100)))
block, out["b_grows"] = change(lambda: charged(B, lambda: realloc(block, 5000)))
block, out["b_moves"] = change(lambda: charged(B, lambda: realloc(block, 64 << 20)))
_, out["b_frees_resized"] = change(lambda: charged(B, lambda: free(block)))

# Shrunk in place, a block never held both sizes, so the peak never sees them.
def shrink_in_place():
    block = charged(A, lambda: malloc(64 << 10))
    start = owners()[A]
    process.k4bench_reset_owner_peaks()
    resized = charged(B, lambda: realloc(block, 32 << 10))
    owners()
    result = resized == block, peak[A] - start, held[A] - start
    free(resized)
    return result

out["in_place"], out["in_place_peak"], out["in_place_change"] = shrink_in_place()

# Moved, it held both while realloc copied.
def move():
    block = charged(A, lambda: malloc(1000))
    old_bytes = owners()[A]
    process.k4bench_reset_owner_peaks()
    moved = charged(A, lambda: realloc(block, 64 << 20))
    owners()
    result = peak[A] - held[A]
    free(moved)
    return result, old_bytes - owners()[A]

out["moved_extra_peak"], out["moved_old_bytes"] = move()

# Blocks it never recorded are nobody's: freeing them changes nothing.
counters.paused = True
paused, out["paused_allocates"] = change(lambda: charged(A, lambda: malloc(1 << 20)))
counters.paused = False
foreign = process.__libc_malloc(1 << 20)  # glibc's own, as before ownership started
_, out["unrecorded_frees"] = change(lambda: charged(B, lambda: (free(paused), free(foreign))))

# A block allocated on one thread and freed on another goes back to its owner,
# and each thread has an owner of its own.
def on_thread(action):
    result = []
    thread = threading.Thread(target=lambda: result.append(action()))
    thread.start()
    thread.join()
    return result[0]

block, out["a_allocates_on_main"] = change(lambda: charged(A, lambda: malloc(1 << 16)))
_, out["thread_frees"] = change(lambda: on_thread(lambda: free(block)))
block, out["c_allocates_on_thread"] = change(lambda: on_thread(lambda: charged(C, lambda: malloc(1 << 16))))
_, out["main_frees"] = change(lambda: charged(B, lambda: free(block)))

# A forked child has a table of its own: freeing the parent's block there
# changes nothing in the parent.
block = charged(A, lambda: malloc(1 << 16))
before = owners()[A]
child = os.fork()
if child == 0:
    _, freed = change(lambda: free(block))
    os._exit(0 if freed[A] < 0 else 1)
out["child_status"] = os.waitstatus_to_exitcode(os.waitpid(child, 0)[1])
out["parent_after_fork"] = owners()[A] - before
free(block)

# A million blocks grow the table from its first 65536 slots.
blocks = (ctypes.c_void_p * 1_000_000)()
def allocate():
    for i in range(len(blocks)):
        blocks[i] = malloc(32)
def free_all():
    for b in blocks:
        free(b)
_, out["million_allocated"] = change(lambda: charged(A, allocate))
process.k4bench_owner_table(ctypes.byref(table))
out["million_table"] = [table.entries, table.peak_entries, table.mapped_bytes, table.failed]
_, out["million_freed"] = change(free_all)
print(json.dumps(out))
"""


@pytest.mark.integration
@requires_gaudi
def test_alloc_counter_charges_each_block_to_its_owner_until_it_is_freed(built_plugin):
    """With K4BENCH_ALLOC_OWNERS=1, every block is held by the owner of the
    thread that allocated it until it is freed, whoever frees it and on
    whichever thread; resizing keeps its owner, and a block the counter never
    recorded changes nothing when freed."""
    assert built_plugin.returncode == 0, "Skipped: build already failed"
    out = _probe(_OWNER_PROBE, owners=True)
    A, B, C = 5, 6, 7

    def only(owner, change):
        return [value for index, value in enumerate(change) if index != owner]

    assert out["enabled"] and out["owners"] >= 1024
    allocated = out["a_allocates"][A]
    assert allocated >= 1 << 20
    assert out["b_frees"][A] == -allocated
    assert out["b_frees"][B] == 0, "the freeing owner gives back nothing of its own"
    assert set(only(A, out["a_allocates"]) + only(A, out["b_frees"])) == {0}

    small = out["a_allocates_small"][A]
    grown, moved = out["b_grows"][A], out["b_moves"][A]
    assert grown > 4800 and moved > 63 << 20
    assert out["b_frees_resized"][A] == -(small + grown + moved)
    assert set(only(A, out["b_grows"]) + only(A, out["b_moves"]) + only(A, out["b_frees_resized"])) == {0}

    assert out["in_place"]
    assert out["in_place_peak"] == 0, "shrunk in place, a block never held both sizes"
    assert -(33 << 10) < out["in_place_change"] < -(31 << 10)
    assert out["moved_extra_peak"] == out["moved_old_bytes"] > 1000, "moved, it held both while copied"

    assert set(out["paused_allocates"]) == {0}
    assert set(out["unrecorded_frees"]) == {0}

    on_main = out["a_allocates_on_main"][A]
    assert on_main >= 1 << 16
    assert out["thread_frees"][A] == -on_main
    assert out["c_allocates_on_thread"][C] >= 1 << 16
    assert out["main_frees"][C] == -out["c_allocates_on_thread"][C]
    assert out["main_frees"][B] == 0

    assert out["child_status"] == 0, "the child gave the block back in its own table"
    assert out["parent_after_fork"] == 0

    # malloc(32) takes a 48-byte chunk; Python's loop allocates a little of its own.
    blocks = 1_000_000 * 48
    assert blocks <= out["million_allocated"][A] < blocks + 4096
    entries, peak_entries, mapped_bytes, failed = out["million_table"]
    assert not failed
    assert entries >= 1_000_000 and peak_entries >= entries
    assert mapped_bytes >= 2 * peak_entries * 16, "the table doubles before it is half full"
    assert -blocks - 4096 < out["million_freed"][A] <= -blocks


_FORK_PROBE = _PROBE_PRELUDE + """\
import os, threading

A, C = 5, 7
held = (ctypes.c_int64 * 8)()
malloc, free = process.malloc, process.free

def held_by(owner):
    process.k4bench_read_owners(held, None, len(held))
    return held[owner]

blocks = (ctypes.c_void_p * 10_000)()
process.k4bench_exchange_alloc_owner(A)
for i in range(len(blocks)):
    blocks[i] = malloc(32)
process.k4bench_exchange_alloc_owner(0)

# Another thread keeps the table growing all the while, so forks also land
# while it moves the entries into a larger one (3 or 4 children in 100 lost
# blocks when the lock was only released in the child).
stop = threading.Event()
def churn():
    process.k4bench_exchange_alloc_owner(C)
    kept = (ctypes.c_void_p * 4_000_000)()
    for i in range(len(kept)):
        if stop.is_set():
            break
        kept[i] = malloc(16)
thread = threading.Thread(target=churn)
thread.start()

failures = 0
for _ in range(100):
    child = os.fork()
    if child == 0:
        before = held_by(A)
        for block in blocks:
            free(block)
        # A block the child's table lost would stay A's: malloc(32) takes 48 bytes.
        os._exit(0 if before - held_by(A) >= len(blocks) * 48 else 1)
    failures += os.waitstatus_to_exitcode(os.waitpid(child, 0)[1]) != 0
stop.set()
thread.join()
print(json.dumps({"failures": failures}))
"""


@pytest.mark.integration
@requires_gaudi
def test_alloc_counter_copies_its_table_whole_into_a_forked_child(built_plugin):
    """A process forked while another thread changes the table gets the table
    whole: every block allocated before the fork is still its owner's in the
    child, and freeing them all gives back exactly their memory."""
    import warnings

    assert built_plugin.returncode == 0, "Skipped: build already failed"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # fork with threads running
        assert _probe(_FORK_PROBE, owners=True) == {"failures": 0}


@pytest.mark.integration
@requires_gaudi
def test_alloc_counter_records_no_owners_by_default(built_plugin):
    """Without K4BENCH_ALLOC_OWNERS=1, the counter keeps no table and reports
    no owners."""
    assert built_plugin.returncode == 0, "Skipped: build already failed"
    out = _probe(
        _PROBE_PRELUDE
        + "held = (ctypes.c_int64 * 8)()\n"
        + "process.k4bench_exchange_alloc_owner(5)\n"
        + "process.free(process.malloc(1 << 20))\n"
        + "print(json.dumps({'enabled': process.k4bench_owner_table(ctypes.byref(OwnerTable())),\n"
        + "                  'owners': process.k4bench_read_owners(held, held, 8)}))\n",
        owners=False,
    )
    assert out == {"enabled": False, "owners": 0}


def _run_audited_job(tmp_path, label, options, *, owners=False):
    """Run the job *options* under the auditor with gaudirun.py and return its
    component timing and event table. With *owners*, the job records which
    component allocated each heap block, and the auditor writes its heap file."""
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
        heap_json_path=tmp_path / f"{label}_heap.json" if owners else None,
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
    and as the thread blocking, and write an event file the ddsim readers
    accept."""
    import json

    from k4bench.analysis.components import THREAD_METRICS
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
    assert timing.components.loc["Nap", "impl_library"] == timing.components.loc["Nap", "library"]
    assert timing.components.loc["Nap", "execute_calls"] == 2
    wall, cpu = timing.execute["wall_s"], timing.execute["cpu_s"]
    assert wall.index.tolist() == [0, 1]
    assert ((wall["Nap"] > 0.99) & (wall["Nap"] < 1.5)).all()
    assert (cpu["Nap"] < 0.1).all()
    assert set(THREAD_METRICS) <= set(timing.execute)
    assert (timing.execute["voluntary_context_switches"]["Nap"] >= 1).all(), "a sleep blocks"
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
@pytest.mark.skipif(
    not any(
        (__import__("pathlib").Path(d) / "libk4MarlinWrapperPlugins.so").exists()
        for d in __import__("os").environ.get("LD_LIBRARY_PATH", "").split(":")
        if d
    ),
    reason="needs k4MarlinWrapper",
)
def test_auditor_finds_the_library_of_a_wrapped_marlin_processor(built_plugin, tmp_path):
    """A MarlinProcessorWrapper running Marlin's own Statusmonitor: the class
    doing the work is the processor, defined in Marlin's library, which the
    auditor finds through Marlin's ProcessorMgr rather than naming the
    wrapper's own library."""
    from pathlib import Path

    assert built_plugin.returncode == 0, "Skipped: build already failed"
    timing, _ = _run_audited_job(
        tmp_path,
        "wrapped",
        "from Configurables import ApplicationMgr, MarlinProcessorWrapper\n"
        "status = MarlinProcessorWrapper('Status', ProcessorType='Statusmonitor',\n"
        "                                Parameters={'HowOften': ['1']})\n"
        "ApplicationMgr(EvtSel='NONE', EvtMax=1, TopAlg=[status])\n",
    )

    status = timing.components.loc["Status"]
    assert status["type"] == "MarlinProcessorWrapper"
    assert status["impl"] == "Statusmonitor"
    assert Path(status["impl_library"]).name.startswith("libMarlin.so")
    assert status["impl_library"] != status["library"]


@pytest.mark.integration
@requires_gaudi
def test_auditor_records_each_algorithms_data_handles(built_plugin, tmp_path):
    """A producer, a transformer and a consumer of Gaudi's functional test
    suite: each algorithm's declared reads and writes are its data handles'
    keys, and a component holding none (a service) has none."""
    import json

    assert built_plugin.returncode == 0, "Skipped: build already failed"
    _run_audited_job(
        tmp_path,
        "handles",
        "from Configurables import ApplicationMgr\n"
        "from Configurables import Gaudi__TestSuite__IntDataProducer as Producer\n"
        "from Configurables import Gaudi__TestSuite__IntToFloatData as Converter\n"
        "from Configurables import Gaudi__TestSuite__IntDataConsumer as Consumer\n"
        "ApplicationMgr(EvtSel='NONE', EvtMax=1, TopAlg=[\n"
        "    Producer('Producer', OutputLocation='/Event/MyInt'),\n"
        "    Converter('Converter', InputLocation='/Event/MyInt', OutputLocation='/Event/MyFloat'),\n"
        "    Consumer('Consumer', InputLocation='/Event/MyInt'),\n"
        "])\n",
    )
    components = {
        c["name"]: c for c in json.loads((tmp_path / "handles_components.json").read_text())["components"]
    }
    assert (components["Producer"]["inputs"], components["Producer"]["outputs"]) == ([], ["/Event/MyInt"])
    assert components["Converter"]["inputs"] == ["/Event/MyInt"]
    assert components["Converter"]["outputs"] == ["/Event/MyFloat"]
    assert components["Consumer"]["inputs"] == ["/Event/MyInt"]
    assert components["EventLoopMgr"]["inputs"] is None


# An algorithm putting three objects into the event store per event, which
# the store frees outside any audited call, in a sequencer.
_PUT_IN_A_SEQUENCER = (
    "from Configurables import ApplicationMgr, Gaudi__Sequencer, GaudiTesting__PutDataObjectAlg\n"
    "put = GaudiTesting__PutDataObjectAlg('Put', Paths=['A', 'B', 'C'])\n"
    "ApplicationMgr(EvtSel='NONE', EvtMax=3, TopAlg=[Gaudi__Sequencer('Top', Members=[put])])\n"
)


@pytest.mark.integration
@requires_gaudi
def test_auditor_counts_each_components_heap_allocations(built_plugin, tmp_path):
    """An algorithm putting three objects into the event store per event: it
    allocates them and keeps them past its call, since the store is cleared
    outside any audited call. The sequencer around it allocates nothing of its
    own once warmed up, although the auditor's bookkeeping for the nested call
    runs inside the sequencer's call, but its heap peak includes the
    algorithm's."""
    from k4bench.analysis.components import ALLOCATION_METRICS

    assert built_plugin.returncode == 0, "Skipped: build already failed"
    timing, _ = _run_audited_job(tmp_path, "put", _PUT_IN_A_SEQUENCER)

    assert set(ALLOCATION_METRICS) <= set(timing.execute)
    assert set(ALLOCATION_METRICS) <= set(timing.lifecycle["initialize"])
    allocations = timing.execute["allocations"]
    kept = timing.execute["net_allocated_bytes"]["Put"]
    peak = timing.execute["peak_heap_bytes"]
    largest = timing.execute["largest_allocation_bytes"]
    assert (allocations["Put"] >= 3).all()
    assert (kept > 0).all()
    assert (peak["Put"] >= kept).all()
    assert (allocations["Top"].iloc[1:] == 0).all()
    assert (peak["Top"] >= peak["Put"]).all()
    assert ((largest["Put"] > 0) & (largest["Put"] <= peak["Put"])).all()
    assert (largest["Top"] >= largest["Put"]).all()


@pytest.mark.integration
@requires_gaudi
def test_auditor_charges_each_component_the_heap_it_holds(built_plugin, tmp_path):
    """Recording heap owners, the algorithm holds its three objects at each
    event's end. The store frees them between events, outside any audited
    call, and they go back to the algorithm, so they are no longer its own at
    the next event's start, never pile up, and are gone once the job ends. The sequencer holds nothing during events
    beyond what it kept from initialize: what its member allocates is the
    member's. The components' heap adds up to what malloc counts as in use."""
    from k4bench.analysis.heap import NO_COMPONENT
    from k4bench.analysis.loader import load_component_heap

    assert built_plugin.returncode == 0, "Skipped: build already failed"
    timing, _ = _run_audited_job(tmp_path, "put", _PUT_IN_A_SEQUENCER, owners=True)
    heap = load_component_heap(tmp_path)["put"]

    start, held, peak = (heap.execute[m] for m in ("start_held_bytes", "held_bytes", "peak_held_bytes"))
    assert held.columns.tolist() == [NO_COMPONENT, *timing.components.index]
    assert held.index.tolist() == [0, 1, 2]
    added = held["Put"] - start["Put"]
    assert (added >= 3 * 64).all(), "three objects of at least 64 bytes each"
    # Freed by the store between events, they are no longer Put's when the
    # next event starts: it holds only what it keeps across events, as at the
    # job's end, and each event adds the same three objects again.
    kept = heap.at_end.loc["Put", "held_bytes"]
    assert start.loc[0, "Put"] == 0
    assert (start["Put"].iloc[1:] == kept).all()
    assert (added.iloc[1:] == added.iloc[1]).all()
    assert (held["Put"] == held["Put"].iloc[0]).all()
    assert (heap.working_set()["Put"] >= added).all()
    assert (peak["Put"] >= held["Put"]).all()
    # The sequencer keeps a block from its first call, as its own counts say,
    # and nothing more.
    first_call = timing.execute["net_allocated_bytes"].loc[0, "Top"]
    assert (held["Top"] - heap.before_first_event.loc["Top", "held_bytes"] == first_call).all()
    assert (peak["Top"] == held["Top"]).all()

    # malloc also counts the blocks allocated while the process was loading,
    # before the counter started, the auditor's own and the free ones it
    # caches per thread: 0.3% of the heap in this job, so 1% leaves room.
    moments = [
        (heap.before_first_event["held_bytes"].sum(), heap.process_before_first_event["malloc_in_use_bytes"]),
        *zip(held.sum(axis=1), heap.process["malloc_in_use_bytes"]),
        (heap.at_end["held_bytes"].sum(), heap.process_at_end["malloc_in_use_bytes"]),
    ]
    for components, in_use in moments:
        assert 0 <= in_use - components <= 0.01 * in_use
    assert (heap.process["rss_anon_mb"] > 0).all()
    assert heap.table_peak_entries > 0
    assert heap.table_mapped_bytes >= 2 * 16 * heap.table_peak_entries


@pytest.mark.integration
@requires_gaudi
def test_auditor_reports_heap_peaks_only_for_events_run_one_at_a_time(built_plugin, tmp_path):
    """Events run one at a time, even through several top-level algorithms,
    each have their heap peaks. With two events in flight at once, the
    owners' peaks are the process's, so no event has a peak of its own: they
    are missing, rather than one event's peak erased by the next opening."""
    from k4bench.analysis.loader import load_component_heap

    assert built_plugin.returncode == 0, "Skipped: build already failed"
    serial, concurrent = tmp_path / "serial", tmp_path / "concurrent"
    serial.mkdir()
    concurrent.mkdir()
    _run_audited_job(
        serial,
        "pair",
        "from Configurables import ApplicationMgr, GaudiTesting__SleepyAlg\n"
        "pair = [GaudiTesting__SleepyAlg(name, SleepTime=0) for name in ('First', 'Second')]\n"
        "ApplicationMgr(EvtSel='NONE', EvtMax=3, TopAlg=pair)\n",
        owners=True,
    )
    timing, _ = _run_audited_job(
        concurrent,
        "slots",
        "from Configurables import ApplicationMgr, AvalancheSchedulerSvc, HiveSlimEventLoopMgr\n"
        "from Configurables import HiveWhiteBoard, GaudiTesting__SleepyAlg\n"
        "slots = HiveWhiteBoard('EventDataSvc', EventSlots=2)\n"
        "AvalancheSchedulerSvc(ThreadPoolSize=2)\n"
        "loop = HiveSlimEventLoopMgr(SchedulerName='AvalancheSchedulerSvc')\n"
        "nap = GaudiTesting__SleepyAlg('Nap', SleepTime=1, Cardinality=2)\n"
        "ApplicationMgr(EvtSel='NONE', EvtMax=4, ExtSvc=[slots], EventLoop=loop, TopAlg=[nap])\n",
        owners=True,
    )

    assert load_component_heap(serial)["pair"].execute["peak_held_bytes"].notna().all().all()
    assert timing.threads == 2
    heap = load_component_heap(concurrent)["slots"]
    assert heap.execute["peak_held_bytes"].isna().all().all()
    assert heap.execute["held_bytes"].notna().all().all()
    assert heap.before_first_event["peak_held_bytes"].notna().all()


@pytest.mark.integration
@requires_gaudi
def test_without_heap_owners_the_auditor_writes_no_heap_file_and_counts_the_same(built_plugin, tmp_path):
    """Only a job recording heap owners writes a heap file, and recording them
    leaves the per-event allocation counts as they were: the counter's table
    never allocates through malloc."""
    import json

    from k4bench.analysis.components import ALLOCATION_METRICS

    assert built_plugin.returncode == 0, "Skipped: build already failed"
    runs = {}
    for owners in (False, True):
        run_dir = tmp_path / ("owners" if owners else "counting")
        run_dir.mkdir()
        _run_audited_job(run_dir, "put", _PUT_IN_A_SEQUENCER, owners=owners)
        runs[owners] = {
            kind: json.loads((run_dir / f"put_{kind}.json").read_text()) for kind in ("components", "events")
        }
        assert (run_dir / "put_heap.json").exists() == owners

    counting, owners = runs[False], runs[True]
    assert list(counting["components"]) == list(owners["components"])
    assert counting["components"]["components"] == owners["components"]["components"]
    for metric in ALLOCATION_METRICS:
        assert counting["components"]["execute"][metric] == owners["components"]["execute"][metric], metric
    assert list(counting["events"]) == list(owners["events"])


@pytest.mark.integration
@requires_gaudi
def test_auditor_counts_instructions_where_the_hardware_counters_are_readable(
    built_plugin, tmp_path
):
    """The same algorithm doing the same work every event executes nearly the
    same number of instructions once warmed up, whatever the host's load."""
    from k4bench.analysis.components import HARDWARE_METRICS

    assert built_plugin.returncode == 0, "Skipped: build already failed"
    timing, _ = _run_audited_job(
        tmp_path,
        "put",
        "from Configurables import ApplicationMgr, GaudiTesting__PutDataObjectAlg\n"
        "put = GaudiTesting__PutDataObjectAlg('Put', Paths=['A', 'B', 'C'])\n"
        "ApplicationMgr(EvtSel='NONE', EvtMax=4, TopAlg=[put])\n",
    )
    if not set(HARDWARE_METRICS) <= set(timing.execute):
        assert timing.hardware_counters_error, "a job without the counters says why"
        pytest.skip(
            f"the hardware performance counters cannot be read here ({timing.hardware_counters_error})"
        )

    assert timing.hardware_counters_error is None
    instructions = timing.execute["instructions"]["Put"]
    assert (instructions > 0).all()
    assert (timing.execute["cycles"]["Put"] > 0).all()
    warm = instructions.iloc[1:]
    assert warm.max() / warm.min() < 1.05


@pytest.mark.integration
@requires_gaudi
def test_auditor_dates_the_process_start_to_a_clock_tick(built_plugin, tmp_path):
    """The job's process started after the test launched it and before the
    auditor's first measured call, to within the clock tick /proc counts the
    start time in, so the configure phase before it is timed that closely."""
    import os
    import time

    assert built_plugin.returncode == 0, "Skipped: build already failed"
    launched_ns = time.time_ns()
    timing, _ = _run_audited_job(
        tmp_path,
        "put",
        "from Configurables import ApplicationMgr, GaudiTesting__PutDataObjectAlg\n"
        "ApplicationMgr(EvtSel='NONE', EvtMax=1, TopAlg=[GaudiTesting__PutDataObjectAlg('Put')])\n",
    )
    tick_ns = 1_000_000_000 // os.sysconf("SC_CLK_TCK")
    started_ns = timing.process_start_epoch_ns
    assert launched_ns - tick_ns <= started_ns <= timing.phases["initialize"][0]


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
