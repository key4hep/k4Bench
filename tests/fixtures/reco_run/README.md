# Reconstruction run fixture

One nightly k4run run, as the nightly uploaded it to
`runs/reco/CLD/CLD_o2_v09/x86_64-el9-gcc16-opt/key4hep-2026-10-17/p8_ee_Zbb_ecm91/2026-10-17/`:
CLD_o2_v09 reconstruction of 100 frozen `p8_ee_Zbb_ecm91` events, labels `baseline`
and `variant_truth_tracking`.

The `_events.json`, `_results.csv` and `machine_info.json` files are copied
unchanged. `run_info.json` keeps only the run's identity and k4run fields (the
package lists are dropped), and each `.log` is an excerpt: its first lines, the
first `ERROR` lines and its `/usr/bin/time` tail.

Each `_components.json` keeps the first 12 events (0–11), the metrics
`wall_s`, `cpu_s`, `peak_rss_increase_mb`, `instructions`, `cycles` and
`allocations`, the `initialize` lifecycle, the `initialize` and `execute`
phases, and each component's library file name only. Each `_joboptions.opts`
keeps a few lines of the dump, among them k4Bench's own settings and a Marlin
`Parameters` mapping.

Each `_dataflow.json` was built with `k4bench.analysis.dataflow.build_dataflow`
from the run's full `_components.json` and `_joboptions.opts` and `Marlin -x` of
its stack (the run predates both, which were backfilled). That auditor did not
record data handles yet, so the Reader's outputs are the input file's
collections, passed as `reader_collections`.
