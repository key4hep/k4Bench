# Reconstruction nights fixture

Three nightly k4run runs of CLD_o2_v09 reconstructing the same frozen 100
`p8_ee_Zbb_ecm91` events, cut down from
`runs/reco/CLD/CLD_o2_v09/x86_64-el9-gcc16-opt/key4hep-{night}/p8_ee_Zbb_ecm91/{night}/`:

- `2026-10-14` — fcc-ironic-03, the night the host ran the same work slower:
  identical instructions, instructions per cycle 1.36 → 1.00, tracking ~+40%.
- `2026-10-16`, `2026-10-17` — fcc-ironic-01, the same software twice.
  `2026-10-17` also holds the `variant_truth_tracking` label, which replaces
  `MyConformalTracking` and `ClonesAndSplitTracksFinder` by `MyTruthTrackFinder`.

Each `_components.json` keeps the first 12 events (0–11), the metrics
`wall_s`, `cpu_s`, `peak_rss_increase_mb`, `instructions`, `cycles` and
`allocations`, the `initialize` lifecycle and phases, and each component's
library file name only. Each `_joboptions.opts` keeps a few of the dump's
lines, among them k4Bench's own settings, a Marlin `Parameters` mapping and
paths below the stack and k4Bench's working directories. `run_info.json` and
`machine_info.json` keep the fields a comparison reads.
