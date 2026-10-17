# Reconstruction run fixture

One nightly k4run run, as the nightly uploaded it to
`runs/reco/CLD/CLD_o2_v09/x86_64-el9-gcc16-opt/key4hep-2026-10-17/p8_ee_Zbb_ecm91/2026-10-17/`:
CLD_o2_v09 reconstruction of 100 frozen `p8_ee_Zbb_ecm91` events, labels `baseline`
and `variant_truth_tracking`.

The `_events.json`, `_results.csv` and `machine_info.json` files are copied
unchanged. `run_info.json` keeps only the run's identity and k4run fields (the
package lists are dropped), and each `.log` is an excerpt: its first lines, the
first `ERROR` lines and its `/usr/bin/time` tail.
