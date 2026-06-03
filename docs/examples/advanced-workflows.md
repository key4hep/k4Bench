# Advanced workflows

Realistic scenarios beyond the basics: physics events, steering files, full-sim
detectors, measurement stabilisation, and cross-release comparison. These mirror
what the nightly CI does — see `.github/scripts/nightly_benchmark.sh` and
`.github/benchmarks/*.yml` for the production versions.

## Benchmark with real physics events (HepMC)

Instead of a particle gun, feed generated events. The HepMC file is passed
through `--ddsim-args` (there's no dedicated flag).

```bash
# HepMC text files can't stream over xrootd — fetch locally first.
xrdcp root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/hepmc/p8_ee_Zbb_ecm91/events_noVtxSmear.hepmc \
      /tmp/zbb.hepmc

k4bench --xml $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml \
        --events 100 \
        --ddsim-args="--inputFiles /tmp/zbb.hepmc"
```

!!! warning "Local path, and no `--enableGun`"
    Streaming HepMC over xrootd causes a ROOT mis-parse → SIGSEGV, so always
    `xrdcp` it local first. And `--inputFiles` is mutually exclusive with
    `--enableGun`.

## Use a ddsim steering file

Full-sim detectors (notably IDEA) configure physics through a `--steeringFile`.
Environment variables like `$FCCCONFIG` are expanded by your shell:

```bash
k4bench --xml $K4GEO/FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml \
        --events 100 \
        --ddsim-args="--steeringFile $FCCCONFIG/FullSim/IDEA/IDEA_o1_v03/SteeringFile_IDEA_o1_v03.py \
                      --random.enableEventSeed --random.seed 42"
```

!!! note "Steering files can override actions"
    If a steering file sets the DDG4 action list, it may replace the k4Bench
    region actions, so `_regions.json` won't be written. The event action is
    injected separately and usually survives. See
    [Timing plugins](../user-guide/features/timing-plugins.md#using-your-own-actions).

## Tweak a steering file on the fly

To benchmark a variant (e.g. calorimeter off) without editing the original:

```bash
sed 's/^simulateCalo = True/simulateCalo = False/' \
    "$FCCCONFIG/FullSim/IDEA/IDEA_o1_v03/SteeringFile_IDEA_o1_v03.py" \
    > /tmp/IDEA_noCalo.py

k4bench --xml $K4GEO/FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml \
        --events 100 \
        --ddsim-args="--steeringFile /tmp/IDEA_noCalo.py --enableGun --gun.particle e- --gun.energy '10*GeV'"
```

## Stabilise measurements with CPU pinning

Background processes and CPU migration add noise. Pin the whole run to a fixed
CPU set with `taskset` (this is exactly what the nightly does via `RUNNER_CPU_SET`):

```bash
taskset -c 0-7 k4bench --xml ALLEGRO_o1_v03.xml --sweep \
        --ddsim-args="--enableGun --gun.particle e-"
```

Combine with a fixed seed and a quiet machine for the most reproducible numbers.
See the [FAQ on noise](../faq.md#why-do-my-numbers-vary-between-runs).

## Full IDEA full-sim sweep

A complete IDEA configuration with steering file and seed, swept per detector:

```bash
k4bench --xml $K4GEO/FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml \
        --sweep --events 100 \
        --ddsim-args="--steeringFile $FCCCONFIG/FullSim/IDEA/IDEA_o1_v03/SteeringFile_IDEA_o1_v03.py \
                      --enableGun --gun.distribution uniform --gun.energy '10*GeV' \
                      --gun.particle e- --random.enableEventSeed --random.seed 42"
```

## Compare two Key4hep releases

Run the same benchmark under two stacks, then diff. The cleanest way is two
shells (or two `setup.sh` invocations with different `KEY4HEP_VERSION`), writing
to release-tagged directories:

```bash
# Shell A
KEY4HEP_VERSION=2026-03-10 source setup.sh
k4bench --xml ALLEGRO_o1_v03.xml --sweep --output-dir logs/rel_2026-03-10 \
        --ddsim-args="--enableGun --gun.particle e-"

# Shell B
KEY4HEP_VERSION=2026-04-08 source setup.sh
k4bench --xml ALLEGRO_o1_v03.xml --sweep --output-dir logs/rel_2026-04-08 \
        --ddsim-args="--enableGun --gun.particle e-"
```

```python
from k4bench.analysis import load_results
a = load_results("logs/rel_2026-03-10").set_index("label")["wall_time_s"]
b = load_results("logs/rel_2026-04-08").set_index("label")["wall_time_s"]
print((b - a).sort_values())   # negative = faster in the newer release
```

For continuous cross-release tracking, the
[dashboard Trends tab](../user-guide/features/dashboard.md#trends-tab) does this
automatically on nightly data — no manual runs needed.

## Drive a parameter scan from Python

When you want to vary, say, gun energy programmatically:

```python
from pathlib import Path
from k4bench.benchmark.ddsim import BenchmarkConfig, SweepMode, run_sweep

for energy in ["1*GeV", "10*GeV", "100*GeV"]:
    cfg = BenchmarkConfig(
        xml_path=Path("ALLEGRO_o1_v03.xml"),
        n_events=200,
        output_file=Path("/tmp/out.edm4hep.root"),
        log_dir=Path(f"logs/scan_{energy.replace('*', '')}"),
        mode=SweepMode.BASELINE,
        extra_args=["--enableGun", "--gun.particle", "e-", "--gun.energy", energy],
    )
    run_sweep(cfg)
```

## See also

- [Common workflows](common-workflows.md)
- [Configuration](../user-guide/configuration.md) — passthrough rules.
- [Dashboard](../user-guide/features/dashboard.md) — historical comparison.
