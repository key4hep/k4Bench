<p align="center">
  <a href="https://k4bench-dashboard.app.cern.ch/">
    <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1E90FF,100:00C9A7&height=130&section=header&text=k4Bench&fontSize=48&fontAlignY=40&fontColor=ffffff" />
    </a><br>
  <b>Detector-agnostic performance benchmarking for Key4hep simulations</b><br>
  👉 <a href="https://k4bench-dashboard.app.cern.ch/">Open the live dashboard</a> &nbsp;·&nbsp; 📖 <a href="https://key4hep.github.io/k4Bench/">Read the docs</a>
</p>

<br>

[![Release](https://img.shields.io/github/v/release/key4hep/k4Bench?include_prereleases)](https://github.com/key4hep/k4Bench/releases) [![Build status](https://img.shields.io/github/actions/workflow/status/key4hep/k4Bench/ci.yml?branch=main)](https://github.com/key4hep/k4Bench/actions/workflows/ci.yml?query=branch%3Amain) [![codecov](https://codecov.io/gh/key4hep/k4Bench/graph/badge.svg?token=oYOxyHkHuP)](https://codecov.io/gh/key4hep/k4Bench) [![DOI](https://zenodo.org/badge/1229933191.svg)](https://doi.org/10.5281/zenodo.20268042)

---

**k4Bench** measures *where the time and memory go* in DD4hep / Geant4 detector
simulations run through `ddsim` in the [Key4hep](https://key4hep.github.io/key4hep-doc/) stack.

Point it at any DD4hep compact geometry and it will tell you how long a
simulation takes, how much memory it needs, and — crucially — **which
subdetector is responsible**. It does this without you editing a single XML
file or recompiling anything.

## What it does

- ⚡ **Geometry sweeps** — automatically run a baseline, then re-run with each
  subdetector removed (or only a chosen subset kept) to measure each
  detector's cost. The original geometry is never touched.
- ⏱️ **Per-event & per-detector timing** — C++ Geant4 timing plugins record
  per-event wall time, RSS memory, and per-subdetector stepping time.
- 📊 **Analysis & dashboard** — load results into pandas, plot them with the
  bundled helpers, or browse historical trends across Key4hep releases on the
  [live dashboard](https://k4bench-dashboard.app.cern.ch/).
- 🔭 **Detector-agnostic** — works on any DD4hep compact XML. FCC-ee detectors
  (ALLEGRO, IDEA) are the worked examples and nightly-CI targets, not a limit.

## Quick start

The recommended install is from source, so the C++ timing plugins are built and
you get the full set of metrics:

```bash
# 1. Clone the repository
git clone https://github.com/key4hep/k4Bench.git
cd k4Bench

# 2. Setup setup.sh to source Key4hep, make a CVMFS-aware venv, install deps,
# build the timing plugins, and install pre-commit hooks.
source setup.sh

# 3. Install the k4bench command (editable)
pip install --no-build-isolation -e .

# 4. Benchmark a geometry (single particle-gun run)
k4bench --xml $K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml \
        --events 100 \
        --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"
```

Want to know each subdetector's cost? Add `--sweep`:

```bash
k4bench --xml ALLEGRO_o1_v03.xml --sweep \
        --ddsim-args="--enableGun --gun.particle e- --gun.distribution uniform"
```

Results print as a summary table and are written as CSV (plus per-event /
per-region JSON) under `logs/<geometry>/`.

> **Also on PyPI:** `pip install k4bench --no-deps` (inside Key4hep) gives you
> run-level metrics, but **not** the C++ timing plugins — so per-event and
> per-detector timing are unavailable. Installing from source is recommended.

## Analyse and plot the results

The bundled analysis helpers load a run directory into pandas and produce
ready-made Plotly figures:

```python
from k4bench.analysis import load_results, plot_run_overview

df = load_results("logs/ALLEGRO_o1_v03")        # one row per run
plot_run_overview("logs/ALLEGRO_o1_v03").show()  # bar charts across runs
```

## Documentation

Full documentation — installation, every CLI option, the sweep modes, the
timing plugins, the architecture, and the dashboard — lives at:

### 📖 **<https://key4hep.github.io/k4Bench/>**

| I want to… | Start here |
| --- | --- |
| Install and run my first benchmark | [Getting started](https://key4hep.github.io/k4Bench/getting-started/installation/) |
| Understand sweep modes & options | [User guide](https://key4hep.github.io/k4Bench/user-guide/overview/) |
| Understand how it works | [Architecture](https://key4hep.github.io/k4Bench/architecture/overview/) |

## License

Distributed under the terms of the [LICENSE](LICENSE) in this repository.
