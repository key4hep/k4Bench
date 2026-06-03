<p align="center">
  <a href="https://k4bench-dashboard.app.cern.ch/">
    <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1E90FF,100:00C9A7&height=130&section=header&text=k4Bench&fontSize=48&fontAlignY=40&fontColor=ffffff" />
    </a><br>
  <b>Explore runs, metrics, and performance at a glance</b><br>
  👉 <a href="https://k4bench-dashboard.app.cern.ch/">Open Dashboard</a>
</p>

<br>

[![Release](https://img.shields.io/github/v/release/key4hep/k4Bench?include_prereleases)](https://github.com/key4hep/k4Bench/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/key4hep/k4Bench/ci.yml?branch=main)](https://github.com/key4hep/k4Bench/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/key4hep/k4Bench/graph/badge.svg?token=oYOxyHkHuP)](https://codecov.io/gh/key4hep/k4Bench)
[![DOI](https://zenodo.org/badge/1229933191.svg)](https://doi.org/10.5281/zenodo.20268042)


Performance benchmarking for DD4hep-based simulations and reconstruction in Key4hep.

## Installation inside key4hep environment

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-04-08

# Create a local virtualenv that inherits Key4hep packages
python -m venv ~/.venvs/k4bench --system-site-packages

# Activate it
source ~/.venvs/k4bench/bin/activate

# Install k4bench
pip install k4bench --no-deps
```

## Basic Usage

```python
from k4bench import benchmark

result = benchmark.ddsim(
    compact_file="ALLEGRO_o1_v03.xml",
    ddsim_args={
        "numberOfEvents": 100,
        "gun.particle": "e-",
        "gun.distribution": "uniform",
    },
    scan=True,
)
```
