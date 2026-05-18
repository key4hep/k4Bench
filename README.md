# DD4bench

[![Release](https://img.shields.io/github/v/release/jbeirer/DD4bench?include_prereleases)](https://github.com/jbeirer/DD4bench/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/jbeirer/DD4bench/ci.yml?branch=main)](https://github.com/jbeirer/DD4bench/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/jbeirer/DD4bench/graph/badge.svg?token=oYOxyHkHuP)](https://codecov.io/gh/jbeirer/DD4bench)
[![DOI](https://zenodo.org/badge/1229933191.svg)](https://doi.org/10.5281/zenodo.20268041)

Performance benchmarking for DD4hep-based simulations and reconstruction in Key4hep.

## Installation inside key4hep environment

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-04-08

# Create a local virtualenv that inherits Key4hep packages
python -m venv ~/.venvs/dd4bench --system-site-packages

# Activate it
source ~/.venvs/dd4bench/bin/activate

# Install dd4bench
pip install dd4bench --no-deps
```

## Basic Usage

```python
from dd4bench import benchmark

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
