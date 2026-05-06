# DD4bench

Performance benchmarking for DD4hep-based simulations and reconstruction in Key4hep.

## Installation

```bash
pip install dd4bench
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