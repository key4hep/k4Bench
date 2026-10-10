# Repository layout

An annotated map of the repository, so you know where things live before you
change them.

```text
k4Bench/
├── k4bench/                  # the installable Python package
│   ├── __init__.py           #   version (via importlib.metadata / setuptools-scm)
│   ├── cli.py                #   argparse CLI → BenchmarkConfig; the `k4bench` entry point
│   ├── artifacts.py          #   file names of a run directory's artifacts (data contract)
│   ├── benchmark/
│   │   ├── ddsim.py          #   orchestrator: BenchmarkConfig, SweepMode, select_sweep, run_sweep
│   │   └── k4run.py          #   orchestrator: K4runConfig, staging, variants, run_k4run_benchmark
│   ├── geometry/
│   │   ├── errors.py         #   geometry exception hierarchy
│   │   ├── index.py          #   immutable include/detector/plugin structure
│   │   ├── references.py     #   shared DD4hep filesystem/document-ref rules
│   │   ├── scanner.py        #   lenient discovery façade
│   │   └── patcher.py        #   one validated detector-removal engine
│   ├── runner/
│   │   ├── ddsim.py          #   run_ddsim: ddsim command line, DDG4 plugin wiring
│   │   ├── k4run.py          #   run_k4run: k4run command line, auditor wiring, working dir
│   │   ├── result.py         #   run_result: timed process + event JSON → RunResult
│   │   ├── process.py        #   run_timed: any command under time -v, log streaming, Ctrl-C
│   │   └── parser.py         #   parse_time_output
│   ├── results/
│   │   ├── model.py          #   RunResult dataclass
│   │   └── reporter.py       #   print_summary, save_csv
│   ├── plugin/
│   │   ├── runtime.py        #   locate/build C++ plugins, set env vars
│   │   └── schema.py         #   schema_version contracts of the plugins' JSON files
│   └── analysis/
│       ├── loader.py         #   load_results, load_event_timing, load_region_timing
│       └── plots/            #   Plotly figures (overview, event, region) + theme/utils
│
├── plugin/                   # C++ DDG4 timing plugins and Gaudi auditor (built, not pip-installed)
│   ├── k4BenchTimingAction.cpp        #   per-event wall time + RSS
│   ├── k4BenchRegionTimingAction.cpp  #   per-detector stepping time (3 actions, 1 .so)
│   ├── auditor/                       #   k4run: k4BenchAuditor + its preloaded allocation counter
│   ├── CMakeLists.txt
│   └── build.sh              #   idempotent build helper
│
├── dashboard/                # Streamlit app (separate from the package)
│   ├── app.py                #   page layout + tab wiring
│   ├── config.py             #   Config.from_env (K4BENCH_DATA_DIR/_URL/_CACHE_DIR)
│   ├── data.py               #   @st.cache_data loaders + trend aggregation
│   ├── remote.py             #   WebEOS discovery + atomic immutable run cache
│   ├── remote_cache.py       #   Streamlit-cached wrappers around remote.py
│   ├── stats.py              #   summary-statistics tables
│   ├── trend_window.py       #   pure window-resolution logic (unit-tested)
│   ├── ui_chrome.py / ui_utils.py
│   ├── tabs/                 #   one module per dashboard tab
│   ├── Dockerfile
│   └── requirements.txt
│
├── openshift/                # CERN PaaS manifests (Deployment/Service/Route/PVC)
│
├── .github/
│   ├── workflows/            #   ci.yml, nightly.yml, benchmark-detector.yml,
│   │                         #   deploy-dashboard.yml, on-release-main.yml, docs.yml
│   ├── benchmarks/           #   *.yml nightly benchmark configs (one per detector)
│   └── scripts/              #   nightly_benchmark.sh, benchmark_job.py, list_benchmarks.py,
│                             #   machine_info.py, run_info.py, …
│
├── tests/
│   ├── conftest.py           #   matplotlib Agg backend
│   ├── fixtures/             #   minimal_geometry/, time_output.txt
│   ├── unit/                 #   pure-python tests (no ddsim)
│   └── integration/          #   real ddsim + plugin build (marked `integration`)
│
├── docs/                     # this documentation (MkDocs Material)
│   ├── requirements.txt      #   docs build deps
│   └── gen_ref_pages.py      #   auto-generates the API reference
│
├── JupyterNotebooks/         # analysis.ipynb
├── pyproject.toml            # package metadata, deps, pytest/ruff config
├── setup.sh                  # dev environment bootstrap
├── mkdocs.yml                # docs site config
└── requirements.txt          # dev tooling only (codespell, pre-commit)
```

## Mental shortcuts

- **"Where does the CLI turn flags into behaviour?"** → `cli.py` then
  `benchmark/ddsim.py` (`run_sweep`).
- **"Where does a number come from?"** → `runner/parser.py` parses it,
  `results/model.py` stores it, `results/reporter.py` prints/saves it.
- **"Where is the geometry magic?"** → `geometry/index.py` + `geometry/patcher.py`.
- **"Where does ddsim actually get run?"** → `runner/ddsim.py` builds the
  command; `runner/process.py` runs and times it.
- **"Where does a k4run job get run?"** → `benchmark/k4run.py` stages its
  working directory, `runner/k4run.py` builds the command with the auditor's
  options file, and `runner/process.py` runs and times it.
- **"Where do per-event/-detector numbers come from?"** → `plugin/*.cpp`, wired
  by `k4bench/plugin/runtime.py`.

## What's installable vs not

| Directory | Shipped to PyPI? | How it's used |
| --- | --- | --- |
| `k4bench/` | ✅ yes | `pip install k4bench` |
| `plugin/` | ❌ no | built from a source checkout (`build.sh`) |
| `dashboard/` | ❌ no | containerised separately (`Dockerfile`) |
| `.github/`, `tests/`, `docs/` | ❌ no | dev / CI only |

The package include list in `pyproject.toml` is `include = ["k4bench*"]`, so only
the Python package is packaged.

!!! note "Repository artefacts you can ignore"
    A stale `dd4bench.egg-info/` and an empty `?/` directory exist from the
    pre-rename history; they are not part of the build. `run.sh` is a personal
    scratch file (and still references the old `dd4bench` command). None of these
    are documented as features.

## See also

- [Development setup](development-setup.md) — getting a working dev environment.
- [Architecture overview](../architecture/overview.md) — how these pieces relate.
