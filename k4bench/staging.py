"""Where the nightly copies a k4run job's inputs and data files before it runs.

The nightly job (``.github/scripts/benchmark_job.py``) copies every input and
data file of a k4run job to a local path and passes that path to the job, so
the ``k4run_args`` a run records name these paths. A reproducer that fetches
the same sources to the same paths (:mod:`k4bench.blame.reproduce`) can
therefore run the recorded arguments as they are. Like :mod:`k4bench.artifacts`
this is a leaf module, so the nightly and the blame tooling share one spelling
without importing each other.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Where a k4run job's inputs are copied, so the job never times a network read.
LOCAL_INPUT_DIR = Path("/tmp/k4bench-inputs")
#: Where a k4run job's data files are gathered, all in one directory.
LOCAL_DATA_DIR = Path("/tmp/k4bench-data")
#: Data files with these suffixes are archives, unpacked where they land.
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz")
#: Sources fetched over the network; anything else is a local path.
REMOTE_PREFIXES = ("root://", "http://", "https://")


def input_key(source: str) -> str:
    """The directory below :data:`LOCAL_INPUT_DIR` one input is copied into.

    Named by a hash of the URL, so two inputs that share a file name never
    overwrite each other."""
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def data_key(sources: list[str]) -> str:
    """The directory below :data:`LOCAL_DATA_DIR` a job's data files share,
    named by a hash of all of them."""
    return hashlib.sha256(" ".join(sources).encode()).hexdigest()[:16]


def download_argv(source: str, local: str) -> list[str]:
    """The command copying the ``root://`` (xrdcp) or ``http(s)://`` (curl)
    *source* to *local*."""
    if source.startswith("root://"):
        return ["xrdcp", "--force", source, local]
    return ["curl", "-fsSL", "--retry", "3", "--retry-delay", "10", "-o", local, source]
