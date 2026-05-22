from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str
    # When set, the dashboard fetches run data from this WebEOS HTTPS base URL
    # instead of reading from a local directory.
    # Example: https://dd4bench-data.web.cern.ch
    data_url: str | None

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            data_dir=os.environ.get("DD4BENCH_DATA_DIR", "logs"),
            data_url=os.environ.get("DD4BENCH_DATA_URL"),
        )
