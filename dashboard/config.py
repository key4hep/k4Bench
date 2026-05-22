from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str

    @classmethod
    def from_env(cls) -> Config:
        return cls(data_dir=os.environ.get("DD4BENCH_DATA_DIR", "logs"))
