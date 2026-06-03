"""k4Bench — performance benchmarking for DD4hep-based simulations and reconstruction in Key4hep."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("k4bench")
except PackageNotFoundError:
    __version__ = "unknown"