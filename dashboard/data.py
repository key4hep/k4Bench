from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd
import streamlit as st

from k4bench.analysis.components import ComponentTiming
from k4bench.analysis.dataflow import DataFlow, load_dataflow
from k4bench.analysis.event_size import OutputFile, load_event_size
from k4bench.analysis.event_source import EventSource, load_event_source
from k4bench.analysis.heap import ComponentHeap
from k4bench.analysis.loader import judgeable_config_data, load_each_component_heap, load_each_component_timing
from k4bench.analysis.loader import load_component_heap, load_component_timing, load_event_timing, load_region_timing, load_results  # noqa: F401 (re-exported)
from k4bench.analysis import trend as _trend
from k4bench.analysis.trend import EXPECTED_LOAD_ERRORS as _EXPECTED_ERRORS
from k4bench.analysis.trend import parse_run_dir as _parse_run_dir  # noqa: F401 (re-exported)
from k4bench.artifacts import (
    COMPONENTS_SUFFIX,
    DATAFLOW_SUFFIX,
    EVENT_SIZE_SUFFIX,
    EVENT_SOURCE,
    EVENTS_SUFFIX,
    HEAP_SUFFIX,
    MACHINE_INFO,
    REGIONS_SUFFIX,
    RESULTS_SUFFIX,
    RUN_INFO,
    matching,
)
from k4bench.provenance.sources import load_run_packages
from k4bench.provenance.stack import STACK_MANIFESTS

_log = logging.getLogger(__name__)

#: Some of a run's files: their names, or patterns over them
#: (:func:`k4bench.artifacts.matching`).
Artifacts = tuple[str, ...]

# What each loader below reads from a run directory. In remote mode a view
# downloads these files of the runs it draws before it loads them
# (:attr:`RunData.directory`, ``sidebar.TrendWindow.run_dirs``), and every
# cached load is keyed by which of them its directories hold (:func:`_keyed_on`).
#: What every view of the selected run reads — its metadata, host and
#: results — downloaded with it; of the window's runs, what the trend views'
#: results and reliability are built from.
RUN_BASE: Artifacts = (RUN_INFO, MACHINE_INFO, f"*{RESULTS_SUFFIX}")
RESULTS: Artifacts = (RUN_INFO, f"*{RESULTS_SUFFIX}")
MACHINE: Artifacts = (RUN_INFO, MACHINE_INFO)
EVENTS: Artifacts = (RUN_INFO, f"*{EVENTS_SUFFIX}")
REGIONS: Artifacts = (RUN_INFO, f"*{REGIONS_SUFFIX}")
COMPONENTS: Artifacts = (RUN_INFO, f"*{COMPONENTS_SUFFIX}")
HEAP: Artifacts = (RUN_INFO, f"*{HEAP_SUFFIX}")
DATAFLOW: Artifacts = (f"*{DATAFLOW_SUFFIX}",)
EVENT_SIZE: Artifacts = (RUN_INFO, f"*{EVENT_SIZE_SUFFIX}")
EVENT_SOURCE_FILES: Artifacts = (EVENT_SOURCE,)
#: The stack's packages and, from its manifests, where each was installed.
PACKAGES: Artifacts = (RUN_INFO, *STACK_MANIFESTS)


def _keyed_on(artifacts: Artifacts) -> dict:
    """``hash_funcs`` keying a cached load by which files of *artifacts* its
    run directories hold, besides the directories themselves.

    A remote run's files arrive one view at a time, into the same directory,
    and never change once there; so the names present say what the load
    reads. Keyed by the path alone, a load made before its files arrived
    would be served after they had.
    """
    def key(run_dir: str) -> bytes:
        try:
            names = sorted(matching(os.listdir(run_dir), artifacts))
        except OSError:
            names = []
        # Bytes, which Streamlit takes as they are: a str would come back here.
        return "\0".join([run_dir, *names]).encode()
    return {str: key}


@st.cache_data(show_spinner="Loading benchmark results...", hash_funcs=_keyed_on(RESULTS))
def cached_load_results(data_dir: str) -> pd.DataFrame | None:
    try:
        df = load_results(data_dir)
        return df if not df.empty else None
    except _EXPECTED_ERRORS:
        return None
    except Exception:
        _log.exception("cached_load_results: unexpected error loading '%s'", data_dir)
        st.warning(f"Unexpected error loading benchmark results from '{data_dir}'. Check logs for details.")
        return None


@st.cache_data(show_spinner="Loading event timing...", hash_funcs=_keyed_on(EVENTS))
def cached_load_event_timing(data_dir: str) -> dict | None:
    try:
        d = load_event_timing(data_dir)
        return d if d else None
    except _EXPECTED_ERRORS:
        return None
    except Exception:
        _log.exception("cached_load_event_timing: unexpected error loading '%s'", data_dir)
        st.warning(f"Unexpected error loading event timing from '{data_dir}'. Check logs for details.")
        return None


@st.cache_data(show_spinner="Loading region timing...", hash_funcs=_keyed_on(REGIONS))
def cached_load_region_timing(data_dir: str) -> dict | None:
    try:
        d = load_region_timing(data_dir)
        return d if d else None
    except _EXPECTED_ERRORS:
        return None
    except Exception:
        _log.exception("cached_load_region_timing: unexpected error loading '%s'", data_dir)
        st.warning(f"Unexpected error loading region timing from '{data_dir}'. Check logs for details.")
        return None


# Each run's matrices are ~1 MB pickled; the bound keeps a long session's
# comparisons from piling up in the pod's memory.
@st.cache_data(
    show_spinner="Loading component costs...", max_entries=16, hash_funcs=_keyed_on(COMPONENTS),
)
def cached_load_component_timing(
    data_dir: str,
) -> tuple[dict[str, ComponentTiming], dict[str, str]]:
    """Every label's ``_components.json`` in *data_dir* that could be read,
    and why each other could not
    (:func:`k4bench.analysis.loader.load_each_component_timing`): one that is
    malformed hides no other configuration. Both empty when it has none."""
    return load_each_component_timing(data_dir)


# A long single-electron run's heap file is 1.5 MB of JSON; the bound keeps a
# session's runs from piling up in the pod's memory, as for component costs.
@st.cache_data(
    show_spinner="Loading component heap...", max_entries=16, hash_funcs=_keyed_on(HEAP),
)
def cached_load_component_heap(data_dir: str) -> tuple[dict[str, ComponentHeap], dict[str, str]]:
    """Every label's ``_heap.json`` in *data_dir* that could be read, and why
    each other could not
    (:func:`k4bench.analysis.loader.load_each_component_heap`): one that is
    malformed hides no other configuration. Both empty when it has none."""
    return load_each_component_heap(data_dir)


@st.cache_data(show_spinner=False, max_entries=16, hash_funcs=_keyed_on(PACKAGES))
def cached_load_run_packages(data_dir: str) -> dict[str, dict]:
    """The stack packages of the run in *data_dir*; empty when it recorded none."""
    return load_run_packages(data_dir)


@st.cache_data(show_spinner=False, max_entries=16, hash_funcs=_keyed_on(DATAFLOW))
def cached_load_dataflow(data_dir: str) -> dict[str, DataFlow]:
    """Every label's ``_dataflow.json`` in *data_dir*; empty when it has none."""
    try:
        return load_dataflow(data_dir)
    except _EXPECTED_ERRORS:
        return {}
    except Exception:
        _log.exception("cached_load_dataflow: unexpected error loading '%s'", data_dir)
        st.warning(f"Unexpected error loading the data flow from '{data_dir}'. Check logs for details.")
        return {}


@st.cache_data(show_spinner=False, max_entries=16, hash_funcs=_keyed_on(EVENT_SIZE))
def cached_load_event_size(
    data_dir: str,
) -> tuple[dict[str, tuple[OutputFile, ...]], dict[str, str]]:
    """Every label's output files in *data_dir* that could be read, and why
    each other could not (:func:`k4bench.analysis.event_size.load_event_size`)."""
    try:
        return load_event_size(data_dir)
    except Exception:
        _log.exception("cached_load_event_size: unexpected error loading '%s'", data_dir)
        st.warning(f"Unexpected error loading event sizes from '{data_dir}'. Check logs for details.")
        return {}, {}


# A long single-electron run's source holds ~1300 events of particles; the
# bound keeps a session's runs from piling up, as for component costs.
@st.cache_data(show_spinner=False, max_entries=16, hash_funcs=_keyed_on(EVENT_SOURCE_FILES))
def cached_load_event_source(data_dir: str) -> tuple[EventSource | None, str | None]:
    """The run's ``event_source.json.gz`` and why it could not be read:
    ``(None, None)`` for a run without one
    (:func:`k4bench.analysis.event_source.load_event_source`)."""
    try:
        return load_event_source(data_dir), None
    except (OSError, ValueError) as exc:
        return None, str(exc)
    except Exception:
        _log.exception("cached_load_event_source: unexpected error loading '%s'", data_dir)
        return None, "unexpected error, see the logs"


def run_metadata(run_dir: str) -> dict:
    """Return metadata for a single run (date) directory.

    Thin public wrapper over :func:`k4bench.analysis.trend.parse_run_dir` for
    callers that already hold one concrete run directory (e.g. the sidebar's
    selected run, for single-run tabs).
    """
    return _parse_run_dir(Path(run_dir))


# The trend loaders below are thin ``@st.cache_data`` shims over the pure
# builders in ``k4bench.analysis.trend`` (shared with the nightly regression
# report, which must run without Streamlit). See that module for behaviour docs.

@st.cache_data(show_spinner="Loading trend data...", ttl=3600, hash_funcs=_keyed_on(RESULTS))
def cached_load_trend_results(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_results_trend` for the Trends tab."""
    return _trend.build_results_trend(run_dirs)


@st.cache_data(
    show_spinner="Loading region timing trends...", ttl=3600, hash_funcs=_keyed_on(REGIONS),
)
def cached_load_trend_region_timing(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_region_timing_trend`."""
    return _trend.build_region_timing_trend(run_dirs)


@st.cache_data(
    show_spinner="Loading job phase trends...", ttl=3600, hash_funcs=_keyed_on(COMPONENTS),
)
def cached_load_trend_phases(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_phase_trend`."""
    return _trend.build_phase_trend(run_dirs)


@st.cache_data(
    show_spinner="Loading algorithm timing trends...", ttl=3600,
    hash_funcs=_keyed_on(COMPONENTS),
)
def cached_load_trend_algorithm_timing(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_algorithm_timing_trend`."""
    return _trend.build_algorithm_timing_trend(run_dirs)


@st.cache_data(
    show_spinner="Loading component history...", ttl=3600, hash_funcs=_keyed_on(COMPONENTS),
)
def cached_load_trend_component_summary(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_component_summary_trend`."""
    return _trend.build_component_summary_trend(run_dirs)


@st.cache_data(show_spinner="Loading heap trends...", ttl=3600, hash_funcs=_keyed_on(HEAP))
def cached_load_trend_heap(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_heap_trend`."""
    return _trend.build_heap_trend(run_dirs)


@st.cache_data(
    show_spinner="Loading event size trends...", ttl=3600, hash_funcs=_keyed_on(EVENT_SIZE),
)
def cached_load_trend_event_size(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_event_size_trend`."""
    return _trend.build_event_size_trend(run_dirs)


@st.cache_data(
    show_spinner="Loading event timing trends...", ttl=3600, hash_funcs=_keyed_on(EVENTS),
)
def cached_load_trend_event_timing(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_event_timing_trend`."""
    return _trend.build_event_timing_trend(run_dirs)


@st.cache_data(
    show_spinner="Loading machine info trends...", ttl=3600, hash_funcs=_keyed_on(MACHINE),
)
def cached_load_trend_machine_info(run_dirs: tuple[str, ...]) -> pd.DataFrame | None:
    """Cached :func:`k4bench.analysis.trend.build_machine_info_trend`."""
    return _trend.build_machine_info_trend(run_dirs)


@st.cache_data(show_spinner=False, ttl=60, hash_funcs=_keyed_on((MACHINE_INFO,)))
def load_machine_info(run_dir: str) -> dict | None:
    """Load ``machine_info.json`` from a run directory, or return ``None`` if absent."""
    return _trend.load_machine_info(run_dir)


@dataclass(frozen=True)
class RunData:
    """The selected run: its results, read with it, and the rest of what its
    views draw, read — and in remote mode downloaded — when one first asks."""

    results: pd.DataFrame | None
    #: The run's directory once it holds the files of the given artifacts,
    #: which in remote mode are downloaded first; raises
    #: :class:`remote_cache.FilesUnavailable` when they could not be.
    directory: Callable[[Artifacts], str]
    #: Whether the page reads the run's per-component costs (k4run runs have
    #: them, ddsim runs none).
    reads_components: bool = False

    @property
    def labels(self) -> list[str]:
        """Every configuration the run's results name, sorted."""
        if self.results is None or "label" not in self.results.columns:
            return []
        return sorted(self.results["label"].dropna().unique())

    # Current-run event/region files can be partial leftovers from a crashed
    # config, or orphans with no result row at all. These views present their
    # inputs as ordinary evidence (unlike regression plots, which can mark a
    # failed point explicitly), so only successful result-backed labels
    # belong here.
    @cached_property
    def event_data(self) -> dict | None:
        """Each successful configuration's per-event timing and memory."""
        return judgeable_config_data(
            cached_load_event_timing(self.directory(EVENTS)), self.results,
        )

    @cached_property
    def region_data(self) -> dict | None:
        """Each successful configuration's per-region timing."""
        return judgeable_config_data(
            cached_load_region_timing(self.directory(REGIONS)), self.results,
        )

    @cached_property
    def _component_costs(self) -> tuple[dict[str, ComponentTiming], dict[str, str]]:
        if not self.reads_components:
            return {}, {}
        costs, unreadable = cached_load_component_timing(self.directory(COMPONENTS))
        # A failed configuration is not offered whatever its file holds, so
        # why its file could not be read is no news.
        return (
            judgeable_config_data(costs, self.results) or {},
            judgeable_config_data(unreadable, self.results) or {},
        )

    @property
    def components(self) -> dict[str, ComponentTiming]:
        """Each successful configuration's per-component costs; empty for a
        page that does not read them."""
        return self._component_costs[0]

    @property
    def unreadable_components(self) -> dict[str, str]:
        """Why each successful configuration's ``_components.json`` could not
        be read, by label; such a configuration is absent from :attr:`components`."""
        return self._component_costs[1]


def load_run(
    data_dir: str,
    directory: Callable[[Artifacts], str] | None = None,
    *,
    components: bool = False,
) -> RunData:
    """The run in *data_dir*, its results read; the rest of its files are
    read as its views ask for them, with its per-component costs when
    *components*. *directory* downloads them (:attr:`RunData.directory`);
    without one, *data_dir* holds them already, as a local run does."""
    return RunData(
        cached_load_results(data_dir), directory or (lambda _artifacts: data_dir), components,
    )
