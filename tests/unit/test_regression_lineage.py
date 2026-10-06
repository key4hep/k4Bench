"""Unit tests for platform and detector succession (:mod:`k4bench.regression.lineage`)."""

from __future__ import annotations

import pytest

from k4bench.regression import lineage
from k4bench.regression.lineage import (
    DETECTOR_SUCCESSORS,
    PLATFORM_SUCCESSORS,
    detector_predecessor_of,
    detector_successors_of,
    is_detector_replaced,
    is_replaced,
    predecessor_of,
    predecessor_series,
    successor_series,
    successors_of,
)

_SPACK = "x86_64-almalinux9-gcc14.2.0-opt"
_LCG = "x86_64-el9-gcc16-opt"


def test_the_lcg_platform_continues_the_spack_platform():
    assert predecessor_of(_LCG) == _SPACK
    assert successors_of(_SPACK) == (_LCG,)
    assert is_replaced(_SPACK)
    assert not is_replaced(_LCG)


def test_an_unmapped_platform_has_neither_predecessor_nor_successor():
    # Platforms benchmarked side by side are independent series.
    assert predecessor_of("aarch64-el9-gcc16-opt") is None
    assert successors_of("aarch64-el9-gcc16-opt") == ()
    assert not is_replaced("aarch64-el9-gcc16-opt")


def test_succession_does_not_chain():
    # A successor continues its predecessor's own runs only; a predecessor that
    # had itself replaced a platform would pass on a history cut short.
    assert not set(PLATFORM_SUCCESSORS) & set(PLATFORM_SUCCESSORS.values())


def test_no_platform_replaces_itself():
    assert not [p for p, pre in PLATFORM_SUCCESSORS.items() if p == pre]


@pytest.fixture
def detector_map(monkeypatch):
    monkeypatch.setattr(lineage, "DETECTOR_SUCCESSORS", {"ALLEGRO_o1_v04": "ALLEGRO_o1_v03"})


def test_a_geometry_version_continues_the_version_it_replaced(detector_map):
    assert detector_predecessor_of("ALLEGRO_o1_v04") == "ALLEGRO_o1_v03"
    assert detector_successors_of("ALLEGRO_o1_v03") == ("ALLEGRO_o1_v04",)
    assert is_detector_replaced("ALLEGRO_o1_v03")
    assert not is_detector_replaced("ALLEGRO_o1_v04")
    # Another option of the same concept is an independent series.
    assert detector_predecessor_of("ALLEGRO_o2_v01") is None


def test_a_retired_detector_stays_replaced_without_a_successor_entry(monkeypatch):
    monkeypatch.setattr(lineage, "DETECTOR_SUCCESSORS", {"ALLEGRO_o1_v05": "ALLEGRO_o1_v04"})
    monkeypatch.setattr(lineage, "RETIRED_DETECTORS", {"ALLEGRO_o1_v03": "ALLEGRO_o1_v04"})
    assert is_detector_replaced("ALLEGRO_o1_v03")
    # v05 counts too, so v03 stops being expected even if v04 never ran.
    assert detector_successors_of("ALLEGRO_o1_v03") == ("ALLEGRO_o1_v04", "ALLEGRO_o1_v05")
    assert successor_series("ALLEGRO_o1_v03", _LCG) == (
        ("ALLEGRO_o1_v04", _LCG), ("ALLEGRO_o1_v05", _LCG),
    )
    # Retirement never continues a history: v04's own entry is gone.
    assert detector_predecessor_of("ALLEGRO_o1_v04") is None
    assert predecessor_series("ALLEGRO_o1_v04", _LCG) == (("ALLEGRO_o1_v04", _SPACK),)
    assert not is_detector_replaced("ALLEGRO_o1_v05")


def test_the_replaced_detector_is_the_nearest_predecessor_series(detector_map):
    # A version bump after the platform migration: the new version only ever
    # ran on the new platform, so its own predecessor on that platform comes
    # first, and the platform's predecessor is the fallback.
    assert predecessor_series("ALLEGRO_o1_v04", _LCG) == (
        ("ALLEGRO_o1_v03", _LCG), ("ALLEGRO_o1_v04", _SPACK),
    )
    assert predecessor_series("ALLEGRO_o1_v03", _LCG) == (("ALLEGRO_o1_v03", _SPACK),)
    assert predecessor_series("ALLEGRO_o1_v03", _SPACK) == ()


def test_successor_series_inverts_predecessor_series(detector_map):
    assert successor_series("ALLEGRO_o1_v03", _LCG) == (("ALLEGRO_o1_v04", _LCG),)
    assert successor_series("ALLEGRO_o1_v03", _SPACK) == (
        ("ALLEGRO_o1_v04", _SPACK), ("ALLEGRO_o1_v03", _LCG),
    )
    assert successor_series("ALLEGRO_o1_v04", _LCG) == ()


def test_detector_succession_does_not_chain():
    assert not set(DETECTOR_SUCCESSORS) & set(DETECTOR_SUCCESSORS.values())
    assert not [d for d, pre in DETECTOR_SUCCESSORS.items() if d == pre]


def test_every_successor_detector_is_a_benchmarked_config():
    # A successor names a config file in .github/benchmarks/; a typo would
    # silently leave the new version cold and the old one reported missing.
    from pathlib import Path

    configs = {
        p.stem for p in (Path(__file__).parents[2] / ".github" / "benchmarks").glob("*.yml")
    }
    assert set(DETECTOR_SUCCESSORS) <= configs
    assert not set(DETECTOR_SUCCESSORS.values()) & configs
