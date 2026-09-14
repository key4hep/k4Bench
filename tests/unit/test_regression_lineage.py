"""Unit tests for platform succession (:mod:`k4bench.regression.lineage`)."""

from __future__ import annotations

from k4bench.regression.lineage import (
    PLATFORM_SUCCESSORS,
    is_replaced,
    predecessor_of,
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


def test_a_chain_of_migrations_resolves_exactly_one_link(monkeypatch):
    monkeypatch.setitem(PLATFORM_SUCCESSORS, "gcc16", "gcc14")
    monkeypatch.setitem(PLATFORM_SUCCESSORS, "gcc17", "gcc16")
    assert predecessor_of("gcc17") == "gcc16"
    assert predecessor_of("gcc16") == "gcc14"
    assert successors_of("gcc14") == ("gcc16",)
    assert is_replaced("gcc16") and not is_replaced("gcc17")


def test_no_platform_replaces_itself():
    assert not [p for p, pre in PLATFORM_SUCCESSORS.items() if p == pre]
