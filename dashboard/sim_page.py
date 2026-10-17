"""The Simulation page: the ``sim`` suite's ddsim benchmarks.

Its sections, scopes and renderers are the registries in :mod:`sections` and
:mod:`section_dispatch`. It is the dashboard's default page, served at ``/``,
which is where the nightly mail's and pull-request comments' ``?tab=`` deep
links point.
"""
from __future__ import annotations

from k4bench.layout import SIM
from section_dispatch import RENDERERS
from sections import REMOTE_ONLY, SECTION_NAMES, SECTION_SCOPE
from suite_page import SuitePage, render_page

SIM_PAGE = SuitePage(
    suite=SIM,
    sections=tuple(SECTION_NAMES),
    remote_only=REMOTE_ONLY,
    scopes=SECTION_SCOPE,
    renderers=RENDERERS,
)


def main() -> None:
    render_page(SIM_PAGE)
