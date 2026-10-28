"""Build the runnable benchmark recipe one blame comment links to.

The recipe is a standalone plain-text file (:func:`render_text`) published
beside the nightly data under a name derived from the measurement it
reproduces (:func:`artifact_name`), so the comment spends one table cell on a
link rather than dozens of lines on a fold-out shell script.

Each half of a recipe reruns the command its own nightly run recorded: a ddsim
run's (:mod:`.ddsim`) is ``k4bench`` on its geometry, a k4run job's
(:mod:`.k4run`) ``k4bench k4run`` on its frozen input. Both are read from the
two run records the same way (:mod:`.records`), written in the same shell
(:mod:`.shell`), and framed by the same file (:mod:`.recipe`).

This package deliberately owns no I/O.  Callers fetch the two immutable
``run_info.json`` records (and, for legacy HepMC runs, may fill ``input_files``
from the checked-in benchmark config) before handing them to :func:`facts_from`,
and callers publish whatever :func:`render_text` returns.
"""

from k4bench.blame.reproduce.ddsim import ReproducerFacts, sweep_flag
from k4bench.blame.reproduce.k4run import K4runReproducerFacts
from k4bench.blame.reproduce.recipe import (
    RecipeFacts,
    artifact_name,
    facts_from,
    render_text,
)

__all__ = [
    "K4runReproducerFacts",
    "RecipeFacts",
    "ReproducerFacts",
    "artifact_name",
    "facts_from",
    "render_text",
    "sweep_flag",
]
