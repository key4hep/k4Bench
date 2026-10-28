"""The shell a recipe is written in, common to both tools' recipes.

Every value a recipe interpolates reaches a reader as part of a command they
are invited to paste into a shell, so it is quoted (:func:`_safe`), and every
word of prose is a ``#`` comment (:func:`_note`), so the whole file is one
paste. Each half runs in a subshell over its own worktree, set up the same way
for either tool (:func:`_half_setup`).
"""

from __future__ import annotations

import shlex
import textwrap

_NIGHTLY_REPO = "/cvmfs/sw-nightlies.hsf.org"


def _safe(value: object) -> str:
    """Shell-quote untrusted text: every value below reaches the reader as part
    of a command they are invited to paste into a shell."""
    return shlex.quote(str(value))


def _stack_source(release: str, setup: str) -> list[str]:
    """Source a recorded LCG view, or the dated legacy Spack release."""
    if not setup:
        return [
            f"source {_safe(_NIGHTLY_REPO + '/key4hep/setup.sh')} --spack -r {_safe(release)}"
        ]
    return [
        f"stack_setup={_safe(setup)}",
        'stack_generated="$(sed -n \'s/^# *Generated: *//p\' "$stack_setup" | head -1)"',
        f'[[ "$(date -u -d "$stack_generated" +%F)" == {_safe(release)} ]] || '
        '{ echo "Recorded LCG view is no longer available" >&2; exit 1; }',
        'source "$stack_setup"',
    ]


def _subshell(lines: list[str]) -> str:
    """*lines* as one ``( ... )`` subshell, indented.

    Indented per *logical* line: a quoted argument can carry a newline of its
    own, and indenting inside it would change the value that reaches the job.
    """
    body = "\n".join(f"  {line}" for line in lines)
    return f"(\n{body}\n)"


def _continued(parts: list[str]) -> list[str]:
    """One command over several lines: the *parts* after the first indented
    under it, every line but the last continued with a backslash."""
    lines = [parts[0], *(f"        {part}" for part in parts[1:])]
    return [f"{line} \\" for line in lines[:-1]] + lines[-1:]


def _half_setup(worktree: str, release: str, stack_setup: str) -> list[str]:
    """The opening lines of one half of the recipe, a subshell over its own
    worktree.

    A subshell is a separate process holding a copy of the environment, so each
    half sources its own Key4hep release from a clean start: the release the
    other half sourced cannot leak into it, and neither reaches the shell the
    reader pasted into. That is what lets the two halves — which the Key4hep
    setup script forbids sourcing into one environment — run one after the other
    in a single paste.

    The worktree is what keeps the *filesystem* apart, and it is not optional:
    ``setup.sh`` reuses an existing ``py-venv`` and ``plugin/build.sh`` reuses an
    up-to-date ``.so``, so one shared checkout would hand the second half a
    virtual environment and a timing plugin built against the first half's
    release — measuring the two stacks with one stack's binaries, which is
    precisely the difference the recipe exists to show. Both worktrees are
    created from the one clone outside, each at the commit its nightly ran.
    """
    return [
        # Inside the subshell, never at the top: a failed setup would otherwise
        # benchmark a broken environment and report the difference as a result,
        # and an errexit set in the reader's own shell would outlive the paste.
        "set -e",
        f"cd {_safe(worktree)}",
        *_stack_source(release, stack_setup),
        "source setup.sh",
        "pip install --no-build-isolation -e .",
    ]


def _fetch_stage(release: str, setup: str, fetch: list[str]) -> str:
    """A fetch both halves read from, as a subshell of its own.

    ``xrdcp`` comes from the Key4hep stack rather than from the host, so a
    fetch that runs before any release is sourced is a ``command not found``.
    This one sources the BEFORE release — any release fetches the same bytes —
    and takes its own subshell, so that release reaches neither half below:
    each of those sources the release its own nightly ran, and the Key4hep
    setup script forbids a second one in an environment that already holds one.
    """
    return _subshell(["set -e", *_stack_source(release, setup), *fetch])


def _differed(parity_diffs: tuple[str, ...]) -> str:
    """The warning a recipe opens with when its two runs were not configured
    alike."""
    return _note(
        "WARNING: these runs did NOT measure the same workload; the "
        "recorded values differed for: "
        + ", ".join(parity_diffs)
        + ". The commands below reproduce each run as it was, so that "
        "difference is reproduced too."
    )


def _rule(title: str, width: int = 78) -> str:
    """A titled section rule, as a comment of a fixed width, so the halves are
    equally easy to find when scrolling a pasted file."""
    prefix = f"# --- {title} "
    return prefix + "-" * max(3, width - len(prefix))


def _note(text: str, width: int = 78) -> str:
    """Soft-wrapped prose as shell comments, so it reads in a terminal as well
    as a browser and the file stays paste-safe end to end. Commands are never
    passed through here: a wrapped command is a broken one.

    Hyphens and long words never break: the prose names directories and paths
    (``k4Bench-before/logs/before-<run>``), and a reader who has to reassemble
    one from two lines is being asked to guess whether the hyphen was ours.
    """
    lines = textwrap.wrap(
        text, width=width - 2, break_on_hyphens=False, break_long_words=False
    ) or [text]
    return "\n".join(f"# {line}" for line in lines)
