#!/usr/bin/env python3
"""Plan detector bumps from an installed nightly; write only with --apply."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
VERSION = re.compile(r"([A-Za-z0-9_-]+)_v(\d+)")
# GITHUB_TOKEN cannot edit workflow files; their examples are version-independent.
EXAMPLES = (".github/scripts/nightly_benchmark.sh",)


@dataclass(frozen=True)
class Bump:
    family: str
    old: str
    new: str
    compact: str


def detect(root: Path, k4geo: Path, family: str | None = None) -> list[Bump]:
    if not k4geo.is_dir():
        raise ValueError(f"Missing K4GEO directory: {k4geo}")
    current = {}
    for path in sorted((root / ".github/benchmarks").glob("*.yml")):
        match = VERSION.fullmatch(path.stem)
        cfg = yaml.safe_load(path.read_text())
        # A k4run benchmark reconstructs a frozen input simulated with its
        # geometry, so that geometry is never bumped.
        if cfg.get("tool", "ddsim") != "ddsim":
            continue
        xml = cfg["xml"]
        if not match or xml.startswith("$") or Path(xml).is_absolute():
            continue
        name, number = match[1], int(match[2])
        if family and name != family:
            continue
        if name not in current or number > current[name][0]:
            current[name] = number, path.stem, xml
    bumps = []
    for name, (number, old, xml) in sorted(current.items()):
        rel = Path(xml).parent.parent
        if Path(xml).name != f"{old}.xml" or Path(xml).parent.name != old:
            raise ValueError(f"Unexpected geometry layout: {xml}")
        compact = k4geo / rel
        if not compact.is_dir():
            raise ValueError(f"Missing compact directory: {compact}")
        candidates = []
        for candidate in compact.iterdir():
            match = VERSION.fullmatch(candidate.name)
            if (
                match
                and match[1] == name
                and int(match[2]) > number
                and candidate.is_dir()
                and (candidate / f"{candidate.name}.xml").is_file()
            ):
                candidates.append((int(match[2]), candidate.name))
        if candidates:
            new = max(candidates)[1]
            bumps.append(Bump(name, old, new, rel.as_posix()))
    return bumps


def edit_lineage(text: str, old: str, new: str) -> str:
    # The initial map may be empty before the first detector transition.
    text = text.replace(
        "DETECTOR_SUCCESSORS: dict[str, str] = {}\n", "DETECTOR_SUCCESSORS: dict[str, str] = {\n}\n"
    )
    pattern = re.compile(r"(?m)^DETECTOR_SUCCESSORS: dict\[str, str\] = \{\n(?P<body>.*?)^\}", re.S)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError("Unexpected DETECTOR_SUCCESSORS block")
    match = matches[0]
    lines = match["body"].splitlines(keepends=True)
    entry = re.compile(r'    "([A-Za-z0-9_-]+)":\s+"([A-Za-z0-9_-]+)",\n')
    if any(not entry.fullmatch(line) for line in lines):
        raise ValueError("Unexpected DETECTOR_SUCCESSORS entries")
    values = ast.literal_eval("{" + match["body"] + "}")
    if new in values:
        raise ValueError(f"Successor already exists: {new}")
    previous = values.pop(old, None)
    values[new] = old
    if set(values) & set(values.values()):
        raise ValueError("Detector succession would chain")
    body = "".join(line for line in lines if entry.fullmatch(line)[1] != old)
    body += f'    "{new}": "{old}",\n'
    text = text[: match.start("body")] + body + text[match.end("body") :]
    # Dropping the one-hop entry must not make its predecessor look active again:
    # it stays replaced by the config that succeeded it.
    return edit_retired(text, previous, old) if previous else text


def edit_retired(text: str, detector: str, successor: str) -> str:
    text = text.replace(
        "RETIRED_DETECTORS: dict[str, str] = {}\n", "RETIRED_DETECTORS: dict[str, str] = {\n}\n"
    )
    pattern = re.compile(r"(?m)^RETIRED_DETECTORS: dict\[str, str\] = \{\n(?P<body>.*?)^\}", re.S)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError("Unexpected RETIRED_DETECTORS block")
    match = matches[0]
    lines = match["body"].splitlines(keepends=True)
    entry = re.compile(r'    "([A-Za-z0-9_-]+)":\s+"([A-Za-z0-9_-]+)",\n')
    if any(not entry.fullmatch(line) for line in lines):
        raise ValueError("Unexpected RETIRED_DETECTORS entries")
    retired = dict(entry.fullmatch(line).groups() for line in lines)
    if retired.get(detector, successor) != successor:
        raise ValueError(f"Retired detector already has another successor: {detector}")
    retired[detector] = successor
    body = "".join(f'    "{name}": "{retired[name]}",\n' for name in sorted(retired))
    return text[: match.start("body")] + body + text[match.end("body") :]


def replace_references(text: str, replacements: dict[str, str]) -> str:
    """Replace once, preferring whole paths over the version names inside them."""
    pattern = "|".join(re.escape(key) for key in sorted(replacements, key=len, reverse=True))
    return re.sub(pattern, lambda match: replacements[match[0]], text)


def edit_benchmark(text: str, bump: Bump, fcc_config: Path) -> tuple[str, list[str]]:
    """Update references while preserving YAML formatting and reviewer comments."""
    cfg = yaml.safe_load(text)
    notes = []
    # Headers mix separators: "CLD o2_v09", "ILD_FCCee v02", "IDEA o1 v04".
    name_pattern = "[ _]".join(map(re.escape, bump.old.split("_")))
    old_version = bump.old.rsplit("_", 1)[1]
    new_version = bump.new.rsplit("_", 1)[1]
    replacements = {
        name: name.removesuffix(old_version) + new_version
        for name in re.findall(name_pattern, text)
    }
    for record in [cfg, *cfg.get("samples", [])]:
        steering = record.get("steering_file", "")
        if not steering:
            continue
        proposed = re.sub(re.escape(bump.family) + r"_v\d+", bump.new, steering)
        relative = re.sub(r"^\$(?:FCCCONFIG|\{FCCCONFIG\})/", "", proposed)
        if proposed != steering and relative != proposed and (fcc_config / relative).is_file():
            replacement = proposed
            notes.append(f"Steering updated: `{steering}` → `{proposed}`.")
        else:
            replacement = steering
            notes.append(
                f"Steering retained: `{steering}`; no verified replacement for {bump.new}."
            )
        # The path suffix also appears in the FCC-config link comment.
        replacements[steering] = replacement
        suffix = steering.split("/", 1)[-1]
        replacements[suffix] = replacement.split("/", 1)[-1]
    return replace_references(text, replacements), notes or ["No steering file configured."]


def prepare(root: Path, bump: Bump, fcc_config: Path) -> tuple[dict[Path, str], list[str]]:
    """Return all proposed edits and review notes without mutating files or the bump."""
    source = root / f".github/benchmarks/{bump.old}.yml"
    text, notes = edit_benchmark(source.read_text(), bump, fcc_config)
    lineage = root / "k4bench/regression/lineage.py"
    edits = {
        root / f".github/benchmarks/{bump.new}.yml": text,
        lineage: edit_lineage(lineage.read_text(), bump.old, bump.new),
    }
    for name in EXAMPLES:
        path = root / name
        if path.exists():
            original = path.read_text()
            if bump.old in original:
                edits[path] = original.replace(bump.old, bump.new)
    return edits, notes


def body(bump: Bump, notes: list[str]) -> str:
    base = "https://github.com/key4hep/k4geo/tree/main/" + bump.compact
    return (
        f"Bump `{bump.old}` → `{bump.new}`, available in the Key4hep nightly.\n\n"
        f"Geometry trees: [old]({base}/{bump.old}) · [new]({base}/{bump.new}).\n\n"
        + "\n".join(notes)
        + "\n\n"
        "CI smoke-tests the changed geometry/steering pairs with one particle-gun event. "
        "Physics correctness still needs review.\n\n"
        "- [ ] Click **Ready for review** on this draft to start CI and PR-Agent.\n"
        "- [ ] Review header comment prose and steering compatibility.\n"
        "- [ ] Confirm sweep selection and timeout remain sensible.\n"
        "- [ ] Review CI `run` (including geometry patching and simulation) and PR-Agent results.\n"
        "- [ ] Approve the baseline transition before merging; no auto-merge.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k4geo", type=Path, required=True)
    parser.add_argument("--fcc-config", type=Path, required=True)
    parser.add_argument("--family")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--rejected",
        type=Path,
        help="JSON list of proposed configs (e.g. IDEA_o1_v05) whose bump PR was closed unmerged",
    )
    args = parser.parse_args()
    if args.apply and not args.family:
        parser.error("--apply requires --family (one independent PR per family)")
    bumps = detect(ROOT, args.k4geo, args.family)
    if args.rejected:
        # A rejected version would otherwise be proposed again every run,
        # holding back every other family; a newer version is a new proposal.
        rejected = set(json.loads(args.rejected.read_text()))
        bumps = [bump for bump in bumps if bump.new not in rejected]
    summaries = []
    for bump in bumps:
        edits, notes = prepare(ROOT, bump, args.fcc_config)
        summaries.append({**asdict(bump), "notes": notes})
        if args.apply:
            old = ROOT / f".github/benchmarks/{bump.old}.yml"
            new = ROOT / f".github/benchmarks/{bump.new}.yml"
            if new.exists():
                raise ValueError(f"Refusing to overwrite {new}")
            subprocess.run(["git", "mv", str(old), str(new)], cwd=ROOT, check=True)
            for path, content in edits.items():
                path.write_text(content)
            output = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir()))
            (output / f"bump-{bump.family}.md").write_text(body(bump, notes))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
