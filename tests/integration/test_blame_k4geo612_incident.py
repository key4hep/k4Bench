"""The 2026-09-25 k4geo#612 incident, composed end to end without a model.

That night confirmed a −11.7% VmPeak step on ALLEGRO_o2_v01 across 09-23 → 09-24,
caused by k4geo#612 (``SiWr_nLayers`` 2 → 1 in ``ALLEGRO_o2_v01/DectDimensions.xml``).
The ranker never saw that line, and was told the benchmark host had changed:

* the onset release ran on fcc-ironic-03 *and* fcc-ironic-01, the release
  before it on fcc-ironic-01 alone, and a set comparison reported "01 -> 03"
  although fcc-ironic-01 measured the new level too;
* #612's alphabetically first hunks (CMakeLists, the o1_v03 and o1_v04
  variants) spent its whole diff sample, so the o2_v01 hunk was dropped.

Everything here goes through the public path production takes: report JSON
through :func:`~k4bench.regression.render.from_json`, GitHub responses through
the real :func:`~k4bench.blame.github.resolve_repo_prs`, the builder's own
request assembly, and the ranker's prompt. The data is reduced and partly
synthetic; the shapes are the incident's.
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit

from k4bench.blame.builder import build_blame_report
from k4bench.blame.github import GitHubClient
from k4bench.blame.rank import RankResult, _build_user_prompt
from k4bench.regression.render import from_json

_PLAT = "x86_64-almalinux9-gcc14.2.0-opt"
_OWN = "FCCee/ALLEGRO/compact/ALLEGRO_o2_v01/"
_SIWR = '+    <constant name="SiWr_nLayers" value="1"/>'
_IRONIC01 = {"name": "fcc-ironic-01", "cpu_cores": 64}
_IRONIC03 = {"name": "fcc-ironic-03", "cpu_cores": 64}


# ── The report ────────────────────────────────────────────────────────────────

def _point(release, value, hosts, severity="OK", direction="NONE"):
    return {
        "run_date": release, "value": value, "n_runs": len(hosts),
        "n_judged": len(hosts), "severity": severity, "direction": direction,
        "hosts": [host for host, _ in hosts],
        "host_levels": [{"host": host, "value": level} for host, level in hosts],
    }


def _report_json() -> dict:
    verdict = {
        "detector": "ALLEGRO_o2_v01", "platform": _PLAT, "sample": "single_e-",
        "label": "baseline", "metric_family": "memory", "metric": "vmpeak_mb",
        "sub_detector": None, "run_id": "2026-09-25", "run_date": "2026-09-24",
        "value": 5850.0, "baseline_median": 6620.0, "baseline_mad": 5.0,
        "pct_change": -0.1163, "z_score": -154.0, "severity": "CONFIRMED",
        "direction": "DOWN", "reason": "step",
        "onset_run_id": "2026-09-24", "onset_run_date": "2026-09-24",
        "last_accepted_run_id": "2026-09-23", "last_accepted_run_date": "2026-09-23",
        "first_confirmed_run_id": "2026-09-25",
        "history": [
            _point("2026-09-22", 6618.0, [(_IRONIC01, 6618.0)]),
            _point("2026-09-23", 6621.0, [(_IRONIC01, 6621.0)]),
            _point("2026-09-24", 5850.0, [(_IRONIC03, 5849.0), (_IRONIC01, 5851.0)],
                   severity="CONFIRMED", direction="DOWN"),
        ],
    }
    return {
        "generated_at": "2026-09-25T06:00:00",
        "groups": [{
            "detector": "ALLEGRO_o2_v01", "platform": _PLAT, "sample": "single_e-",
            "k4h_release": "key4hep-2026-09-24", "run_date": "2026-09-25",
            "run_id": "2026-09-25", "verdicts": [verdict], "reliable": True,
            "geometry_path": _OWN + "ALLEGRO_o2_v01.xml",
        }],
    }


# ── GitHub ────────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body
        self.headers = {}

    def json(self):
        return self._body


class _GitHub:
    """Serves ``/repos/{slug}/…`` reads from *prs*: ``{slug: {number: files}}``."""

    def __init__(self, prs: dict[str, dict[int, list[dict]]]):
        self.prs = prs

    def request(self, method, url, headers=None, params=None, json=None, timeout=None):
        parts = urlsplit(url).path.split("/")[2:]  # owner, repo, …
        slug, rest = "/".join(parts[:2]), parts[2:]
        prs = self.prs.get(slug, {})
        if rest[0] == "compare":
            commits = [
                {"sha": f"{n:040d}", "commit": {"message": f"Change (#{n})"}}
                for n in prs
            ]
            return _Resp(200, {"total_commits": len(commits), "commits": commits})
        number = int(rest[1])
        if rest[2:] == ["files"]:
            return _Resp(200, prs[number] if (params or {}).get("page") == 1 else [])
        return _Resp(200, {
            "title": f"PR {number}", "user": {"login": "someone"},
            "html_url": f"https://github.com/{slug}/pull/{number}",
            "merged_at": "2026-09-23T12:00:00Z", "additions": 100, "deletions": 50,
            "changed_files": len(prs[number]), "body": "",
        })


def _lines(fill: str, size: int) -> str:
    """Added lines of *fill*, about *size* characters. Like GitHub's ``patch``,
    no trailing newline."""
    return "\n".join([f"+{fill}"] * max(1, size // (len(fill) + 2)))


def _hunk(fill: str, size: int) -> str:
    return "@@ -1,1 +1,1 @@\n" + _lines(fill, size)


def _k4geo_612() -> list[dict]:
    # The decisive line sits ~3920 characters into its hunk, as in the real one.
    dimensions = (
        "@@ -60,40 +60,40 @@\n" + _lines("pitch change", 3900)
        + "\n-    <constant name=\"SiWr_nLayers\" value=\"2\"/>\n" + _SIWR + "\n"
        + _lines("more pitch changes", 1900)
    )
    assert dimensions.index(_SIWR) > 3900
    return [
        {"filename": "CMakeLists.txt", "patch": _hunk("cmake", 1500)},
        {"filename": "FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/DectDimensions.xml",
         "patch": _hunk("o1_v03", 5000)},
        {"filename": "FCCee/ALLEGRO/compact/ALLEGRO_o1_v04/DectDimensions.xml",
         "patch": _hunk("o1_v04", 5000)},
        {"filename": "FCCee/ALLEGRO/compact/ALLEGRO_o1_v04/SiliconWrapper.xml",
         "patch": _hunk("o1_v04 wrapper", 3000)},
        {"filename": _OWN + "DectDimensions.xml", "patch": dimensions},
        {"filename": _OWN + "display.xml", "patch": _hunk("colour", 300)},
        {"filename": "FCCee/IDEA/compact/IDEA_o2_v01/IDEA_o2_v01.xml",
         "patch": _hunk("idea", 400)},
        {"filename": "FCCee/IDEA/compact/IDEA_o2_v01_CI/IDEA_o2_v01_CI.xml",
         "patch": _hunk("idea ci", 400)},
    ]


def _competitor(number: int) -> list[dict]:
    return [
        {"filename": f"DDCore/src/File{number}_{i}.cpp", "patch": _hunk(f"dd4hep {number}", 4500)}
        for i in range(3)
    ]


def _provenance(platform, release):
    commit = {"2026-09-23": "a", "2026-09-24": "c"}.get(release)
    if commit is None:
        return None
    return {
        "k4geo": {"commit": commit * 40, "version": "develop",
                  "repo_url": "https://github.com/key4hep/k4geo.git"},
        "DD4hep": {"commit": commit * 40, "version": "develop",
                   "repo_url": "https://github.com/AIDASoft/DD4hep.git"},
    }


class _CapturingRanker:
    def __init__(self):
        self.requests = []

    def rank(self, request):
        self.requests.append(request)
        return RankResult(rankings={})


def _incident_prompt() -> str:
    report = from_json(json.loads(json.dumps(_report_json())))
    github = GitHubClient(token="t", session=_GitHub({
        "key4hep/k4geo": {612: _k4geo_612()},
        "AIDASoft/DD4hep": {n: _competitor(n) for n in range(1500, 1520)},
    }))
    ranker = _CapturingRanker()
    build_blame_report(
        report, packages_for_release=_provenance, github=github, ranker=ranker,
        historical_diffs=False,
    )
    (request,) = ranker.requests
    assert len(request.candidates) == 21
    return _build_user_prompt(request)


def _candidate_block(prompt: str, number: int) -> str:
    """One candidate's lines: up to the next candidate or the next section."""
    start = prompt.index(f"- #{number} — ")
    ends = [e for e in (prompt.find("\n- #", start + 1), prompt.find("\n\n", start)) if e >= 0]
    return prompt[start:min(ends)] if ends else prompt[start:]


# ── The incident ──────────────────────────────────────────────────────────────

def test_the_host_evidence_says_the_step_reproduced_on_the_same_machine():
    prompt = _incident_prompt()
    assert "The benchmark host changed exactly at the onset release" not in prompt
    assert (
        "fcc-ironic-01 (64 cores) measured both the release before the onset and "
        "the onset release, and moved with the step: switching machines does not "
        "explain it."
    ) in prompt


def test_the_decisive_hunk_of_k4geo_612_reaches_the_ranker():
    prompt = _incident_prompt()
    block = _candidate_block(prompt, 612)
    assert f"own compact directory {_OWN} (DectDimensions.xml, display.xml)" in block
    assert block.index(f"--- {_OWN}DectDimensions.xml ---") < block.index(
        "--- FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/DectDimensions.xml ---"
    )
    assert _SIWR in block
    # Favoured: it is served before the even share every competitor gets.
    assert len(block) > 3 * len(_candidate_block(prompt, 1500))
