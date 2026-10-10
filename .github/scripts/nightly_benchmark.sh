#!/bin/bash
#
# Runs a single k4bench benchmark and uploads results to CERN EOS.
# The matrix in .github/workflows/nightly.yml expands .github/benchmarks/*.yml
# into a flat set of jobs (see .github/scripts/list_benchmarks.py). This script
# sources the Key4hep stack, installs k4bench, has benchmark_job.py run the job
# and record it, and uploads the result.
#
# Required env vars (set by the workflow):
#   BENCHMARK_CONFIG  — config file stem, e.g. "ALLEGRO_o1_v04"
#   BENCHMARK_SAMPLE  — sample name, e.g. "single_e-_10GeV"
#   BENCHMARK_JOB     — the job's record from list_benchmarks.py, as JSON
#   X509_USER_CERT, X509_USER_KEY — EOS service certificate paths
#   GITHUB_RUN_ID, GITHUB_SHA, GITHUB_REPOSITORY, GITHUB_SERVER_URL
#
# Optional env vars:
#   K4H_RELEASE_REQUESTED — publication date resolved once per night
#   K4H_STACK_SETUP       — exact LCG view setup.sh resolved with that date
#
# EOS layout written by this script (below {EOS_ROOT}/_k4run/ for a k4run job):
#   {EOS_ROOT}/{detector}/{platform}/key4hep-{release}/{sample}/{YYYY-MM-DD}/
#     run_info.json
#     machine_info.json
#     {config}_results.csv
#     {config}_events.json
#     {config}_regions.json       ddsim only
#     {config}_components.json    k4run only
#     {config}_joboptions.opts    k4run only
#     {config}.log

set -euo pipefail

# This script's own checkout, so the steps that run before k4bench is installed
# do not depend on the directory it was invoked from.
K4BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Personal EOS area.
EOS_FQDN="eosuser.cern.ch"
EOS_ROOT="/eos/user/j/jbeirer/k4bench"

SAMPLE="${BENCHMARK_SAMPLE}"

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "::group::1. System dependencies"
dnf install -y --quiet time voms-clients-cpp systemd-libs
echo "::endgroup::"

# ── 2. Job parameters ─────────────────────────────────────────────────────────
echo "::group::2. Job parameters"
echo "${BENCHMARK_JOB}"
echo "::endgroup::"

# ── 3. Key4hep nightly ────────────────────────────────────────────────────────
echo "::group::3. Key4hep nightly"
[[ -f "${K4H_STACK_SETUP:-}" ]] || { echo "ERROR: LCG setup not found: ${K4H_STACK_SETUP:-<unset>}" >&2; exit 1; }
# Reads the release date out of the view's header, so the view need not be
# sourced. This runs before section 4 installs k4bench, so the checkout is put
# on the path explicitly rather than relying on the interpreter's cwd; `|| true`
# keeps a failure in the guard below instead of aborting with no message worth
# reading.
read_k4h_identity() {
    IFS='|' read -r K4H_RELEASE K4H_PLATFORM <<< "$(PYTHONPATH="${K4BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 -c \
        'import sys; from k4bench.provenance.stack import stack_identity; print("|".join(stack_identity(sys.argv[1])))' \
        "${K4H_STACK_SETUP}" || true)"
}
read_k4h_identity
# A runner's CVMFS client can keep serving last week's weekday slot after the
# new nightly is published, until its catalog TTL (4 min by default) expires and
# its kernel cache drains (1 min more), so give it that long plus a margin.
CVMFS_STALE_TIMEOUT=360
CVMFS_DEADLINE=$(( SECONDS + CVMFS_STALE_TIMEOUT ))
while [[ "${K4H_RELEASE}" != "${K4H_RELEASE_REQUESTED:-${K4H_RELEASE}}" ]] && (( SECONDS < CVMFS_DEADLINE )); do
    echo "WARNING: CVMFS still serves ${K4H_RELEASE:-<unreadable>} instead of ${K4H_RELEASE_REQUESTED}; retrying in 30 s"
    sleep 30
    read_k4h_identity
done
[[ -n "${K4H_RELEASE}" ]] || { echo "ERROR: Failed to read Key4hep publication date from K4H_STACK_SETUP" >&2; exit 1; }
# The resolved LCG setup is the source of truth for the label and the EOS path, so
# a pinned source that lands somewhere else would file results under a release
# that never produced them. Mislabelled results outlive a red job.
if [[ -n "${K4H_RELEASE_REQUESTED:-}" && "${K4H_RELEASE}" != "${K4H_RELEASE_REQUESTED}" ]]; then
    echo "ERROR: requested Key4hep release ${K4H_RELEASE_REQUESTED} but ${K4H_STACK_SETUP} still provides ${K4H_RELEASE}" >&2
    exit 1
fi
set +u
source "${K4H_STACK_SETUP}"
set -u
[[ -n "${KEY4HEP_STACK:-}" ]] || { echo "ERROR: KEY4HEP_STACK not set after sourcing Key4hep setup" >&2; exit 1; }
# benchmark_job.py records the run under this release and platform.
export K4H_RELEASE K4H_PLATFORM
echo "Release : key4hep-${K4H_RELEASE}"
echo "Platform: ${K4H_PLATFORM}"
echo "View    : ${K4H_STACK_SETUP}"
echo "::endgroup::"

# Outside the log group: which release produced these numbers is the first thing
# anyone reading a regression asks. Unset when the script runs outside Actions.
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    echo "- \`${BENCHMARK_CONFIG}\` / \`${SAMPLE}\`: Key4hep \`${K4H_RELEASE}\` (\`${K4H_PLATFORM}\`)" \
        >> "${GITHUB_STEP_SUMMARY}" || true
fi

# ── 4. Install k4bench ───────────────────────────────────────────────────────
echo "::group::4. Install k4bench"
export K4BENCH_REPO="${K4BENCH_ROOT}"
export LD_LIBRARY_PATH="${K4BENCH_REPO}/plugin/install/lib:${K4BENCH_REPO}/plugin/build:${LD_LIBRARY_PATH:-}"
mkdir -p ~/.local/bin
export PATH=~/.local/bin:"${PATH}"

if [ ! -f ~/.local/bin/cvmfs-venv ]; then
    curl -sL https://raw.githubusercontent.com/jbeirer/cvmfs-venv/main/cvmfs-venv.sh \
        -o ~/.local/bin/cvmfs-venv
    chmod +x ~/.local/bin/cvmfs-venv
fi
cvmfs-venv py-venv
. py-venv/bin/activate
pip install --no-build-isolation --quiet "."
bash plugin/build.sh
echo "::endgroup::"

# ── 5.–8. Run the benchmark job ──────────────────────────────────────────────
# Resolves the job's inputs, snapshots the machine around the benchmark, runs
# k4bench and writes run_info.json (see benchmark_job.py). Its exit code is the
# benchmark's: a sweep may produce valid results for 27/28 configs and fail one,
# so a failure is surfaced only after what succeeded is uploaded. A job that
# could not start records no run_info.json and has nothing to upload.
OUTPUT_DIR="logs/benchmark"
set +e
python3 .github/scripts/benchmark_job.py "${OUTPUT_DIR}"
BENCH_RC=$?
set -e
[[ -f "${OUTPUT_DIR}/run_info.json" ]] || { echo "ERROR: the benchmark job did not start; nothing to upload" >&2; exit 1; }
# The run's tool, detector and date, as recorded: the EOS path must agree with them.
read -r TOOL DETECTOR DATE <<< "$(python3 -c \
    'import json, sys; info = json.load(open(sys.argv[1])); print(info["tool"], info["detector"], info["date"])' \
    "${OUTPUT_DIR}/run_info.json")"

# ── 9. Upload to EOS ──────────────────────────────────────────────────────────
echo "::group::9. Upload to EOS"
export X509_CERT_DIR=/cvmfs/grid.cern.ch/etc/grid-security/certificates
export X509_VOMS_DIR=/cvmfs/grid.cern.ch/etc/grid-security/vomsdir
export VOMS_USERCONF=/cvmfs/grid.cern.ch/etc/vomses
export X509_USER_PROXY=/tmp/x509_proxy
voms-proxy-init \
  --cert "${X509_USER_CERT}" \
  --key "${X509_USER_KEY}" \
  --out "${X509_USER_PROXY}"

unset X509_USER_CERT
unset X509_USER_KEY

# The run directory k4bench/layout.py defines: {detector}/{platform}/key4hep-{release}/{sample}/{date},
# for a k4run job below its reserved tree, which no reader of the ddsim runs walks.
EOS_TREE="${EOS_ROOT}"
if [[ "${TOOL}" == "k4run" ]]; then
    EOS_TREE="${EOS_ROOT}/_k4run"
fi
EOS_RUN="${EOS_TREE}/${DETECTOR}/${K4H_PLATFORM}/key4hep-${K4H_RELEASE}/${SAMPLE}/${DATE}"
EOS_URL="root://${EOS_FQDN}/${EOS_RUN}"

command -v xrdfs >/dev/null || { echo "ERROR: xrdfs not found" >&2; exit 1; }
command -v xrdcp >/dev/null || { echo "ERROR: xrdcp not found" >&2; exit 1; }

xrdfs "root://${EOS_FQDN}" mkdir -p "${EOS_RUN}"

for f in "${OUTPUT_DIR}"/*; do
    echo "  → $(basename "${f}")"
    xrdcp --force "${f}" "${EOS_URL}/$(basename "${f}")" \
        || { echo "ERROR: Failed to upload ${f}" >&2; exit 1; }
done
echo "Uploaded to: ${EOS_URL}"
echo "::endgroup::"

# Surface the benchmark exit code now that results are safely uploaded, so a
# failed sweep config (or any other ddsim or k4run failure) still turns the job red.
if [[ "${BENCH_RC}" -ne 0 ]]; then
    echo "ERROR: benchmark exited with code ${BENCH_RC} (one or more runs failed); results uploaded regardless" >&2
    exit "${BENCH_RC}"
fi
