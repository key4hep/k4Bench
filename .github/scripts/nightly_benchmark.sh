#!/bin/bash
#
# Runs a single k4bench benchmark and uploads results to CERN EOS.
# All configuration arrives via environment variables; the matrix in
# .github/workflows/nightly.yml expands .github/benchmarks/*.yml into a
# flat set of jobs (see .github/scripts/list_benchmarks.py).
#
# Required env vars (set by the workflow):
#   BENCHMARK_CONFIG  — config file stem, e.g. "ALLEGRO_o1_v04"
#   BENCHMARK_SAMPLE  — sample name, e.g. "single_e-_10GeV"
#   XML_PATH          — detector geometry, $K4GEO-relative or absolute
#   N_EVENTS          — positive integer
#   DDSIM_ARGS        — verbatim ddsim flags (string, may be empty)
#   INPUT_FILES       — space-separated HepMC paths (may be empty)
#   STEERING_FILE     — optional ddsim --steeringFile path; $VAR expansion supported
#   SWEEP             — "true"/"false"
#   SWEEP_DETECTORS   — space-separated subdetector names (may be empty): partial sweep
#   VERBOSE           — "true"/"false"
#   INCLUDE_ONLY      — space-separated subdetector names (may be empty)
#   EXCLUDE_ONLY      — space-separated subdetector names (may be empty)
#   X509_USER_CERT, X509_USER_KEY — EOS service certificate paths
#   GITHUB_RUN_ID, GITHUB_SHA, GITHUB_REPOSITORY, GITHUB_SERVER_URL
#
# Optional env vars:
#   K4H_RELEASE_REQUESTED — publication date resolved once per night
#   K4H_STACK_SETUP       — exact LCG view setup.sh resolved with that date
#
# EOS layout written by this script:
#   {EOS_ROOT}/{detector}/{platform}/key4hep-{release}/{sample}/{YYYY-MM-DD}/
#     run_info.json
#     machine_info.json
#     {config}_results.csv
#     {config}_events.json
#     {config}_regions.json
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
echo "  config       : ${BENCHMARK_CONFIG}"
echo "  sample       : ${SAMPLE}"
echo "  xml          : ${XML_PATH}"
echo "  n_events     : ${N_EVENTS}"
echo "  verbose      : ${VERBOSE}"
echo "  sweep        : ${SWEEP}"
echo "  sweep_dets   : ${SWEEP_DETECTORS:-<none>}"
echo "  include_only : ${INCLUDE_ONLY:-<none>}"
echo "  exclude_only : ${EXCLUDE_ONLY:-<none>}"
echo "  input_files  : ${INPUT_FILES:-<none>}"
echo "  steering_file: ${STEERING_FILE:-<none>}"
echo "  ddsim_args   : ${DDSIM_ARGS:-<none>}"
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

# ── 5. Resolve inputs (geometry + optional ddsim steering file) ───────────────
echo "::group::5. Resolve inputs"
# The XML path may reference Key4hep env vars (e.g. $DD4hepINSTALL, used by
# DD4hep's own reference/example detectors, which live outside $K4GEO) so expand
# them here, after the Key4hep stack is sourced, before checking for an absolute
# path.
CONFIGURED_XML_PATH="${XML_PATH}"
XML_PATH=$(python3 -c "import os, sys; print(os.path.expandvars(sys.argv[1]))" "${XML_PATH}")
if [[ "${XML_PATH}" = /* ]]; then
    DETECTOR_XML="${XML_PATH}"
else
    DETECTOR_XML="${K4GEO}/${XML_PATH}"
fi
[[ -f "${DETECTOR_XML}" ]] || { echo "ERROR: XML not found: ${DETECTOR_XML}"; exit 1; }
DETECTOR=$(basename "${DETECTOR_XML}" .xml)
echo "Detector : ${DETECTOR}"
echo "XML      : ${DETECTOR_XML}"

# Optional steering file. The path may reference Key4hep env vars (e.g. $FCCCONFIG)
# so we expand it here, after the Key4hep stack is sourced. Prepended to DDSIM_ARGS
# so a sample-level --steeringFile flag would override it if both are given.
STEERING_PATH=""
if [[ -n "${STEERING_FILE}" ]]; then
    STEERING_PATH=$(python3 -c "import os, sys; print(os.path.expandvars(sys.argv[1]))" "${STEERING_FILE}")
    [[ -f "${STEERING_PATH}" ]] || { echo "ERROR: steering file not found: ${STEERING_PATH}"; exit 1; }
    DDSIM_ARGS="--steeringFile ${STEERING_PATH} ${DDSIM_ARGS}"
    echo "Steering : ${STEERING_PATH}"
    # ddsim exec()s the steering file directly, without adding its own directory
    # to sys.path, so a steering file that does its own relative import of a
    # sibling module (e.g. CLDConfig's cld_arc_steer.py -> `from cld_steer
    # import *`) fails unless that directory is already importable.
    export PYTHONPATH="$(dirname "${STEERING_PATH}"):${PYTHONPATH:-}"
fi

# k4bench has no top-level --inputFiles flag; it forwards everything in
# --ddsim-args verbatim to ddsim. So we prepend --inputFiles into DDSIM_ARGS.
if [[ -n "${INPUT_FILES}" ]]; then
    # HepMC inputs can't be streamed over xrootd (ROOT mis-parses the text as a
    # ROOT file → SIGSEGV), so fetch to a local path first.
    LOCAL_INPUT="/tmp/$(basename "${INPUT_FILES}")"
    xrdcp --force "${INPUT_FILES}" "${LOCAL_INPUT}"
    DDSIM_ARGS="--inputFiles ${LOCAL_INPUT} ${DDSIM_ARGS}"
    echo "Inputs   : ${LOCAL_INPUT}"
fi
echo "::endgroup::"

# Capture the date once here so run_info.json and the EOS upload path always agree,
# even if the benchmark runs across a midnight boundary.
DATE=$(date +%Y-%m-%d)

# ── 6. Collect machine info (start snapshot, before benchmark) ────────────────
echo "::group::6. Collect machine info (start)"
python3 .github/scripts/machine_info.py start "logs/${DETECTOR}"
echo "::endgroup::"

# ── 7. Run benchmark ──────────────────────────────────────────────────────────
echo "::group::7. Run benchmark"
CMD=(k4bench
    --xml        "${DETECTOR_XML}"
    --events     "${N_EVENTS}"
    --output-dir "logs/${DETECTOR}"
)
[[ "${SWEEP}"   == "true" ]] && CMD+=(--sweep)
[[ -n "${SWEEP_DETECTORS}" ]] && read -ra _arr <<< "${SWEEP_DETECTORS}" && CMD+=(--sweep-detectors "${_arr[@]}")
[[ -n "${INCLUDE_ONLY}" ]]   && read -ra _arr <<< "${INCLUDE_ONLY}" && CMD+=(--include-only "${_arr[@]}")
[[ -n "${EXCLUDE_ONLY}" ]]   && read -ra _arr <<< "${EXCLUDE_ONLY}" && CMD+=(--exclude-only "${_arr[@]}")
[[ "${VERBOSE}" == "true" ]] && CMD+=(--verbose)
[[ -n "${DDSIM_ARGS}" ]]     && CMD+=(--ddsim-args="${DDSIM_ARGS}")

# Don't let a single failed sweep config abort the script: a sweep may produce
# valid results for 27/28 configs and fail one. We still want to upload what
# succeeded, then surface the failure via the final exit code so the job goes red.
echo "$ ${RUNNER_CPU_SET:+taskset -c $RUNNER_CPU_SET }${CMD[*]}"
set +e
${RUNNER_CPU_SET:+taskset -c "$RUNNER_CPU_SET"} "${CMD[@]}"
BENCH_RC=$?
set -e
echo "::endgroup::"

# ── 8. Write run_info.json + finalise machine_info.json ───────────────────────
echo "::group::8. Write run metadata"
python3 .github/scripts/run_info.py "logs/${DETECTOR}" \
    --detector="${DETECTOR}" \
    --sample="${SAMPLE}" \
    --date="${DATE}" \
    --platform="${K4H_PLATFORM}" \
    --release="${K4H_RELEASE}" \
    --n-events="${N_EVENTS}" \
    --sweep="${SWEEP}" \
    --detector-xml="${DETECTOR_XML}" \
    --xml-path="${XML_PATH}" \
    --configured-xml-path="${CONFIGURED_XML_PATH}" \
    --ddsim-args="${DDSIM_ARGS}" \
    --input-files="${INPUT_FILES}" \
    --steering-file="${STEERING_FILE}" \
    --resolved-steering-file="${STEERING_PATH}" \
    --sweep-detectors="${SWEEP_DETECTORS}" \
    --include-only="${INCLUDE_ONLY}" \
    --exclude-only="${EXCLUDE_ONLY}"

# machine_info.json (merge start snapshot + end-of-run dynamic fields)
python3 .github/scripts/machine_info.py finalize "logs/${DETECTOR}"
echo "::endgroup::"

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

# New EOS path: {detector}/{platform}/key4hep-{release}/{sample}/{date}
EOS_RUN="${EOS_ROOT}/${DETECTOR}/${K4H_PLATFORM}/key4hep-${K4H_RELEASE}/${SAMPLE}/${DATE}"
EOS_URL="root://${EOS_FQDN}/${EOS_RUN}"

command -v xrdfs >/dev/null || { echo "ERROR: xrdfs not found" >&2; exit 1; }
command -v xrdcp >/dev/null || { echo "ERROR: xrdcp not found" >&2; exit 1; }

xrdfs "root://${EOS_FQDN}" mkdir -p "${EOS_RUN}"

for f in "logs/${DETECTOR}"/*; do
    echo "  → $(basename "${f}")"
    xrdcp --force "${f}" "${EOS_URL}/$(basename "${f}")" \
        || { echo "ERROR: Failed to upload ${f}" >&2; exit 1; }
done
echo "Uploaded to: ${EOS_URL}"
echo "::endgroup::"

# Surface the benchmark exit code now that results are safely uploaded, so a
# failed sweep config (or any other ddsim failure) still turns the job red.
if [[ "${BENCH_RC}" -ne 0 ]]; then
    echo "ERROR: benchmark exited with code ${BENCH_RC} (one or more runs failed); results uploaded regardless" >&2
    exit "${BENCH_RC}"
fi
