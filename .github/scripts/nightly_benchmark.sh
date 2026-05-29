#!/bin/bash
#
# Runs a single dd4bench benchmark (sequential sweep or single run) and uploads
# results to CERN EOS. All configuration arrives via environment variables; the
# matrix in .github/workflows/benchmark-detector.yml expands
# .github/benchmarks/*.yml into a flat set of jobs (see list_benchmarks.py).
#
# Failure handling: the benchmark's exit code is captured rather than allowed to
# abort the script, so results — including failed configs' CSV + .log — are
# always written and uploaded. The script then exits with that code, so the CI
# job still goes red on a benchmark failure while the data reaches EOS. Pure
# infra errors (missing XML, Key4hep, EOS auth) still hard-fail with no upload.
#
# Required env vars (set by the workflow):
#   BENCHMARK_CONFIG  — config file stem, e.g. "ALLEGRO_o1_v03"
#   BENCHMARK_SAMPLE  — sample name, e.g. "single_e-_10GeV"
#   XML_PATH          — detector geometry, $K4GEO-relative or absolute
#   N_EVENTS          — positive integer
#   DDSIM_ARGS        — verbatim ddsim flags (string, may be empty)
#   INPUT_FILES       — space-separated HepMC paths (may be empty)
#   STEERING_FILE     — optional ddsim --steeringFile path; $VAR expansion supported
#   SWEEP             — "true"/"false"
#   VERBOSE           — "true"/"false"
#   TIMEOUT           — optional per-run wall-clock limit in seconds (may be empty)
#   INCLUDE_ONLY      — space-separated subdetector names (may be empty)
#   EXCLUDE_ONLY      — space-separated subdetector names (may be empty)
#   X509_USER_CERT, X509_USER_KEY — EOS service certificate paths
#   GITHUB_RUN_ID, GITHUB_SHA, GITHUB_REPOSITORY, GITHUB_SERVER_URL
#
# Optional (set by the parallel-sweep path, sweep-parallel.yml, so every variant
# job writes into one shared EOS run dir; unset for the sequential path):
#   EOS_DATE, EOS_PLATFORM, EOS_RELEASE — pinned run-path components
#   PARALLEL                            — "true" to tag run_info.json
#
# EOS layout written by this script:
#   {EOS_ROOT}/{detector}/{platform}/key4hep-{release}/{sample}/{YYYY-MM-DD}/
#     run_info.json  machine_info.json
#     {config}_results.csv  {config}_events.json  {config}_regions.json  {config}.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.github/scripts/lib_setup.sh
source "${SCRIPT_DIR}/lib_setup.sh"

# TODO: switch to eospublic once directory creation is allowed there (!d)
EOS_FQDN="eosuser.cern.ch"
EOS_ROOT="/eos/user/j/jbeirer/dd4bench"

SAMPLE="${BENCHMARK_SAMPLE}"

# ── Job parameters ─────────────────────────────────────────────────────────────
echo "::group::Job parameters"
echo "  config       : ${BENCHMARK_CONFIG}"
echo "  sample       : ${SAMPLE}"
echo "  xml          : ${XML_PATH}"
echo "  n_events     : ${N_EVENTS}"
echo "  verbose      : ${VERBOSE}"
echo "  sweep        : ${SWEEP}"
echo "  timeout      : ${TIMEOUT:-<none>}"
echo "  include_only : ${INCLUDE_ONLY:-<none>}"
echo "  exclude_only : ${EXCLUDE_ONLY:-<none>}"
echo "  input_files  : ${INPUT_FILES:-<none>}"
echo "  steering_file: ${STEERING_FILE:-<none>}"
echo "  ddsim_args   : ${DDSIM_ARGS:-<none>}"
echo "::endgroup::"

# ── Environment ────────────────────────────────────────────────────────────────
setup_system_deps
setup_key4hep
install_dd4bench
resolve_geometry
resolve_inputs

# Path components. A parallel-sweep variant job inherits EOS_DATE/EOS_PLATFORM/
# EOS_RELEASE pinned by the discover job, so every variant lands in one EOS run
# dir. The sequential path leaves them unset and derives them here (date captured
# once so run_info.json and the upload path agree across a midnight boundary).
DATE="${EOS_DATE:-$(date +%Y-%m-%d)}"
PLATFORM="${EOS_PLATFORM:-${K4H_PLATFORM}}"
RELEASE="${EOS_RELEASE:-${K4H_RELEASE}}"
LOG_DIR="logs/${DETECTOR}"

# ── Collect machine info (start snapshot, before benchmark) ────────────────────
echo "::group::Collect machine info (start)"
python3 .github/scripts/machine_info.py start "${LOG_DIR}"
echo "::endgroup::"

# ── Run benchmark (capture exit code; never abort the upload below) ────────────
echo "::group::Run benchmark"
CMD=(dd4bench
    --xml        "${DETECTOR_XML}"
    --events     "${N_EVENTS}"
    --output-dir "${LOG_DIR}"
)
[[ "${SWEEP}"   == "true" ]] && CMD+=(--sweep)
[[ -n "${INCLUDE_ONLY}" ]]   && read -ra _arr <<< "${INCLUDE_ONLY}" && CMD+=(--include-only "${_arr[@]}")
[[ -n "${EXCLUDE_ONLY}" ]]   && read -ra _arr <<< "${EXCLUDE_ONLY}" && CMD+=(--exclude-only "${_arr[@]}")
[[ "${VERBOSE}" == "true" ]] && CMD+=(--verbose)
[[ -n "${TIMEOUT:-}" ]]      && CMD+=(--timeout "${TIMEOUT}")
[[ -n "${DDSIM_ARGS}" ]]     && CMD+=(--ddsim-args="${DDSIM_ARGS}")

echo "$ ${CMD[*]}"
BENCH_RC=0
"${CMD[@]}" || BENCH_RC=$?
echo "benchmark exit code: ${BENCH_RC}"
echo "::endgroup::"

# ── Write run metadata (always, even on benchmark failure) ─────────────────────
echo "::group::Write run metadata"
python3 .github/scripts/write_run_info.py \
    --results-dir "${LOG_DIR}" \
    --detector    "${DETECTOR}" \
    --sample      "${SAMPLE}" \
    --date        "${DATE}" \
    --platform    "${PLATFORM}" \
    --release     "${RELEASE}" \
    --n-events    "${N_EVENTS}" \
    --sweep       "${SWEEP}" \
    --parallel    "${PARALLEL:-false}"
python3 .github/scripts/machine_info.py finalize "${LOG_DIR}"
echo "::endgroup::"

# ── Upload to EOS (always) ─────────────────────────────────────────────────────
echo "::group::Upload to EOS"
setup_eos_proxy
EOS_RUN="${EOS_ROOT}/${DETECTOR}/${PLATFORM}/key4hep-${RELEASE}/${SAMPLE}/${DATE}"
eos_upload_dir "${LOG_DIR}" "${EOS_FQDN}" "${EOS_RUN}"
echo "Uploaded to: root://${EOS_FQDN}/${EOS_RUN}"
echo "::endgroup::"

# Reflect the benchmark outcome in the job status now that data is safely on EOS.
exit "${BENCH_RC}"
