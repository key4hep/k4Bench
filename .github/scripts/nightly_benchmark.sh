#!/bin/bash
#
# Runs a single dd4bench benchmark and uploads results to CERN EOS.
# Detector configuration is read from .github/benchmarks/${BENCHMARK_CONFIG}.yml.
#
# Required env vars (set by the workflow):
#   BENCHMARK_CONFIG  — config file stem, e.g. "ALLEGRO_o1_v03"
#   X509_USER_CERT, X509_USER_KEY — EOS service certificate paths
#   GITHUB_RUN_ID, GITHUB_SHA, GITHUB_REPOSITORY, GITHUB_SERVER_URL

set -euo pipefail

EOS_FQDN="eospublic.cern.ch"
EOS_ROOT="/eos/experiment/fcc/ee/dd4bench"
CONFIG_FILE=".github/benchmarks/${BENCHMARK_CONFIG}.yml"

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "::group::1. System dependencies"
dnf install -y --quiet time python3-pyyaml voms-clients
echo "::endgroup::"

# ── 2. Read detector config ───────────────────────────────────────────────────
echo "::group::2. Detector config (${CONFIG_FILE})"

# Read a scalar value from the YAML config.  Usage: _cfg <key> [default]
_cfg() {
    python3 -c "
import sys, yaml
try:
    with open('${CONFIG_FILE}') as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f'ERROR: Failed to read config: {e}', file=sys.stderr)
    sys.exit(1)
val = cfg.get(sys.argv[1])
if val is None:
    print(sys.argv[2] if len(sys.argv) > 2 else '')
elif isinstance(val, bool):
    print(str(val).lower())
else:
    print(str(val).strip())
" "$@"
}

# Read a list value as space-separated tokens.  Usage: _cfg_list <key>
_cfg_list() {
    python3 -c "
import sys, yaml
try:
    with open('${CONFIG_FILE}') as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f'ERROR: Failed to read config: {e}', file=sys.stderr)
    sys.exit(1)
val = cfg.get(sys.argv[1])
if isinstance(val, list):
    print(' '.join(str(v) for v in val))
elif val is not None:
    print(str(val).strip())
" "$@"
}

XML_PATH=$(_cfg xml)
DDSIM_ARGS=$(_cfg ddsim_args "")
VERBOSE=$(_cfg verbose "false")
SWEEP=$(_cfg sweep "false")
INCLUDE_ONLY=$(_cfg_list include_only)
EXCLUDE_ONLY=$(_cfg_list exclude_only)
N_EVENTS=$(_cfg n_events)
[[ "${N_EVENTS}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: 'n_events' must be a positive integer in ${CONFIG_FILE}"; exit 1; }

# sweep / include_only / exclude_only are mutually exclusive
SWEEP_MODES=0
[[ "${SWEEP}"         == "true" ]] && (( SWEEP_MODES++ )) || true
[[ -n "${INCLUDE_ONLY}" ]]         && (( SWEEP_MODES++ )) || true
[[ -n "${EXCLUDE_ONLY}" ]]         && (( SWEEP_MODES++ )) || true
(( SWEEP_MODES <= 1 )) || { echo "ERROR: sweep, include_only, and exclude_only are mutually exclusive"; exit 1; }

echo "  xml          : ${XML_PATH}"
echo "  n_events     : ${N_EVENTS}"
echo "  verbose      : ${VERBOSE}"
echo "  sweep        : ${SWEEP}"
echo "  include_only : ${INCLUDE_ONLY:-<none>}"
echo "  exclude_only : ${EXCLUDE_ONLY:-<none>}"
echo "  ddsim_args   : ${DDSIM_ARGS:-<none>}"
echo "::endgroup::"

# ── 3. Key4hep nightly ────────────────────────────────────────────────────────
echo "::group::3. Key4hep nightly"
set +u
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
set -u
[[ -n "${KEY4HEP_STACK:-}" ]] || { echo "ERROR: KEY4HEP_STACK not set after sourcing Key4hep setup" >&2; exit 1; }
K4H_RELEASE="$(grep -oP '\d{4}-\d{2}-\d{2}' <<< "${KEY4HEP_STACK}" | head -1 || true)"
[[ -n "${K4H_RELEASE}" ]] || { echo "ERROR: Failed to extract Key4hep release date from KEY4HEP_STACK" >&2; exit 1; }
echo "Release: key4hep-${K4H_RELEASE}"
echo "Stack  : ${KEY4HEP_STACK}"
echo "::endgroup::"

# ── 4. Install dd4bench ───────────────────────────────────────────────────────
echo "::group::4. Install dd4bench"
export DD4BENCH_REPO="$(pwd)"
export LD_LIBRARY_PATH="${DD4BENCH_REPO}/plugin/install/lib:${DD4BENCH_REPO}/plugin/build:${LD_LIBRARY_PATH:-}"
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

# ── 5. Resolve geometry ───────────────────────────────────────────────────────
echo "::group::5. Resolve geometry"
if [[ "${XML_PATH}" = /* ]]; then
    DETECTOR_XML="${XML_PATH}"
else
    DETECTOR_XML="${K4GEO}/${XML_PATH}"
fi
[[ -f "${DETECTOR_XML}" ]] || { echo "ERROR: XML not found: ${DETECTOR_XML}"; exit 1; }
DETECTOR=$(basename "${DETECTOR_XML}" .xml)
echo "Detector : ${DETECTOR}"
echo "XML      : ${DETECTOR_XML}"
echo "::endgroup::"

# ── 6. Run benchmark ──────────────────────────────────────────────────────────
echo "::group::6. Run benchmark"
CMD=(dd4bench
    --xml        "${DETECTOR_XML}"
    --events     "${N_EVENTS}"
    --output-dir "logs/${DETECTOR}"
)
[[ "${SWEEP}"   == "true" ]] && CMD+=(--sweep)
[[ -n "${INCLUDE_ONLY}" ]]   && read -ra _arr <<< "${INCLUDE_ONLY}" && CMD+=(--include-only "${_arr[@]}")
[[ -n "${EXCLUDE_ONLY}" ]]   && read -ra _arr <<< "${EXCLUDE_ONLY}" && CMD+=(--exclude-only "${_arr[@]}")
[[ "${VERBOSE}" == "true" ]] && CMD+=(--verbose)
[[ -n "${DDSIM_ARGS}" ]]     && CMD+=(--ddsim-args="${DDSIM_ARGS}")

echo "$ ${CMD[*]}"
# TEMP: replace the two lines below with '"${CMD[@]}"' to re-enable the benchmark
mkdir -p "logs/${DETECTOR}"
echo "plugin,events,time_s" > "logs/${DETECTOR}/dummy_results.csv"
echo "::endgroup::"

# ── 7. Write run metadata ─────────────────────────────────────────────────────
echo "::group::7. Write run metadata"
DATE=$(date +%Y-%m-%d)
RUN_LABEL="${DATE}_key4hep-${K4H_RELEASE}"

CONFIGS_JSON=$(
    find "logs/${DETECTOR}" -maxdepth 1 -name '*_results.csv' -print0 2>/dev/null \
    | xargs -0 -r -I{} basename {} _results.csv \
    | python3 -c "import sys, json; print(json.dumps(sys.stdin.read().split()))"
)

python3 - <<PYEOF
import json, os

run_info = {
    "date":            "${DATE}",
    "detector":        "${DETECTOR}",
    "key4hep_release": "key4hep-${K4H_RELEASE}",
    "github_run_id":   os.environ["GITHUB_RUN_ID"],
    "github_run_url":  (
        f"{os.environ['GITHUB_SERVER_URL']}"
        f"/{os.environ['GITHUB_REPOSITORY']}"
        f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    ),
    "commit_sha":      os.environ["GITHUB_SHA"],
    "n_events":        int("${N_EVENTS}"),
    "sweep":           "${SWEEP}" == "true",
    "configs":         ${CONFIGS_JSON},
}
with open("logs/${DETECTOR}/run_info.json", "w") as f:
    json.dump(run_info, f, indent=2)
print("Written: logs/${DETECTOR}/run_info.json")
PYEOF
echo "::endgroup::"

# ── 8. Upload to EOS ──────────────────────────────────────────────────────────
echo "::group::8. Upload to EOS"
export X509_CERT_DIR=/cvmfs/grid.cern.ch/etc/grid-security/certificates
export X509_VOMS_DIR=/cvmfs/grid.cern.ch/etc/grid-security/vomsdir
export VOMS_USERCONF=/cvmfs/grid.cern.ch/etc/vomses
export X509_USER_PROXY=/tmp/x509_proxy
# Initialize VOMS proxy for EOS upload
voms-proxy-init \
  --cert "${X509_USER_CERT}" \
  --key "${X509_USER_KEY}" \
  --out "${X509_USER_PROXY}"

EOS_RUN="${EOS_ROOT}/runs/${DETECTOR}/${RUN_LABEL}"
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
