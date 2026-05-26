#!/bin/bash
#
# Runs a single dd4bench benchmark and uploads results to CERN EOS.
# Detector configuration is read from .github/benchmarks/${BENCHMARK_CONFIG}.yml.
#
# Required env vars (set by the workflow):
#   BENCHMARK_CONFIG  — config file stem, e.g. "ALLEGRO_o1_v03"
#   X509_USER_CERT, X509_USER_KEY — EOS service certificate paths
#   GITHUB_RUN_ID, GITHUB_SHA, GITHUB_REPOSITORY, GITHUB_SERVER_URL
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

# TODO: switch to eospublic once directory creation is allowed there (!d)
# (See eos root://eospublic.cern.ch attr ls /eos/experiment/fcc/ee/dd4bench)

# EOS_FQDN="eospublic.cern.ch"
# EOS_ROOT="/eos/experiment/fcc/ee/dd4bench"
EOS_FQDN="eosuser.cern.ch"
EOS_ROOT="/eos/user/j/jbeirer/dd4bench"

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
SAMPLE=$(_cfg sample)
DDSIM_ARGS=$(_cfg ddsim_args "")
VERBOSE=$(_cfg verbose "false")
SWEEP=$(_cfg sweep "false")
INCLUDE_ONLY=$(_cfg_list include_only)
EXCLUDE_ONLY=$(_cfg_list exclude_only)
N_EVENTS=$(_cfg n_events)
[[ "${N_EVENTS}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: 'n_events' must be a positive integer in ${CONFIG_FILE}"; exit 1; }
[[ -n "${SAMPLE}" ]]                  || { echo "ERROR: 'sample' must be set in ${CONFIG_FILE}"; exit 1; }
# Validate sample slug: filesystem-safe characters only
[[ "${SAMPLE}" =~ ^[A-Za-z0-9_.+-]+$ ]] || { echo "ERROR: 'sample' must only contain [A-Za-z0-9_.-+] — got: '${SAMPLE}'"; exit 1; }

# sweep / include_only / exclude_only are mutually exclusive
SWEEP_MODES=0
[[ "${SWEEP}"         == "true" ]] && (( SWEEP_MODES++ )) || true
[[ -n "${INCLUDE_ONLY}" ]]         && (( SWEEP_MODES++ )) || true
[[ -n "${EXCLUDE_ONLY}" ]]         && (( SWEEP_MODES++ )) || true
(( SWEEP_MODES <= 1 )) || { echo "ERROR: sweep, include_only, and exclude_only are mutually exclusive"; exit 1; }

echo "  xml          : ${XML_PATH}"
echo "  sample       : ${SAMPLE}"
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
# Extract platform tag (path component right after the release date, e.g. x86_64-el9-gcc14-opt)
K4H_PLATFORM="$(grep -oP '(?<=\d{4}-\d{2}-\d{2}\/)[^/:]+' <<< "${KEY4HEP_STACK}" | head -1 || true)"
[[ -n "${K4H_PLATFORM}" ]] || { echo "WARNING: Could not extract platform from KEY4HEP_STACK; using 'unknown'" >&2; K4H_PLATFORM="unknown"; }
echo "Release : key4hep-${K4H_RELEASE}"
echo "Platform: ${K4H_PLATFORM}"
echo "Stack   : ${KEY4HEP_STACK}"
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

# ── 6. Collect machine info (start snapshot, before benchmark) ────────────────
echo "::group::6. Collect machine info (start)"
mkdir -p "logs/${DETECTOR}"
python3 - "${DETECTOR}" <<'PYEOF'
import json, os, platform, sys

def _read(path, default=''):
    try:
        with open(path) as f: return f.read()
    except Exception: return default

detector = sys.argv[1]

# CPU
cpuinfo      = _read('/proc/cpuinfo')
cpu_model    = next((l.split(':',1)[1].strip() for l in cpuinfo.splitlines() if 'model name' in l), 'unknown')
cpu_logical  = cpuinfo.count('processor\t:')
phys_ids     = {l.split(':',1)[1].strip() for l in cpuinfo.splitlines() if 'physical id' in l}
cpu_physical = len(phys_ids) if phys_ids else cpu_logical
cpu_flags    = next((l.split(':',1)[1].strip().split() for l in cpuinfo.splitlines() if l.startswith('flags')), [])

# Memory
meminfo = _read('/proc/meminfo')
mem = {}
for line in meminfo.splitlines():
    p = line.split()
    if len(p) >= 2:
        try: mem[p[0].rstrip(':')] = int(p[1])
        except ValueError: pass

# Load average
loadavg = _read('/proc/loadavg').split()

# OS
os_release = _read('/etc/os-release')
os_name = next(
    (l.split('=',1)[1].strip('"') for l in os_release.splitlines() if l.startswith('PRETTY_NAME=')),
    'unknown',
)

info = {
    "cpu_model":              cpu_model,
    "cpu_physical_cores":     cpu_physical,
    "cpu_logical_cores":      cpu_logical,
    "cpu_flags":              cpu_flags[:30],
    "ram_total_gb":           round(mem.get('MemTotal',    0) / 1024**2, 2),
    "ram_available_gb_start": round(mem.get('MemAvailable',0) / 1024**2, 2),
    "swap_total_gb":          round(mem.get('SwapTotal',   0) / 1024**2, 2),
    "load_avg_1m_start":      float(loadavg[0]) if len(loadavg) > 0 else None,
    "load_avg_5m_start":      float(loadavg[1]) if len(loadavg) > 1 else None,
    "kernel":                 platform.release(),
    "os":                     os_name,
    "hostname":               os.uname().nodename,
    "in_container":           os.path.exists('/.dockerenv'),
}
with open(f"logs/{detector}/_machine_info_start.json", "w") as f:
    json.dump(info, f, indent=2)
print(f"cpu_model        : {info['cpu_model']}")
print(f"cpu_logical_cores: {info['cpu_logical_cores']}")
print(f"ram_total_gb     : {info['ram_total_gb']:.2f} GB")
print(f"ram_available    : {info['ram_available_gb_start']:.2f} GB")
print(f"load_avg_1m      : {info['load_avg_1m_start']}")
PYEOF
echo "::endgroup::"

# ── 7. Run benchmark ──────────────────────────────────────────────────────────
echo "::group::7. Run benchmark"
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
"${CMD[@]}"
echo "::endgroup::"

# ── 8. Write run_info.json + finalise machine_info.json ───────────────────────
echo "::group::8. Write run metadata"
DATE=$(date +%Y-%m-%d)

CONFIGS_JSON=$(
    find "logs/${DETECTOR}" -maxdepth 1 -name '*_results.csv' -print0 2>/dev/null \
    | xargs -0 -r -I{} basename {} _results.csv \
    | python3 -c "import sys, json; print(json.dumps(sys.stdin.read().split()))"
)

python3 - "${DETECTOR}" "${SAMPLE}" "${DATE}" "${K4H_PLATFORM}" "${K4H_RELEASE}" \
          "${N_EVENTS}" "${SWEEP}" <<PYEOF
import json, os, sys

detector   = sys.argv[1]
sample     = sys.argv[2]
date       = sys.argv[3]
platform   = sys.argv[4]
k4h_rel    = sys.argv[5]
n_events   = int(sys.argv[6])
sweep      = sys.argv[7] == "true"

# ── run_info.json ──────────────────────────────────────────────────────────
run_info = {
    "date":             date,
    "platform":         platform,
    "k4h_release":      f"key4hep-{k4h_rel}",
    "k4h_release_date": k4h_rel,
    "detector":         detector,
    "sample":           sample,
    "github_run_id":    os.environ["GITHUB_RUN_ID"],
    "github_run_url": (
        f"{os.environ['GITHUB_SERVER_URL']}"
        f"/{os.environ['GITHUB_REPOSITORY']}"
        f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    ),
    "commit_sha":       os.environ["GITHUB_SHA"],
    "n_events":         n_events,
    "sweep":            sweep,
    "configs":          ${CONFIGS_JSON},
}
with open(f"logs/{detector}/run_info.json", "w") as f:
    json.dump(run_info, f, indent=2)
print(f"Written: logs/{detector}/run_info.json")

# ── machine_info.json: merge start snapshot + end-of-run dynamic fields ────
def _read(path, default=''):
    try:
        with open(path) as fh: return fh.read()
    except Exception: return default

start_path = f"logs/{detector}/_machine_info_start.json"
with open(start_path) as fh:
    machine_info = json.load(fh)
os.remove(start_path)

meminfo = _read('/proc/meminfo')
mem = {}
for line in meminfo.splitlines():
    p = line.split()
    if len(p) >= 2:
        try: mem[p[0].rstrip(':')] = int(p[1])
        except ValueError: pass
loadavg = _read('/proc/loadavg').split()

machine_info["ram_available_gb_end"] = round(mem.get('MemAvailable', 0) / 1024**2, 2)
machine_info["load_avg_1m_end"]      = float(loadavg[0]) if len(loadavg) > 0 else None
machine_info["load_avg_5m_end"]      = float(loadavg[1]) if len(loadavg) > 1 else None

with open(f"logs/{detector}/machine_info.json", "w") as fh:
    json.dump(machine_info, fh, indent=2)
print(f"Written: logs/{detector}/machine_info.json")
PYEOF
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

DATE=$(date +%Y-%m-%d)
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
