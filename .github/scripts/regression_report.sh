#!/bin/bash
#
# Builds the nightly regression report from the EOS run history, uploads it to
# EOS ({EOS_ROOT}/_reports/{YYYY-MM-DD}/report.json, read back by the
# dashboard's Regressions tab) and emails the CERN e-group when the report is
# alertable. Runs after every benchmark job of the nightly workflow — also on
# partial failures, so a crashed detector job surfaces as a FAILURE verdict
# (via its missing run) instead of vanishing.
#
# Required env vars (set by the workflow):
#   K4BENCH_DATA_URL              — WebEOS base URL of the benchmark data
#   X509_USER_CERT, X509_USER_KEY — EOS service certificate paths
#   GITHUB_RUN_URL                — link back to this Actions run
# Optional:
#   K4BENCH_REGRESSION_EGROUP     — e-group recipient (email skipped when empty)
#   K4BENCH_REGRESSION_FROM       — sender address    (email skipped when empty)
#   K4BENCH_DASHBOARD_URL         — dashboard link used in the email body
#   FORCE_SEND                    — "true" to email even on a clean night
#                                   (set by the manual workflow_dispatch)

set -euo pipefail

# Personal EOS area (same as nightly_benchmark.sh).
EOS_FQDN="eosuser.cern.ch"
EOS_ROOT="/eos/user/j/jbeirer/k4bench"

# ── 1. System dependencies ────────────────────────────────────────────────────
echo "::group::1. System dependencies"
dnf install -y --quiet voms-clients-cpp
echo "::endgroup::"

# ── 2. Key4hep stack (python + pandas/numpy/requests + xrootd clients) ────────
echo "::group::2. Key4hep stack"
set +u
source /cvmfs/sw.hsf.org/key4hep/setup.sh
set -u
echo "::endgroup::"

# ── 3. Install k4bench ────────────────────────────────────────────────────────
echo "::group::3. Install k4bench"
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
echo "::endgroup::"

# ── 4. Build the report ───────────────────────────────────────────────────────
echo "::group::4. Build the report"
python .github/scripts/regression_report.py \
    --data-url "${K4BENCH_DATA_URL}" \
    --cache-dir "${RUNNER_TEMP:-/tmp}/k4bench_cache" \
    --output-dir report
NIGHT="$(python -c 'import json; print(json.load(open("report/report.json"))["summary"]["report_night"])')"
[[ -n "${NIGHT}" ]] || { echo "ERROR: report has no report_night (no data on EOS?)" >&2; exit 1; }
echo "::endgroup::"

# ── 5. Upload report.json to EOS ──────────────────────────────────────────────
echo "::group::5. Upload to EOS"
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

EOS_REPORT_DIR="${EOS_ROOT}/_reports/${NIGHT}"
xrdfs "root://${EOS_FQDN}" mkdir -p "${EOS_REPORT_DIR}"
xrdcp --force report/report.json "root://${EOS_FQDN}/${EOS_REPORT_DIR}/report.json"
echo "Uploaded to: ${EOS_REPORT_DIR}/report.json"
echo "::endgroup::"

# ── 6. E-group email (self-gated: sends only on confirmed regressions or hard
#       failures; skips quietly when the e-group vars are unset)
echo "::group::6. E-group notification"
FORCE_ARG=()
[[ "${FORCE_SEND:-}" == "true" ]] && FORCE_ARG=(--force)
python -m k4bench.regression.notify report/report.json \
    --to "${K4BENCH_REGRESSION_EGROUP:-}" \
    --from-addr "${K4BENCH_REGRESSION_FROM:-}" \
    --dashboard-url "${K4BENCH_DASHBOARD_URL:-https://k4bench-dashboard.app.cern.ch}" \
    --actions-url "${GITHUB_RUN_URL:-}" \
    "${FORCE_ARG[@]}"
echo "::endgroup::"
