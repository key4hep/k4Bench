# shellcheck shell=bash
#
# Shared setup helpers for the nightly benchmark scripts.
#
# Sourced by both the sequential runner (.github/scripts/nightly_benchmark.sh)
# and the parallel-sweep jobs (.github/workflows/sweep-parallel.yml) so the two
# paths build an identical environment. Sourcing scripts are expected to run
# under `set -euo pipefail`.
#
# Functions export the variables that later steps rely on:
#   setup_key4hep        -> KEY4HEP_STACK, K4H_RELEASE, K4H_PLATFORM
#   resolve_geometry     -> DETECTOR_XML, DETECTOR
#   resolve_inputs       -> mutates DDSIM_ARGS (steering file + input files)

# Retry a command a few times with a short backoff. Used around network/EOS
# operations so a transient xrootd hiccup does not masquerade as a real failure.
retry() {
    local tries="${1}"; shift
    local n=1
    until "$@"; do
        if (( n >= tries )); then
            echo "ERROR: command failed after ${tries} attempts: $*" >&2
            return 1
        fi
        echo "  retry ${n}/${tries} failed, sleeping ${n}s: $*" >&2
        sleep "${n}"
        ((n++))
    done
}

setup_system_deps() {
    echo "::group::System dependencies"
    dnf install -y --quiet time voms-clients
    echo "::endgroup::"
}

setup_key4hep() {
    echo "::group::Key4hep nightly"
    set +u
    # shellcheck disable=SC1091
    source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
    set -u
    [[ -n "${KEY4HEP_STACK:-}" ]] || { echo "ERROR: KEY4HEP_STACK not set after sourcing Key4hep setup" >&2; exit 1; }
    K4H_RELEASE="$(grep -oP '\d{4}-\d{2}-\d{2}' <<< "${KEY4HEP_STACK}" | head -1 || true)"
    [[ -n "${K4H_RELEASE}" ]] || { echo "ERROR: Failed to extract Key4hep release date from KEY4HEP_STACK" >&2; exit 1; }
    K4H_PLATFORM="$(grep -oP '(?<=\d{4}-\d{2}-\d{2}\/)[^/:]+' <<< "${KEY4HEP_STACK}" | head -1 || true)"
    [[ -n "${K4H_PLATFORM}" ]] || { echo "WARNING: Could not extract platform from KEY4HEP_STACK; using 'unknown'" >&2; K4H_PLATFORM="unknown"; }
    export KEY4HEP_STACK K4H_RELEASE K4H_PLATFORM
    echo "Release : key4hep-${K4H_RELEASE}"
    echo "Platform: ${K4H_PLATFORM}"
    echo "Stack   : ${KEY4HEP_STACK}"
    echo "::endgroup::"
}

install_dd4bench() {
    echo "::group::Install dd4bench"
    DD4BENCH_REPO="$(pwd)"
    export DD4BENCH_REPO
    export LD_LIBRARY_PATH="${DD4BENCH_REPO}/plugin/install/lib:${DD4BENCH_REPO}/plugin/build:${LD_LIBRARY_PATH:-}"
    mkdir -p ~/.local/bin
    export PATH=~/.local/bin:"${PATH}"

    if [ ! -f ~/.local/bin/cvmfs-venv ]; then
        curl -sL https://raw.githubusercontent.com/jbeirer/cvmfs-venv/main/cvmfs-venv.sh \
            -o ~/.local/bin/cvmfs-venv
        chmod +x ~/.local/bin/cvmfs-venv
    fi
    cvmfs-venv py-venv
    # shellcheck disable=SC1091
    . py-venv/bin/activate
    pip install --no-build-isolation --quiet "."
    bash plugin/build.sh
    echo "::endgroup::"
}

# Resolve XML_PATH (K4GEO-relative or absolute) into DETECTOR_XML + DETECTOR.
resolve_geometry() {
    echo "::group::Resolve geometry"
    if [[ "${XML_PATH}" = /* ]]; then
        DETECTOR_XML="${XML_PATH}"
    else
        DETECTOR_XML="${K4GEO}/${XML_PATH}"
    fi
    [[ -f "${DETECTOR_XML}" ]] || { echo "ERROR: XML not found: ${DETECTOR_XML}"; exit 1; }
    DETECTOR=$(basename "${DETECTOR_XML}" .xml)
    export DETECTOR_XML DETECTOR
    echo "Detector : ${DETECTOR}"
    echo "XML      : ${DETECTOR_XML}"
    echo "::endgroup::"
}

# Prepend an optional steering file and input files to DDSIM_ARGS. Run after
# the Key4hep stack is sourced so $VAR references in paths expand correctly.
resolve_inputs() {
    echo "::group::Resolve inputs"
    if [[ -n "${STEERING_FILE:-}" ]]; then
        STEERING_PATH=$(python3 -c "import os, sys; print(os.path.expandvars(sys.argv[1]))" "${STEERING_FILE}")
        [[ -f "${STEERING_PATH}" ]] || { echo "ERROR: steering file not found: ${STEERING_PATH}"; exit 1; }
        DDSIM_ARGS="--steeringFile ${STEERING_PATH} ${DDSIM_ARGS:-}"
        echo "Steering : ${STEERING_PATH}"
    fi

    if [[ -n "${INPUT_FILES:-}" ]]; then
        # HepMC inputs can't be streamed over xrootd (ROOT mis-parses the text as
        # a ROOT file -> SIGSEGV), so fetch to a local path first.
        LOCAL_INPUT="/tmp/$(basename "${INPUT_FILES}")"
        retry 3 xrdcp --force "${INPUT_FILES}" "${LOCAL_INPUT}"
        DDSIM_ARGS="--inputFiles ${LOCAL_INPUT} ${DDSIM_ARGS:-}"
        echo "Inputs   : ${LOCAL_INPUT}"
    fi
    export DDSIM_ARGS
    echo "::endgroup::"
}

# Initialise a VOMS proxy from the mounted service certificate so xrdcp/xrdfs
# can authenticate to EOS. Expects X509_USER_CERT / X509_USER_KEY in the env.
setup_eos_proxy() {
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
}

# Upload every file in a local dir to an EOS run path (idempotent mkdir + retry).
#   eos_upload_dir <local_dir> <eos_fqdn> <eos_run_path>
eos_upload_dir() {
    local local_dir="${1}" fqdn="${2}" run="${3}"
    command -v xrdfs >/dev/null || { echo "ERROR: xrdfs not found" >&2; exit 1; }
    command -v xrdcp >/dev/null || { echo "ERROR: xrdcp not found" >&2; exit 1; }
    retry 3 xrdfs "root://${fqdn}" mkdir -p "${run}"
    local f
    for f in "${local_dir}"/*; do
        [[ -e "${f}" ]] || continue
        echo "  -> $(basename "${f}")"
        retry 3 xrdcp --force "${f}" "root://${fqdn}/${run}/$(basename "${f}")"
    done
}
