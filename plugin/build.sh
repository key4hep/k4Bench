#!/bin/bash
# Build the k4Bench plugins: the DDG4 timing actions for ddsim and, in a Gaudi
# environment, the k4BenchAuditor and its allocation counter for k4run.
#
# Idempotent: skips the build if every library exists and is newer than its
# sources. Run this after sourcing the key4hep/DD4hep environment.
#
# Usage:
#   source setup.sh                  # sets up DD4hep environment
#   bash plugin/build.sh             # builds the plugins
#   bash plugin/build.sh ddg4        # builds only the DDG4 timing actions
#   bash plugin/build.sh auditor     # builds only the k4BenchAuditor
#
# A target builds only its own project, so a failing auditor build cannot take
# the DDG4 timing actions down with it, nor the reverse.

set -euo pipefail

TARGET="${1:-all}"
case "${TARGET}" in
  all|ddg4|auditor) ;;
  *)
    echo "Usage: $0 [ddg4|auditor]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"
PROC_STATS="${SCRIPT_DIR}/k4BenchProcStats.h"

# The first existing match of a library glob under the install tree, or the
# glob itself (which then fails the -f test below).
installed() {
  local match
  match="$(compgen -G "${INSTALL_DIR}/$1" | head -1 || true)"
  echo "${match:-${INSTALL_DIR}/$1}"
}

# needs_build <library> <source>...: true if the library is missing or older
# than any of its sources, its CMakeLists.txt included.
needs_build() {
  local lib="$1"
  shift
  [ -f "${lib}" ] || return 0
  local src
  for src in "$@"; do
    [ "${src}" -nt "${lib}" ] && return 0
  done
  return 1
}

STALE_DDG4=false
if [ "${TARGET}" != auditor ]; then
  LIB_EVENT="$(installed 'lib*/libk4BenchTimingAction.so')"
  LIB_REGION="$(installed 'lib*/libk4BenchRegionTimingAction.so')"
  needs_build "${LIB_EVENT}" "${SCRIPT_DIR}/k4BenchTimingAction.cpp" "${PROC_STATS}" "${SCRIPT_DIR}/CMakeLists.txt" && STALE_DDG4=true
  needs_build "${LIB_REGION}" "${SCRIPT_DIR}/k4BenchRegionTimingAction.cpp" "${SCRIPT_DIR}/CMakeLists.txt" && STALE_DDG4=true
fi

# Every Gaudi environment puts gaudirun.py on PATH; GAUDI_PLUGIN_PATH is set by
# some only (on Linux, Gaudi finds plugins through LD_LIBRARY_PATH). Without
# Gaudi the auditor cannot be built: the default target skips it, while the
# auditor target fails, so its caller learns why there is no auditor.
STALE_AUDITOR=false
if [ "${TARGET}" != ddg4 ]; then
  if [ -z "${GAUDI_PLUGIN_PATH:-}" ] && ! command -v gaudirun.py > /dev/null; then
    if [ "${TARGET}" = auditor ]; then
      echo "❌ No Gaudi environment (gaudirun.py is not on PATH): cannot build the k4BenchAuditor." >&2
      exit 1
    fi
  else
    # Gaudi installs plugins into lib/gaudi-plugins or, in older releases, lib;
    # the allocation counter is installed next to the auditor.
    LIB_AUDITOR="$(installed 'lib*/gaudi-plugins/libk4BenchAuditor.so')"
    [ -f "${LIB_AUDITOR}" ] || LIB_AUDITOR="$(installed 'lib*/libk4BenchAuditor.so')"
    LIB_COUNTER="$(dirname "${LIB_AUDITOR}")/libk4BenchAllocCounter.so"
    for lib in "${LIB_AUDITOR}" "${LIB_COUNTER}"; do
      needs_build "${lib}" "${SCRIPT_DIR}"/auditor/*.{cpp,h} "${PROC_STATS}" \
        "${SCRIPT_DIR}/auditor/CMakeLists.txt" && STALE_AUDITOR=true
    done
  fi
fi

if [ "${STALE_DDG4}" = false ] && [ "${STALE_AUDITOR}" = false ]; then
  echo "✅ k4Bench plugins are up to date."
  exit 0
fi

echo "🔄 Building k4Bench plugins..."

# build_project <source dir> <build dir>: configure, build and install one
# CMake project into the shared install prefix.
build_project() {
  cmake -S "$1" \
        -B "$2" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -Wno-dev \
        --log-level=ERROR \
        > /dev/null
  cmake --build "$2" --parallel "$(nproc)" > /dev/null
  cmake --install "$2" > /dev/null
}

mkdir -p "${BUILD_DIR}"
if [ "${STALE_DDG4}" = true ]; then
  build_project "${SCRIPT_DIR}" "${BUILD_DIR}"
fi
if [ "${STALE_AUDITOR}" = true ]; then
  build_project "${SCRIPT_DIR}/auditor" "${BUILD_DIR}/auditor"
fi

echo "✅ k4Bench plugins built:"
for lib in 'lib*/libk4BenchTimingAction.so' 'lib*/libk4BenchRegionTimingAction.so' \
    'lib*/gaudi-plugins/libk4BenchAuditor.so' 'lib*/libk4BenchAuditor.so' \
    'lib*/gaudi-plugins/libk4BenchAllocCounter.so' 'lib*/libk4BenchAllocCounter.so'; do
  path="$(installed "${lib}")"
  [ -f "${path}" ] && echo "    - ${path}"
done
exit 0
