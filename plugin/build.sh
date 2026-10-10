#!/bin/bash
# Build the k4Bench plugins: the DDG4 timing actions for ddsim and, in a Gaudi
# environment, the k4BenchAuditor for k4run.
#
# Idempotent: skips the build if every library exists and is newer than its
# sources. Run this after sourcing the key4hep/DD4hep environment.
#
# Usage:
#   source setup.sh          # sets up DD4hep environment
#   bash plugin/build.sh     # builds the plugins

set -euo pipefail

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

LIB_EVENT="$(installed 'lib*/libk4BenchTimingAction.so')"
LIB_REGION="$(installed 'lib*/libk4BenchRegionTimingAction.so')"
STALE=false
needs_build "${LIB_EVENT}" "${SCRIPT_DIR}/k4BenchTimingAction.cpp" "${PROC_STATS}" "${SCRIPT_DIR}/CMakeLists.txt" && STALE=true
needs_build "${LIB_REGION}" "${SCRIPT_DIR}/k4BenchRegionTimingAction.cpp" "${SCRIPT_DIR}/CMakeLists.txt" && STALE=true

# GAUDI_PLUGIN_PATH is set by every Gaudi environment; without Gaudi the
# auditor is not built, so it cannot be stale either.
LIB_AUDITOR=""
if [ -n "${GAUDI_PLUGIN_PATH:-}" ]; then
  LIB_AUDITOR="$(installed 'lib*/gaudi-plugins/libk4BenchAuditor.so')"
  needs_build "${LIB_AUDITOR}" "${SCRIPT_DIR}/auditor/k4BenchAuditor.cpp" "${PROC_STATS}" \
    "${SCRIPT_DIR}/auditor/CMakeLists.txt" && STALE=true
fi

if [ "${STALE}" = false ]; then
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
build_project "${SCRIPT_DIR}" "${BUILD_DIR}"
if [ -n "${LIB_AUDITOR}" ]; then
  build_project "${SCRIPT_DIR}/auditor" "${BUILD_DIR}/auditor"
fi

echo "✅ k4Bench plugins built:"
for lib in 'lib*/libk4BenchTimingAction.so' 'lib*/libk4BenchRegionTimingAction.so' 'lib*/gaudi-plugins/libk4BenchAuditor.so'; do
  path="$(installed "${lib}")"
  [ -f "${path}" ] && echo "    - ${path}"
done
exit 0
