#!/bin/bash
# Build the k4Bench timing plugins.
#
# Idempotent: skips the build if both .so files exist and are newer than
# their respective sources. Run this after sourcing the key4hep/DD4hep
# environment.
#
# Usage:
#   source setup.sh          # sets up DD4hep environment
#   bash plugin/build.sh     # builds the plugins

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"

SOURCE_EVENT="${SCRIPT_DIR}/k4BenchTimingAction.cpp"
SOURCE_REGION="${SCRIPT_DIR}/k4BenchRegionTimingAction.cpp"
LIB_EVENT="${INSTALL_DIR}/lib/libk4BenchTimingAction.so"
LIB_REGION="${INSTALL_DIR}/lib/libk4BenchRegionTimingAction.so"

needs_build() {
  local lib="$1"
  local src="$2"
  if [ ! -f "${lib}" ]; then
    return 0  # needs build
  fi
  if [ "${src}" -nt "${lib}" ]; then
    return 0  # needs build
  fi
  return 1  # up to date
}

if ! needs_build "${LIB_EVENT}" "${SOURCE_EVENT}" \
   && ! needs_build "${LIB_REGION}" "${SOURCE_REGION}"; then
  echo "✅ k4Bench timing plugins are up to date."
  exit 0
fi

echo "🔄 Building k4Bench timing plugins..."

mkdir -p "${BUILD_DIR}"

cmake -S "${SCRIPT_DIR}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -Wno-dev \
      --log-level=ERROR \
      > /dev/null

cmake --build "${BUILD_DIR}" --parallel "$(nproc)" > /dev/null
cmake --install "${BUILD_DIR}" > /dev/null

echo "✅ k4Bench timing plugins built:"
echo "    - ${LIB_EVENT}"
echo "    - ${LIB_REGION}"
