#!/bin/bash
# Build the DD4bench timing plugin.
#
# Idempotent: skips the build if the .so already exists and sources are
# unchanged.  Run this after sourcing the key4hep/DD4hep environment.
#
# Usage:
#   source setup.sh          # sets up DD4hep environment
#   bash plugin/build.sh     # builds the plugin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"
SOURCE="${SCRIPT_DIR}/DD4benchTimingAction.cpp"
LIB="${INSTALL_DIR}/lib/libDD4benchTimingAction.so"

# Skip rebuild if the .so is newer than the source
if [ -f "${LIB}" ] && [ "${LIB}" -nt "${SOURCE}" ]; then
    echo "✅ DD4bench timing plugin is up to date."
    exit 0
fi

echo "🔄 Building DD4bench timing plugin..."

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

echo "✅ DD4bench timing plugin built: ${LIB}"
