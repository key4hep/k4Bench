#!/bin/bash
# Resolve one immutable LCG view for the whole nightly fan-out. LCG rotates
# weekday slots, so the generated timestamp in setup.sh -- not the slot name --
# is the release identity.
#
# A view counts only once its slot's build has finished: LCG can publish a
# slot's view before the packages it links are built, and such a view lacks
# whole packages (their Gaudi components, Python modules). The build writes
# isDone-<platform> next to the slot's packages when it completes; a view is
# complete when that marker is at most MARKER_MAX_AGE seconds older than the
# view and not newer than it.

set -euo pipefail

VIEWS_ROOT="${VIEWS_ROOT:-/cvmfs/sft-nightlies.cern.ch/lcg/views/devkey-head}"
NIGHTLIES_ROOT="${NIGHTLIES_ROOT:-/cvmfs/sft-nightlies.cern.ch/lcg/nightlies/devkey-head}"
PLATFORM_DIR="${PLATFORM_DIR:-x86_64-el9-gcc16-opt}"
LATEST_DIR="${LATEST_DIR:-latest}"
RETRIES="${RETRIES:-4}"
RETRY_INTERVAL="${RETRY_INTERVAL:-1800}"
MARKER_MAX_AGE="${MARKER_MAX_AGE:-86400}"
TODAY="$(date -u +%F)"

generated_as() {
    local generated
    generated="$(sed -n 's/^# *Generated: *//p' "$1" | head -1)"
    [[ -n "${generated}" ]] && date -u -d "${generated}" "$2"
}

release_of() {
    generated_as "$1" +%F
}

# Every weekday slot's setup.sh; `latest` only links to one of them.
slot_setups() {
    local setup
    for setup in "${VIEWS_ROOT}"/*/"${PLATFORM_DIR}"/setup.sh; do
        [[ -f "${setup}" && "${setup}" != "${VIEWS_ROOT}/${LATEST_DIR}/"* ]] && echo "${setup}"
    done
}

is_complete() {
    local slot marker view_time done_time
    slot="$(basename "$(dirname "$(dirname "$1")")")"
    marker="${NIGHTLIES_ROOT}/${slot}/isDone-${PLATFORM_DIR}"
    [[ -f "${marker}" ]] || return 1
    view_time="$(stat -c %Y "$1")"
    done_time="$(stat -c %Y "${marker}")"
    (( done_time <= view_time && view_time - done_time <= MARKER_MAX_AGE ))
}

# Prints the view of release $1, or nothing. Returns 2 when that view exists
# but its build has not finished.
find_release() {
    local setup
    while read -r setup; do
        [[ "$(release_of "${setup}" 2>/dev/null || true)" == "$1" ]] || continue
        is_complete "${setup}" || return 2
        readlink -f "${setup}"
        return
    done < <(slot_setups)
    return 1
}

# The complete view generated last, by its Generated stamp (the release
# identity), not by file time.
newest_complete() {
    local setup newest="" newest_time=0 generated_time
    while read -r setup; do
        is_complete "${setup}" || continue
        generated_time="$(generated_as "${setup}" +%s 2>/dev/null || true)"
        [[ -n "${generated_time}" ]] || continue
        if (( generated_time > newest_time )); then
            newest="${setup}"
            newest_time="${generated_time}"
        fi
    done < <(slot_setups)
    [[ -n "${newest}" ]] && readlink -f "${newest}"
}

publish() {
    echo "Release to benchmark: $1  (published today: $2)"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        printf 'release=%s\nis_today=%s\nsetup=%s\n' "$1" "$2" "$3" >> "${GITHUB_OUTPUT}"
    fi
    [[ -z "${GITHUB_STEP_SUMMARY:-}" ]] || echo "$4" >> "${GITHUB_STEP_SUMMARY}"
}

if [[ -n "${RELEASE_REQUESTED:-}" ]]; then
    if [[ ! "${RELEASE_REQUESTED}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "::error::Requested Key4hep release '${RELEASE_REQUESTED}' is not a YYYY-MM-DD date"
        exit 1
    fi
    STATUS=0
    SETUP="$(find_release "${RELEASE_REQUESTED}")" || STATUS=$?
    if (( STATUS == 2 )); then
        echo "::error::Requested Key4hep release ${RELEASE_REQUESTED} was published before its build finished (isDone-${PLATFORM_DIR} under ${NIGHTLIES_ROOT} is missing or newer than the view)"
        exit 1
    fi
    if [[ -z "${SETUP}" ]]; then
        echo "::error::Requested Key4hep release ${RELEASE_REQUESTED} is not available under ${VIEWS_ROOT}"
        exit 1
    fi
    [[ "${RELEASE_REQUESTED}" == "${TODAY}" ]] && IS_TODAY=true || IS_TODAY=false
    publish "${RELEASE_REQUESTED}" "${IS_TODAY}" "${SETUP}" \
        "Key4hep release pinned by workflow_dispatch: \`${RELEASE_REQUESTED}\`."
    exit 0
fi

ATTEMPT=0
while true; do
    STATUS=0
    SETUP="$(find_release "${TODAY}")" || STATUS=$?
    if [[ -n "${SETUP}" ]]; then
        publish "${TODAY}" true "${SETUP}" \
            "Key4hep release benchmarked: \`${TODAY}\` (published today)."
        exit 0
    fi
    (( ATTEMPT >= RETRIES )) && break
    ATTEMPT=$(( ATTEMPT + 1 ))
    if (( STATUS == 2 )); then
        echo "The ${TODAY} LCG view is published but its build has not finished; check ${ATTEMPT}/${RETRIES} in ${RETRY_INTERVAL}s"
    else
        echo "No ${TODAY} LCG view yet; check ${ATTEMPT}/${RETRIES} in ${RETRY_INTERVAL}s"
    fi
    sleep "${RETRY_INTERVAL}"
done

if (( STATUS == 2 )); then
    MISSING="The \`${TODAY}\` LCG view was published before its build finished."
else
    MISSING="No Key4hep nightly for \`${TODAY}\`."
fi

SETUP="$(newest_complete || true)"
FALLBACK="$(release_of "${SETUP}" 2>/dev/null || true)"
if [[ -z "${FALLBACK}" ]]; then
    echo "::error::${MISSING} No complete LCG view under ${VIEWS_ROOT} to fall back to"
    exit 1
fi

echo "::warning::${MISSING} Benchmarking ${FALLBACK}, the newest complete release, instead"
publish "${FALLBACK}" false "${SETUP}" \
    "**${MISSING}** Benchmarking \`${FALLBACK}\`, the newest complete release — usually a re-measurement of an already-benchmarked stack, not a new release."
