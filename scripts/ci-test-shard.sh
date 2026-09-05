#!/usr/bin/env bash

set -euo pipefail

mode="${1:-run}"
shard_count="${CIRCLE_NODE_TOTAL:-${RASTER_TEST_SHARDS:-1}}"
shard_index="${CIRCLE_NODE_INDEX:-${RASTER_TEST_SHARD:-0}}"

if [[ ! "${shard_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "shard count must be a positive integer, got: ${shard_count}" >&2
  exit 2
fi

if [[ ! "${shard_index}" =~ ^[0-9]+$ ]] || (( shard_index >= shard_count )); then
  echo "shard index must be in [0, ${shard_count}), got: ${shard_index}" >&2
  exit 2
fi

case "${mode}" in
  run|--list|--plan) ;;
  *)
    echo "usage: $0 [run|--list|--plan]" >&2
    exit 2
    ;;
esac

# Greedy largest-processing-time assignment from reviewed CI measurements. New files use a
# source-size estimate in the same units; an absent default baseline retains byte balancing.
# No test is selected or excluded based on its timing.
timings="${RASTER_TEST_TIMINGS-test/resources/ci_test_timings.tsv}"
if [[ ! -f "${timings}" && -z "${RASTER_TEST_TIMINGS+x}" ]]; then
  timings=""
fi

plan() {
  find test -type f -name '*_test.clj' -printf '%s\t%p\n' \
    | awk -F '\t' -v timings="${timings}" -f scripts/ci-test-weights.awk \
    | sort -t $'\t' -k1,1nr -k2,2 \
    | awk -F '\t' -v shards="${shard_count}" '
        BEGIN {
          for (i = 0; i < shards; i++) load[i] = 0
        }
        {
          selected = 0
          for (i = 1; i < shards; i++) {
            if (load[i] < load[selected]) selected = i
          }
          load[selected] += $1
          print selected "\t" $1 "\t" $2
        }'
}

# Capture the pipeline status. A failed planner inside process substitution would otherwise
# look like an empty successful plan to the consuming while loop.
plan_rows="$(plan)"

namespace_of() {
  awk '
    /^\(ns[[:space:]]+/ {
      sub(/^\(ns[[:space:]]+/, "")
      split($0, fields, /[[:space:]()]/)
      print fields[1]
      exit
    }' "$1"
}

if [[ "${mode}" == "--plan" ]]; then
  while IFS=$'\t' read -r node bytes file; do
    namespace="$(namespace_of "${file}")"
    if [[ -z "${namespace}" ]]; then
      echo "cannot discover test namespace in ${file}" >&2
      exit 2
    fi
    printf '%s\t%s\t%s\t%s\n' "${node}" "${bytes}" "${namespace}" "${file}"
  done <<< "${plan_rows}"
  exit 0
fi

namespaces=()
estimated_bytes=0
while IFS=$'\t' read -r node bytes file; do
  if (( node == shard_index )); then
    namespace="$(namespace_of "${file}")"
    if [[ -z "${namespace}" ]]; then
      echo "cannot discover test namespace in ${file}" >&2
      exit 2
    fi
    namespaces+=("${namespace}")
    estimated_bytes=$((estimated_bytes + bytes))
  fi
done <<< "${plan_rows}"

if (( ${#namespaces[@]} == 0 )); then
  echo "test shard ${shard_index}/${shard_count} is empty" >&2
  exit 2
fi

if [[ "${mode}" == "--list" ]]; then
  printf '%s\n' "${namespaces[@]}"
  exit 0
fi

echo "Running test shard $((shard_index + 1))/${shard_count}: ${#namespaces[@]} namespaces, ${estimated_bytes} estimated work units (timings: ${timings:-source bytes})"

runner_args=()
for namespace in "${namespaces[@]}"; do
  runner_args+=("-n" "${namespace}")
done

clojure -M:test:ci-timed "${runner_args[@]}"
