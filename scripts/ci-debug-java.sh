#!/usr/bin/env bash
# Clojure's JAVA_CMD adapter: preserve its argument plan, including classpath and JVM options.
# Debug only the timed test runner; dependency/classpath helper JVMs execute normally.
set -euo pipefail

task_java=${RASTER_CI_REAL_JAVA:?RASTER_CI_REAL_JAVA must name the original Java executable}
task_timed_runner=false
for task_arg in "$@"; do
  if [[ "$task_arg" == raster.ci.timed-runner ]]; then
    task_timed_runner=true
  fi
done
if [[ "$task_timed_runner" != true ]]; then
  exec "$task_java" "$@"
fi

task_shard=${CIRCLE_NODE_INDEX:-0}
if [[ ! "$task_shard" =~ ^[0-9]+$ ]]; then
  echo 'Native diagnostic shard index must be nonnegative.' >&2
  exit 2
fi
task_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p test-results
exec gdb --batch --nx --return-child-result \
  -ex "set logging file test-results/native-exit-${task_shard}.log" \
  -ex 'set logging overwrite on' -ex 'set logging enabled on' \
  --command "$task_script_dir/ci-native-exit.gdb" --args "$task_java" "$@"
