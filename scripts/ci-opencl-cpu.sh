#!/usr/bin/env bash
# Execute the OpenCL-gated test namespaces on a CPU OpenCL device (Intel's CPU runtime in CI,
# PoCL or the Intel runtime locally) so emitted kernels compile and run without a GPU. Namespaces
# are discovered by their device gate, so a new gated namespace joins the job without editing this
# script. GPU-only leaves (tuned dispatch, DPAS, 2-D block IO) keep their own capability gates.

set -euo pipefail

export RASTER_OCL_DEVICE_TYPE="${RASTER_OCL_DEVICE_TYPE:-cpu}"

if command -v clinfo >/dev/null 2>&1; then
  clinfo -l
fi

mapfile -t files < <(grep -rl 'opencl-available?' test --include='*_test.clj' | sort)
if (( ${#files[@]} == 0 )); then
  echo "no OpenCL-gated test namespaces found" >&2
  exit 2
fi

args=()
for file in "${files[@]}"; do
  ns="${file#test/}"
  ns="${ns%.clj}"
  ns="${ns//\//.}"
  ns="${ns//_/-}"
  args+=(-n "$ns")
done

echo "running ${#files[@]} OpenCL-gated namespaces on ${RASTER_OCL_DEVICE_TYPE} device(s)"
exec clojure -M:test "${args[@]}"
