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

export RASTER_TEST_SELECTION=opencl
export RASTER_TEST_TIMINGS="${RASTER_TEST_TIMINGS:-test/resources/ci_opencl_timings.tsv}"
exec scripts/ci-test-shard.sh
