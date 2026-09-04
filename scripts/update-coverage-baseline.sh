#!/usr/bin/env bash
# Regenerate the typed-route coverage baseline on a CPU OpenCL device (the same class of device
# the CI gate uses), so route facts in the baseline match what CI measures.
set -euo pipefail
export RASTER_OCL_DEVICE_TYPE="${RASTER_OCL_DEVICE_TYPE:-cpu}"
exec clojure -M:dev -m raster.compiler.coverage update
