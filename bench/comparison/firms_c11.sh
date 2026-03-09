#!/usr/bin/env bash
#
# bench/firms_c11.sh — Build and time the FIRMS C++11 reference implementation
#
# Usage (from project root):
#   bash bench/firms_c11.sh                    # use default 120M agents
#   FIRMS_N=10000000 bash bench/firms_c11.sh   # patch to 10M agents
#
# The C++ code hardcodes agent count in FIRMS-Declarations.h.
# This script optionally patches cTotalNumberOfAgents before building.
#
# Output: wall time and CPU time printed by FIRMS itself.
#
set -euo pipefail

FIRMS_DIR="${FIRMS_DIR:-$(cd "$(dirname "$0")/../../FIRMS_C11" && pwd)}"
FIRMS_N="${FIRMS_N:-}"       # leave empty to use default (120M)
FIRMS_PERIODS="${FIRMS_PERIODS:-10}"  # default 10 periods for per-period timing

echo "=== FIRMS C++11 reference benchmark ==="
echo "Source: $FIRMS_DIR"

cd "$FIRMS_DIR"

# Optionally patch agent count
DECL="FIRMS-Declarations.h"
cp -f "$DECL" "${DECL}.bak"
if [[ -n "$FIRMS_N" ]]; then
  echo "Patching cTotalNumberOfAgents to $FIRMS_N..."
  sed -i "s/const int cTotalNumberOfAgents = [0-9]*/const int cTotalNumberOfAgents = $FIRMS_N/" "$DECL"
fi
echo "Patching cEndingTimeDefault to $FIRMS_PERIODS periods..."
sed -i "s/const int cEndingTimeDefault = [0-9]*/const int cEndingTimeDefault = $FIRMS_PERIODS/" "$DECL"

# Build with optimizations (Makefile has no -O flags — we add them)
echo "Building with -O3 -march=native..."
g++ --std=c++14 -pthread -O3 -march=native \
  "FIRMS-Agent Population.cpp" \
  "FIRMS-Agent.cpp" \
  "FIRMS-Data.cpp" \
  "FIRMS-Economy.cpp" \
  "FIRMS-Firm Population.cpp" \
  "FIRMS-Firm.cpp" \
  "FIRMS-IO.cpp" \
  "FIRMS-Location Population.cpp" \
  "FIRMS-Location.cpp" \
  "FIRMS-Random Numbers.cpp" \
  "FIRMS-Tests.cpp" \
  "FIRMS-Utilities.cpp" \
  "FIRMS.cpp" \
  -o firms_bench

echo "Built. Running $FIRMS_PERIODS periods..."
echo ""
time ./firms_bench

# Restore patched file
mv -f "${DECL}.bak" "$DECL"
