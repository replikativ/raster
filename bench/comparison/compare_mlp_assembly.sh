#!/usr/bin/env bash
# Compare Raster C2 vs JAX XLA assembly for MLP 784-128-10 train step.
#
# Produces a comparison directory with timing, HLO, bytecode, and native asm.
#
# Usage:
#   bench/comparison/compare_mlp_assembly.sh [outdir]
#
# Prerequisites:
#   - Valhalla JDK (source valhalla-env.sh)
#   - JAX installed (pip install jax jaxlib)
#   - hsdis plugin (optional, for C2 native assembly)

set -euo pipefail

OUTDIR="${1:-/tmp/raster_vs_xla}"
mkdir -p "$OUTDIR"

echo "========================================"
echo " MLP 784-128-10 Assembly Comparison"
echo " Output: $OUTDIR"
echo "========================================"

# ── JAX / XLA ──────────────────────────────────────────────────
echo ""
echo "=== JAX / XLA ==="
OPENBLAS_NUM_THREADS=1 python3 bench/comparison/jax_mlp_assembly.py "$OUTDIR"

# For full native assembly, uncomment:
# XLA_FLAGS="--xla_dump_to=$OUTDIR/xla_native --xla_dump_hlo_as_text" \
# OPENBLAS_NUM_THREADS=1 python3 bench/comparison/jax_mlp_assembly.py "$OUTDIR"

# ── Raster / C2 ───────────────────────────────────────────────
echo ""
echo "=== Raster / C2 ==="

# Use Valhalla JDK if available
if [ -f valhalla-env.sh ]; then
    source valhalla-env.sh
fi

OPENBLAS_NUM_THREADS=1 clojure \
    -J--enable-preview \
    -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \
    -J--enable-native-access=ALL-UNNAMED \
    -J--add-modules=jdk.incubator.vector \
    -M:bench -m raster.mlp-assembly "$OUTDIR"

# ── Summary ────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Summary"
echo "========================================"
echo ""
echo "--- JAX timing ---"
cat "$OUTDIR/xla_timing.txt"
echo ""
echo "--- Raster timing ---"
cat "$OUTDIR/c2_timing.txt"
echo ""
echo "--- XLA fusion ---"
head -5 "$OUTDIR/xla_fusion_summary.txt"
echo ""
echo "--- Raster pipeline ---"
cat "$OUTDIR/c2_pipeline_stats.txt"
echo ""
echo "Files:"
ls -la "$OUTDIR"/c2_*.txt "$OUTDIR"/xla_*.txt 2>/dev/null
echo ""
echo "To inspect native assembly:"
echo "  grep -l vmulpd $OUTDIR/c2_native_*.txt  # AVX2 double multiply"
echo "  grep -l vaddpd $OUTDIR/c2_native_*.txt  # AVX2 double add"
echo "  grep -l vfmadd $OUTDIR/c2_native_*.txt  # FMA instructions"
