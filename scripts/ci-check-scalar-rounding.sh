#!/usr/bin/env bash
# Check Raster-generated scalar rounding boundaries without vendor hardware.
set -euo pipefail
target=${1:?Expected cuda or hip}
source_root=${2:-gpu-compile-gates}
rounding_workdir=$(mktemp -d "${TMPDIR:-/tmp}/raster-scalar-rounding.XXXXXX")
trap 'rm -r -- "$rounding_workdir"' EXIT
for precision in f32 f64; do
  for realization in decomposed fused; do
    stem="scalar-product-${precision}-${realization}"
    assembly="$rounding_workdir/${stem}.s"
    case "$target" in
      cuda)
        nvcc -ptx -arch=sm_80 -o "$assembly" "$source_root/cuda/${stem}.cu"
        fused_pattern="fma\\.rn\\.${precision}"
        product_pattern="mul\\.rn\\.${precision}"
        add_pattern="add(\\.rn)?\\.${precision}"
        ;;
      hip)
        # Deliberately enable cross-statement contraction: the product fence, not an
        # incidental compiler default, must protect the decomposed operation.
        hipcc --offload-arch=gfx1100 --cuda-device-only -S -O2 -ffp-contract=fast \
          -o "$assembly" "$source_root/hip/${stem}.hip"
        fused_pattern="v_fma(c)?_${precision}"
        product_pattern="v_mul_${precision}"
        add_pattern="v_(dual_)?add_${precision}"
        ;;
      *) echo "Unsupported scalar-rounding target: $target" >&2; exit 2 ;;
    esac
    if [[ "$realization" == fused ]]; then
      grep -Eq "$fused_pattern" "$assembly"
    else
      grep -Eq "$product_pattern" "$assembly"
      grep -Eq "$add_pattern" "$assembly"
      if grep -Eq "$fused_pattern" "$assembly"; then
        echo "ERROR: decomposed scalar arithmetic contracted: $target $precision" >&2
        exit 1
      fi
    fi
    echo "Checked generated scalar realization: $target $precision $realization"
  done
done
