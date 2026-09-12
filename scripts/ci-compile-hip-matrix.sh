#!/usr/bin/env bash
# Compile a Raster-generated candidate, not a prewritten rocWMMA GEMM. No GPU is required.
set -euo pipefail

source_file=${1:-gpu-compile-gates/hip-matrix/mfma-uniform-epilogue.hip}
test -s "$source_file"
matrix_workdir=$(mktemp -d "${TMPDIR:-/tmp}/raster-hip-matrix.XXXXXX")
trap 'rm -r -- "$matrix_workdir"' EXIT
revision=b5a884dc764d2cb3de480294466895ee7e5efd6a
archive="$matrix_workdir/rocwmma.tar.gz"
if [[ -n "${RASTER_ROCWMMA_ARCHIVE:-}" ]]; then
  cp "$RASTER_ROCWMMA_ARCHIVE" "$archive"
else
  curl --fail --location --retry 3 \
    "https://codeload.github.com/ROCm/rocWMMA/tar.gz/$revision" -o "$archive"
fi
echo "64fd8396ed44b32a56e6271d057b9b639fe28d7d5aa61bdf014e31486bf4ca7b  $archive" | sha256sum --check
tar -xzf "$archive" -C "$matrix_workdir"
include="$matrix_workdir/rocWMMA-$revision/library/include"
binary="$matrix_workdir/candidate.hsaco"
hipcc -D__HIP_PLATFORM_AMD__ -std=c++17 --offload-arch=gfx90a --genco \
  -I "$include" "$source_file" -o "$binary"

# hipcc versions produce either an ELF code object or a Clang offload bundle.
objdump=${HIP_OBJDUMP:-/opt/rocm/llvm/bin/llvm-objdump}
bundler=${HIP_BUNDLER:-/opt/rocm/llvm/bin/clang-offload-bundler}
assembly="$matrix_workdir/candidate.s"
if ! "$objdump" -d "$binary" > "$assembly" 2> "$matrix_workdir/objdump.log"; then
  "$bundler" --unbundle --type=o --input="$binary" \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx90a --output="$matrix_workdir/candidate.elf"
  "$objdump" -d "$matrix_workdir/candidate.elf" > "$assembly"
fi
grep 'v_mfma_f32_16x16x16f16' "$assembly"

if hipcc -D__HIP_PLATFORM_AMD__ -std=c++17 --offload-arch=gfx1100 --genco \
  -I "$include" "$source_file" -o "$binary" > "$matrix_workdir/wrong-target.log" 2>&1; then
  echo 'ERROR: the MFMA/wave64 candidate compiled for RDNA3/wave32' >&2
  exit 1
fi
grep 'Raster MFMA candidate requires gfx90a' "$matrix_workdir/wrong-target.log"
echo 'Generated MFMA candidate compiled; matrix instruction confirmed; incompatible architecture rejected.'
