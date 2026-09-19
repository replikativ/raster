#!/usr/bin/env bash
# Build ggml_oracle against a llama.cpp checkout's own ggml build.
#   LLAMA_CPP=../llama.cpp-new dev/ggml_oracle/build.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
llama="${LLAMA_CPP:?set LLAMA_CPP to a built llama.cpp checkout}"
llama="$(cd "$llama" && pwd)"
cc -O2 -std=c11 -Wall -o "$here/ggml_oracle" "$here/ggml_oracle.c" \
   -I "$llama/ggml/include" -L "$llama/build/bin" \
   -lggml-base -lggml-cpu -Wl,-rpath,"$llama/build/bin"
echo "built $here/ggml_oracle against llama.cpp $(git -C "$llama" rev-parse --short HEAD)"
