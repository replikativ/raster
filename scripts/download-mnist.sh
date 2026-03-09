#!/usr/bin/env bash
# Download MNIST dataset for raster benchmarks and tests.
# Usage: bash script/download-mnist.sh

set -euo pipefail

DEST="data/mnist"
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

FILES=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

mkdir -p "$DEST"

for f in "${FILES[@]}"; do
  if [ -f "$DEST/$f" ]; then
    echo "Already exists: $DEST/$f"
  else
    echo "Downloading $f ..."
    curl -sL "$BASE_URL/$f" -o "$DEST/$f"
    echo "  saved to $DEST/$f"
  fi
done

echo "MNIST data ready in $DEST/"
