#!/bin/bash

set -e -o pipefail

# Check if docker is available or if the needed binaries are available
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_tosa.mlir> <output_linalg.mlir>" >&2
  exit 1
fi

OUTPUT_DIR=$(pwd)/$(dirname $2)
mkdir -p $OUTPUT_DIR
OUTPUT_NAME=$(basename $2)
OUTPUT_PATH=$OUTPUT_DIR/$OUTPUT_NAME

# set -x

$DOCKER_RUN \
mlir-opt \
  -pass-pipeline="builtin.module(func.func(tosa-to-arith, tosa-to-tensor, tosa-to-linalg-named, tosa-to-linalg))" \
  $1 \
  -o $OUTPUT_DIR/02_linalg_on_tensors.mlir
  
$DOCKER_RUN \
mlir-opt \
  -tosa-to-arith="include-apply-rescale=true" \
  -convert-tensor-to-linalg \
  -eliminate-empty-tensors \
  -empty-tensor-to-alloc-tensor \
  -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops" \
  -func-bufferize \
  -finalizing-bufferize -buffer-deallocation \
  -buffer-deallocation-simplification \
  -bufferization-lower-deallocations \
  --buffer-results-to-out-params \
  --canonicalize -cse \
  $OUTPUT_DIR/02_linalg_on_tensors.mlir \
  -o $OUTPUT_PATH

# $DOCKER_RUN \
# soda-opt \
#   -forward-memref-allocations -forward-linalg-fill -forward-memref-copy -forward-memref-allocations \
#   -canonicalize -cse \
#   $2 \
#   -o $2-fwd

set +x