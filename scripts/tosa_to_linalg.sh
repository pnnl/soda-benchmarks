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

### NOTE: Max's passes ###
$DOCKER_RUN \
mlir-opt \
  --tosa-to-arith="include-apply-rescale=true" \
  --canonicalize \
  -convert-tensor-to-linalg \
  -empty-tensor-to-alloc-tensor \
  -eliminate-empty-tensors \
  -one-shot-bufferize="function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries allow-return-allocs-from-loops unknown-type-conversion=identity-layout-map" \
  -func-bufferize \
  -buffer-deallocation-simplification \
  -bufferization-lower-deallocations \
  --buffer-results-to-out-params \
  --canonicalize -cse \
  $OUTPUT_DIR/02_linalg_on_tensors.mlir \
  -o $OUTPUT_PATH

### Original passes ### 

## $DOCKER_RUN \
## mlir-opt \
##   --canonicalize \
##   -tosa-to-arith="include-apply-rescale=true" \
##   -convert-tensor-to-linalg \
##   -empty-tensor-to-alloc-tensor \
##   -eliminate-empty-tensors \
##   -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops" \
##   -func-bufferize \ <------- potentially unneccessary
##   -finalizing-bufferize -buffer-deallocation \ <------- passes that break the flow
##   -buffer-deallocation-simplification \
##   -bufferization-lower-deallocations \
##   --buffer-results-to-out-params \
##   --canonicalize -cse \
##   $OUTPUT_DIR/02_linalg_on_tensors.mlir \
##   -o $OUTPUT_PATH

# $DOCKER_RUN \
# soda-opt \
#   -forward-memref-allocations -forward-linalg-fill -forward-memref-copy -forward-memref-allocations \
#   -canonicalize -cse \
#   $2 \
#   -o $2-fwd

set +x