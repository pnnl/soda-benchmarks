#!/bin/bash

set -e -o pipefail

# Check if docker is available or if the needed binaries are available
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_linalg.mlir> <output_llvm.mlir>" >&2
  exit 1
fi

OUTPUT_DIR=$(pwd)/$(dirname $2)
mkdir -p $OUTPUT_DIR
OUTPUT_NAME=$(basename $2)
OUTPUT_PATH=$OUTPUT_DIR/$OUTPUT_NAME

# set -x

$DOCKER_RUN \
mlir-opt \
  -convert-linalg-to-affine-loops \
  -expand-strided-metadata \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-complex-to-standard \
  -convert-vector-to-llvm \
  --convert-math-to-llvm \
  --convert-math-to-libm \
  -arith-expand \
  -memref-expand \
  -convert-to-llvm="filter-dialects=memref" \
  -finalize-memref-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-complex-to-llvm \
  -convert-func-to-llvm='use-bare-ptr-memref-call-conv=1' \
  --test-lower-to-llvm \
  -convert-cf-to-llvm \
  -reconcile-unrealized-casts \
  -symbol-dce \
  $1 \
  -o $2

# $DOCKER_RUN \
# soda-opt \
#   --lower-all-to-llvm \
#   $1 \
#   -o $2
  
set +x