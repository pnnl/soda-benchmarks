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

# $DOCKER_RUN \
# soda-opt \
# --convert-all-to-soda \
#   -soda-outline-bambu-code \
#   -soda-extract-arguments-to-c-testbench=using-bare-ptr \
#   -soda-generate-bambu-accelcode=no-aa \
#   -lower-all-to-llvm=use-bare-ptr-memref-call-conv \
#   $1 \
#   -o $2

# $DOCKER_RUN \
# soda-opt \
#   -convert-operation-to-soda="anchor-op=linalg.batch_matmul" \
#   -soda-outline-bambu-code \
#   -soda-extract-arguments-to-c-testbench=using-bare-ptr \
#   -soda-generate-bambu-accelcode=no-aa \
#   -lower-all-to-llvm=use-bare-ptr-memref-call-conv \
#   -mlir-print-ir-after-all \
#   $1 \
#   -o $2 \
#   2>&1 | cat > $2.steps.mlir

$DOCKER_RUN \
soda-opt \
  --convert-all-to-soda \
  -soda-outline-bambu-code \
  -soda-extract-arguments-to-c-testbench=using-bare-ptr \
  -soda-generate-bambu-accelcode=no-aa \
  -lower-all-to-llvm=use-bare-ptr-memref-call-conv \
  -mlir-print-ir-after-all \
  $1 \
  -o $2 \
  2>&1 | cat > $2.steps.mlir

# $DOCKER_RUN \
# soda-opt \
#   --lower-all-to-llvm \
#   $1 \
#   -o $2
  
set +x