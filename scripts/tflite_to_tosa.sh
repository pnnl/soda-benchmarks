#!/bin/bash

set -e -o pipefail

# Check if docker is available or if the needed binaries are available
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <tflite model> <output_tosa.mlir>" >&2
  exit 1
fi

OUTPUT_DIR=$(pwd)/$(dirname $2)
mkdir -p $OUTPUT_DIR
OUTPUT_NAME=$(basename $2)
OUTPUT_PATH=$OUTPUT_DIR/$OUTPUT_NAME

# set -x

$DOCKER_RUN \
flatbuffer_translate \
  -tflite-flatbuffer-to-mlir \
  -mlir-print-local-scope \
  -emit-builtin-tflite-ops \
  -lower-tensor-list-ops \
  $1 \
  -o $OUTPUT_DIR/00_tfl.mlir

$DOCKER_RUN \
tf-opt \
  -tf-executor-to-functional-conversion \
  -tf-region-control-flow-to-functional \
  -tf-shape-inference \
  -tf-to-tosa-pipeline \
  -tfl-to-tosa-pipeline \
  -tf-tfl-to-tosa-pipeline \
  -tosa-legalize-tfl \
  -tosa-strip-quant-types \
  -tosa-tflite-verify-fully-converted \
  $OUTPUT_DIR/00_tfl.mlir \
  -o $OUTPUT_PATH

set +x
