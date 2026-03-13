#!/bin/bash

set -e -o pipefail

# Check if docker is available or if the needed binaries are available
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <input_linalg.mlir> <output_llvm.mlir> <transform_sched.mlir" >&2
  exit 1
fi

OUTPUT_DIR=$(pwd)/$(dirname $2)
mkdir -p $OUTPUT_DIR
OUTPUT_NAME=$(basename $2)
OUTPUT_PATH=$OUTPUT_DIR/$OUTPUT_NAME
TRANSFORM_SCHED=$(pwd)/$(dirname $3)/$(basename $3)

# set -x

# If a plugin is available, append the needed arguments to load it
MLIR_PLUGIN_PATH="${MLIR_PLUGIN_PATH:-/workspaces/soda/soda-benchmarks/examples/soda-plugins/build/lib/SODAPlugin.so}"
MLIR_PLUGIN_ARGS=""
if [ -f "$MLIR_PLUGIN_PATH" ]; then
  MLIR_PLUGIN_ARGS="--load-pass-plugin=$MLIR_PLUGIN_PATH --load-dialect-plugin=$MLIR_PLUGIN_PATH"
fi

$DOCKER_RUN \
soda-opt \
  $MLIR_PLUGIN_ARGS \
  --convert-all-to-soda \
  -soda-outline-bambu-code \
  -soda-extract-arguments-to-c-testbench=using-bare-ptr \
  -soda-generate-bambu-accelcode=no-aa \
	--transform-preload-library=transform-library-paths="$TRANSFORM_SCHED" \
	--transform-interpreter \
	--soda-transform-erase-schedule \
  --lower-all-to-llvm="use-bare-ptr-memref-call-conv" \
  --convert-func-to-llvm \
  -mlir-print-ir-after-all \
  $1 \
  -o $2 \
  2>&1 | cat > $2.steps.mlir

  
set +x
