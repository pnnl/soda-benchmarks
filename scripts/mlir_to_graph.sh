#!/bin/bash

set -e -o pipefail

# Check if docker is available or if the needed binaries are available
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input.mlir> <output.dot>" >&2
  exit 1
fi

OUTPUT_DIR=$(pwd)/$(dirname $2)
mkdir -p $OUTPUT_DIR
OUTPUT_NAME=$(basename $2)
OUTPUT_PATH=$OUTPUT_DIR/$OUTPUT_NAME

# set -x

## Options for view-op-graph:
#--view-op-graph                                        -   Print Graphviz visualization of an operation
#  --max-label-len=<uint>                               - Limit attribute/type length to number of chars
#  --print-attrs                                        - Print attributes of operations
#  --print-control-flow-edges                           - Print control flow edges
#  --print-data-flow-edges                              - Print data flow edges
#  --print-result-types                                 - Print result types of operations
$DOCKER_RUN \
mlir-opt \
  $1 \
  --view-op-graph="max-label-len=15" \
  -o /tmp/tmp.mlir 2> $2

set +x