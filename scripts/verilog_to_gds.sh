#!/bin/bash

set -e
set -o pipefail

# Check if docker is available or if the needed binaries are available
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <input.v>" >&2
  echo "  This triggers openroad synthesis in the base directory of the input file." >&2
  exit 1
fi

SYNTHESIS_DIR=$(pwd)/$(dirname $1)

pushd $SYNTHESIS_DIR

$DOCKER_RUN \
/bin/bash ./synthesize_Synthesis_forward_kernel.sh

popd