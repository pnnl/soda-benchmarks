#!/bin/bash

set -e
set -o pipefail

# Check if docker is available or if the needed binaries are available
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <input.v> [top-fname]" >&2
  echo "  This triggers openroad synthesis in the base directory of the input file." >&2
  echo "  top-fname defaults to forward_kernel." >&2
  exit 1
fi

TOP_FNAME="${2:-forward_kernel}"

SYNTHESIS_DIR=$(pwd)/$(dirname $1)

pushd $SYNTHESIS_DIR

$DOCKER_RUN \
/bin/bash ./synthesize_Synthesis_${TOP_FNAME}.sh

popd