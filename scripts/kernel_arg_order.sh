#!/bin/bash
# Print the outlined kernel's argument order, as indices into @forward's parameters.
#
# Usage: kernel_arg_order.sh <input_linalg.mlir>
# Output: one line, e.g. "3 4 0 1 2 5"
#
# -soda-outline-bambu-code hoists the kernel into its own function and orders its
# parameters by first use inside the outlined region, not by @forward's signature.
# For gemm that turns (alpha, beta, C, A, B, out) into (A, B, alpha, beta, C, out).
# Anything that has to call forward_kernel -- the cpu and esp backends' driver --
# therefore needs the permutation, and the soda.launch_func the outliner leaves
# behind states it exactly. Every soda_to_llvm_*.sh script starts with the same
# two passes used here, so the order is the same for all three flows.

set -e -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <input_linalg.mlir>" >&2
  exit 1
fi

SODA_OPT="${SODA_OPT:-/workspaces/soda/builds/soda-opt/build/bin/soda-opt}"
command -v "$SODA_OPT" >/dev/null 2>&1 || SODA_OPT=soda-opt

LAUNCH=$($DOCKER_RUN $SODA_OPT --convert-all-to-soda -soda-outline-bambu-code \
  "$1" -o /dev/stdout 2>/dev/null | grep -m1 'soda.launch_func' || true)

if [ -z "$LAUNCH" ]; then
  echo "Error: no soda.launch_func in the outlined $1" >&2
  exit 1
fi

# args(%arg3 : memref<...>, %arg4 : memref<...>, ...) -> "3 4 ..."
echo "$LAUNCH" | sed 's/.*args(//' | grep -o '%arg[0-9]\+' | sed 's/%arg//' | tr '\n' ' ' | sed 's/ $//'
echo
