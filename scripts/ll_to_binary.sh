#!/bin/bash
# Link a soda-generated LLVM IR kernel into a native executable (cpu backend).
#
# Usage: ll_to_binary.sh <input.ll> <output_binary>
#
# The kernel is compiled as-is: the IR mlir-translate emits carries no target
# triple or datalayout, so clang applies the host's. It is linked against the
# generic driver in scripts/lib/ (which includes the generated testdata.h) and
# against an ESP runtime implementation, so an ESP-lowered kernel resolves its
# esp_* calls. The default runtime is the mock, which prints each call and
# returns -- enough to check that the pass fired, the ABI matches and the
# symbols link, but it never computes, so the driver will report a mismatch.

set -e -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input.ll> <output_binary>" >&2
  exit 1
fi

INPUT_LL="$1"
OUTPUT="$2"

if [ ! -f "$INPUT_LL" ]; then
  echo "Error: $INPUT_LL not found" >&2
  exit 1
fi

CC="${CC:-clang}"
CPU_MAIN="${CPU_MAIN:-$SCRIPT_DIR/lib/soda_testbench_main.c}"
# testdata.h sits beside the .ll, in the experiment's output directory.
TESTDATA_DIR="${TESTDATA_DIR:-$(cd "$(dirname "$INPUT_LL")" && pwd)}"
SODAP_LIB_DIR="${SODAP_LIB_DIR:-/workspaces/soda/soda-benchmarks/examples/soda-plugins/build/lib}"
ESP_RUNTIME_LIB="${ESP_RUNTIME_LIB:-$SODAP_LIB_DIR/libmlir_mockesp_runner_utils.so}"

if [ ! -f "$TESTDATA_DIR/testdata.h" ]; then
  echo "Error: $TESTDATA_DIR/testdata.h not found." >&2
  echo "  Generate it with: python torchscript.py --emit-testdata output/testdata.h" >&2
  echo "  (it needs a benchmark, so the experiment must have been scaffolded" >&2
  echo "   with --benchmark_name)" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
OUTPUT_DIR=$(cd "$(dirname "$OUTPUT")" && pwd)

KERNEL_OBJ="$OUTPUT_DIR/kernel.o"
MAIN_OBJ="$OUTPUT_DIR/main.o"

echo "[ll_to_binary] kernel  : $INPUT_LL"
echo "[ll_to_binary] driver  : $CPU_MAIN"
echo "[ll_to_binary] testdata: $TESTDATA_DIR/testdata.h"
echo "[ll_to_binary] runtime : $ESP_RUNTIME_LIB"

$DOCKER_RUN $CC -O2 -c "$INPUT_LL" -o "$KERNEL_OBJ"
$DOCKER_RUN $CC -O2 -I"$TESTDATA_DIR" -c "$CPU_MAIN" -o "$MAIN_OBJ"

LINK_ARGS=("$MAIN_OBJ" "$KERNEL_OBJ")
if [ -f "$ESP_RUNTIME_LIB" ]; then
  LINK_ARGS+=("$ESP_RUNTIME_LIB" "-Wl,-rpath,$(cd "$(dirname "$ESP_RUNTIME_LIB")" && pwd)")
else
  # A kernel with no ESP lowering has no esp_* references, so a missing runtime
  # is only a problem if the link actually needs it -- let ld say so.
  echo "[ll_to_binary] note: $ESP_RUNTIME_LIB not found, linking without it"
fi

$DOCKER_RUN $CC "${LINK_ARGS[@]}" -lm -o "$OUTPUT"
echo "[ll_to_binary] wrote   : $OUTPUT"
