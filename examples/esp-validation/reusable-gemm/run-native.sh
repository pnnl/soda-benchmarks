#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
generated=$(realpath "${1:?Usage: run-native.sh generated-output fixture-directory}")
mkdir -p "${2:?Fixture directory required}"
fixtures=$(realpath "$2")
grep -q '^mechanism=reusable-gemm$' "$generated/layout.txt"
"${CLANG:-clang-19}" -O2 -ffp-contract=off -DSODAP_GEMM_TRACE \
  "$generated/05_kernel_riscv.ll" "$repo/scripts/lib/soda_testbench_main.c" \
  "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/ReusableGemmRuntime.cpp" \
  "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/esp_prof.c" \
  -I"$generated" -I"$repo/examples/soda-plugins/include" \
  -I"$repo/examples/soda-plugins/include/sodap/ExecutionEngine" \
  -o "$fixtures/native"
SODAP_GEMM_TRACE_DIR="$fixtures" "$fixtures/native" | tee "$fixtures/native.log"
grep -q 'TEST PASSED' "$fixtures/native.log"
