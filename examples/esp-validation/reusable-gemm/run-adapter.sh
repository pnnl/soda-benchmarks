#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
generated=$(realpath "${1:?Usage: run-adapter.sh generated-output fixture-directory}")
fixtures=$(realpath "${2:?Native SODAP_GEMM_TRACE_DIR required}")
grep -q '^mechanism=reusable-gemm$' "$generated/layout.txt"
test -s "$fixtures/parameters.svh"
shopt -s nullglob
memories=("$generated/bambu_syn/"*.mem)
if ((${#memories[@]})); then cp "${memories[@]}" "$fixtures/"; fi
tol=$(sed -n 's/^#define TD_TOL *//p' "$generated/testdata.h")
iverilog -g2012 -I"$fixtures" -s reusable_gemm_tb \
  -Preusable_gemm_tb.TOL="$tol" \
  -Preusable_gemm_tb.TIMEOUT_CYCLES="${LCS_TIMEOUT_CYCLES:-2000000}" \
  -o "$fixtures/test.vvp" "$generated/bambu_syn/forward_kernel.v" \
  "$generated/bambu_syn/panda_libtech.v" \
  "$here/../generated-gemm/sqr_bambu_basic_dma64.sv" "$here/dma64_tb.sv"
cd "$fixtures"
for stall in 0 50000; do
  vvp ./test.vvp +gemm_stall="$stall" | tee "simulation-stall$stall.log"
  grep -q 'REUSABLE_GEMM_RTL PASS' "simulation-stall$stall.log"
done
