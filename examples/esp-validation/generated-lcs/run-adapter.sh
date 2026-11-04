#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
example=$(cd -- "$(dirname -- "$0")" && pwd)
lcs=$(realpath "${1:?Usage: run-adapter.sh output/lcs [build-directory]}")
build=${2:-$(mktemp -d /tmp/esp-lcs-adapter.XXXXXX)}
mkdir -p "$build"
build=$(realpath "$build")
python3 "$example/make-vectors.py" "$lcs" "$build"
shopt -s nullglob
memories=("$lcs/bambu/"*.mem)
if ((${#memories[@]})); then cp "${memories[@]}" "$build/"; fi
iverilog -g2012 -I"$build" -s lcs_dma64_tb -o "$build/test.vvp" \
  "$lcs/bambu/forward_kernel.v" "$lcs/bambu/panda_libtech.v" \
  "$example/../generated-gemm/sqr_bambu_basic_dma64.sv" "$example/dma64_tb.sv"
cd "$build"
vvp ./test.vvp +gemm_stall=50000 | tee simulation.log
grep -q 'LCS_DMA64 PASS errors=0' simulation.log
