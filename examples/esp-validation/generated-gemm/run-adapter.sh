#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
example=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
generated=$(realpath "${1:?Usage: run-adapter.sh extracted-gemm-directory [build-directory]}")
build=${2:-$(mktemp -d /tmp/esp-gemm-adapter.XXXXXX)}
mkdir -p "$build"
build=$(realpath "$build")
python3 "$example/make-vectors.py" "$build"
cp "$generated/bambu/array_429488.mem" "$build/"
iverilog -g2012 -s dma64_tb -o "$build/test.vvp" \
  "$generated/bambu/forward_kernel.v" "$generated/bambu/panda_libtech.v" \
  "$example/sqr_bambu_basic_dma64.sv" "$example/dma64_tb.sv"
cd "$build"
vvp ./test.vvp +gemm_stall=12000 +data_dir="$build/Inputs" | tee simulation.log
grep -q 'GEMM_DMA64 PASS errors=0' simulation.log
