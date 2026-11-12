#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
hardware=$(realpath "${1:?Usage: build-cosim.sh hardware-directory new-build-directory}")
out=${2:?build directory required}
mkdir -p "$out"
out=$(realpath "$out")
shopt -s nullglob
memories=("$hardware/combined/"*.mem)
if ((${#memories[@]})); then cp "${memories[@]}" "$out/"; fi
cd "$hardware/combined"
verilator --cc --top-module sqr_bambu_basic_dma64 --prefix Voperator \
  --Mdir "$out" -Wno-fatal -DSODAP_LCS_GENERIC -GTIMEOUT_CYCLES=2000000000 \
  -CFLAGS '-O3 -fPIC' --output-split 20000 --output-split-cfuncs 5000 \
  +incdir+"$hardware/combined" operator_engines.v forward_kernel.sv \
  "$here/../generated-gemm/sqr_bambu_basic_dma64.sv" > "$out/verilator.log" 2>&1
make -C "$out" -f Voperator.mk -j"${NN_BUILD_JOBS:-8}" > "$out/build.log" 2>&1
# Keep the simulator independent of a particular compiled host/model.
verilator_include=${VERILATOR_INCLUDE:-/usr/share/verilator/include}
c++ -std=c++17 -O3 -fPIC -shared -pthread \
  -DSODAP_OP_EXTERNAL_EXECUTOR -DSODAP_OP_VERIFY_RTL -ffp-contract=off \
  -I"$repo/examples/soda-plugins/include" -I"$out" -I"$verilator_include" -I"$verilator_include/vltstd" \
  "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp" \
  "$here/rtl_executor.cpp" "$out/Voperator__ALL.a" \
  "$verilator_include/verilated.cpp" \
  -o "$out/rtl-runtime.so" > "$out/link.log" 2>&1
echo "RTL runtime: $out/rtl-runtime.so"
