#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
example=$(cd -- "$(dirname -- "$0")" && pwd)
lcs=$(realpath "${1:?Usage: run-adapter.sh output/esp/<flow> [build-directory]}")
build=${2:-$(mktemp -d /tmp/esp-lcs-adapter.XXXXXX)}
mkdir -p "$build"
build=$(realpath "$build")
timeout_cycles=${LCS_TIMEOUT_CYCLES:-1000000}
[[ "$timeout_cycles" =~ ^[1-9][0-9]*$ ]] && ((timeout_cycles < 100000000)) || {
  echo 'LCS_TIMEOUT_CYCLES must be between 1 and 99999999' >&2; exit 1;
}
python3 "$example/make-vectors.py" "$lcs" "$build"
shopt -s nullglob
memories=("$lcs/bambu_syn/"*.mem)
if ((${#memories[@]})); then cp "${memories[@]}" "$build/"; fi
iverilog -g2012 -I"$build" -s lcs_dma64_tb \
  -Plcs_dma64_tb.SERIAL_DMA="${LCS_SERIAL_DMA:-1}" \
  -Plcs_dma64_tb.TIMEOUT_CYCLES="$timeout_cycles" \
  -Plcs_dma64_tb.WRITE_BUFFER_BEATS="${LCS_WRITE_BUFFER_BEATS:-16}" -o "$build/test.vvp" \
  "$lcs/bambu_syn/forward_kernel.v" "$lcs/bambu_syn/panda_libtech.v" \
  "$example/../generated-gemm/sqr_bambu_basic_dma64.sv" "$example/dma64_tb.sv"
cd "$build"
# A long initial stall alone can conceal an early-write deadlock. Exercise
# both immediate commands and delayed commands unless a single case is chosen.
stalls=(0 50000)
if [[ -n ${LCS_STALL_CYCLES+x} ]]; then stalls=("$LCS_STALL_CYCLES"); fi
: > simulation.log
for stall in "${stalls[@]}"; do
  log="simulation-stall${stall}.log"
  vvp ./test.vvp +gemm_stall="$stall" | tee "$log"
  cat "$log" >> simulation.log
  grep -q 'LCS_DMA64 PASS errors=0' "$log"
done
