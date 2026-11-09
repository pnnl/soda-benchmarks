#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Untiled operator isolation and independent reusable hardware generation.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 input_tosa.mlir [output-directory]" >&2
  exit 1
fi
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
source "$repo/scripts/check_docker.sh"
input=$(realpath "$1")
out=${2:-$(dirname "$input")}
mkdir -p "$out"
cd "$out"
plugin=${MLIR_PLUGIN_PATH:-$repo/examples/soda-plugins/build/lib/SODAPlugin.so}
plugins=(--load-dialect-plugin="$plugin" --load-pass-plugin="$plugin")
profile=${LCS_PROFILE:-false}
[[ "$profile" == true || "$profile" == false ]]
[[ ${LCS_DMA_WORD_STRIDE:-2} == 2 ]] || {
  echo 'Reusable GEMM ABI v1 requires LCS_DMA_WORD_STRIDE=2' >&2; exit 1;
}
tm=${LCS_GEMM_TILE_M:-4}
tn=${LCS_GEMM_TILE_N:-4}
tk=${LCS_GEMM_TILE_K:-4}
schedule=${LCS_GEMM_SCHEDULE:-sequential}
case "$schedule" in sequential|lcs) ;; *) echo 'LCS_GEMM_SCHEDULE must be sequential or lcs' >&2; exit 1 ;; esac
for tile in "$tm" "$tn" "$tk"; do
  [[ "$tile" =~ ^[1-9][0-9]*$ ]] && ((tile <= 128)) || {
    echo 'GEMM tile extents must be in [1,128]' >&2; exit 1;
  }
done
printf 'mechanism=reusable-gemm\nabi=1\ndma_word_stride=2\nprofile=%s\n' "$profile" > layout.txt
printf 'tile_m=%s\ntile_n=%s\ntile_k=%s\nschedule=%s\n' "$tm" "$tn" "$tk" "$schedule" > schedule.txt

$DOCKER_RUN mlir-opt "$input" --pass-pipeline='builtin.module(
  func.func(tosa-to-arith,tosa-to-tensor,tosa-to-linalg-named,tosa-to-linalg),
  linalg-fold-unit-extent-dims,
  one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},
  canonicalize,buffer-results-to-out-params{hoist-static-allocs=true},fold-memref-alias-ops)' \
  -o bufferized.mlir
$DOCKER_RUN soda-opt bufferized.mlir --convert-all-to-soda -soda-outline-bambu-code -o outlined.mlir
python3 "$repo/scripts/lcs/argument_order.py" outlined.mlir > argument_order.txt
$DOCKER_RUN soda-opt outlined.mlir -soda-generate-bambu-accelcode=no-aa -canonicalize -o kernel.mlir

# No scheduling options enter this pass. Inspect host.mlir for each invocation
# and its explicit binding to instance zero.
$DOCKER_RUN mlir-opt kernel.mlir "${plugins[@]}" \
  --pass-pipeline='builtin.module(sodap-reusable-gemm-to-esp)' -o host.mlir
cp kernel.mlir cpu.mlir
if [[ "$profile" == true ]]; then
  for ir in host cpu; do
    $DOCKER_RUN mlir-opt "$ir.mlir" "${plugins[@]}" \
      --pass-pipeline='builtin.module(sodap-profile-kernel)' -o "$ir.mlir"
  done
fi
for ir in host cpu; do
  $DOCKER_RUN soda-opt "$ir.mlir" \
    --lower-all-to-llvm=use-bare-ptr-memref-call-conv \
    --convert-func-to-llvm --reconcile-unrealized-casts -o "$ir-llvm.mlir"
done
$DOCKER_RUN mlir-translate --mlir-to-llvmir host-llvm.mlir -o 05_kernel_riscv.ll
$DOCKER_RUN mlir-translate --mlir-to-llvmir cpu-llvm.mlir -o 05_cpu_baseline.ll

# Hardware depends only on the common ABI and the selected implementation.
printf 'module {}\n' > empty.mlir
$DOCKER_RUN mlir-opt empty.mlir "${plugins[@]}" \
  --pass-pipeline="builtin.module(sodap-generate-reusable-gemm{tile-m=$tm tile-n=$tn tile-k=$tk schedule=$schedule},canonicalize,cse)" \
  -o accelerator.mlir
$DOCKER_RUN mlir-opt accelerator.mlir "${plugins[@]}" --pass-pipeline='builtin.module(
  func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=1048576}),
  sodap-emit-bambu-architecture{top-func=forward_kernel file=architecture.xml},
  sodap-convert-dataflow-to-llvm,lower-affine,convert-scf-to-cf,
  convert-math-to-llvm,expand-strided-metadata,lower-affine,
  finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=true},
  convert-index-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)' -o accelerator-llvm.mlir
$DOCKER_RUN mlir-translate --mlir-to-llvmir accelerator-llvm.mlir \
  | $DOCKER_RUN llvm-link -S - ac_channel_specializations.ll \
  | $DOCKER_RUN opt -passes=always-inline -S -o 05_hw_acc.ll
cp "$repo/examples/soda-plugins/include/sodap/ExecutionEngine/ReusableGemmABI.h" interface.h
sha256sum interface.h 05_hw_acc.ll architecture.xml > hardware-inputs.sha256
echo "Reusable GEMM host and hardware: $PWD"
