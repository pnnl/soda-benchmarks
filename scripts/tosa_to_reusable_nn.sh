#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
input=$(realpath "${1:?Usage: tosa_to_reusable_nn.sh input-tosa.mlir output-directory}")
mkdir -p "${2:?output directory required}"
out=$(realpath "$2")
python3 "$repo/scripts/reusable_nn/lower.py" "$input" "$out"
soda-opt "$out/host.mlir" --lower-all-to-llvm=use-bare-ptr-memref-call-conv \
  --convert-func-to-llvm --reconcile-unrealized-casts -o "$out/host-llvm.mlir"
mlir-translate --mlir-to-llvmir "$out/host-llvm.mlir" -o "$out/host.ll"
"${CLANG:-clang-19}" -O3 -fPIC -shared "$out/host.ll" -o "$out/host-only.so"
"${CLANG:-clang-19}" -O3 -ffp-contract=off -fPIC -shared "$out/host.ll" \
  "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp" \
  -I"$repo/examples/soda-plugins/include" -o "$out/native.so"
"${RISCV_CLANG:-clang}" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d \
  -mcmodel=medany -O2 -c "$out/host.ll" -o "$out/host-riscv.o"
sha256sum "$out/host.ll" "$out/host-riscv.o" "$out/parameters.bin" > "$out/host-inputs.sha256"
