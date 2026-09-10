#!/usr/bin/env bash
# Hardware-free ESP lowering, mock ABI and native profiler checks.
# Run after building examples/soda-plugins; see docs/ESPValidation.md.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
plugin_root="$repo/examples/soda-plugins"
test_root="$plugin_root/test/sodap/ESP"
plugin_build=${SODAP_BUILD_DIR:-$plugin_root/build}
llvm_tools=${LLVM_TOOLS_DIR:-$(dirname -- "$(command -v mlir-opt)")}
llvm_libs=${LLVM_LIB_DIR:-$llvm_tools/../lib}
plugin=${MLIR_PLUGIN_PATH:-$plugin_build/lib/SODAPlugin.so}
mock=${ESP_RUNTIME_LIB:-$plugin_build/lib/libmlir_mockesp_runner_utils.so}
opt=("$llvm_tools/mlir-opt" "--load-pass-plugin=$plugin")
check="$llvm_tools/FileCheck"
test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/sodap-esp-tests.XXXXXX")
trap 'rm -f -- "$test_tmp/two-ir.log" "$test_tmp/test_esp_prof"; rmdir -- "$test_tmp"' EXIT

lower() {
  "${opt[@]}" "$1" \
    --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp{$2},canonicalize)"
}

for prefix in CHECK PROF IR; do
  options=vec-len=8
  if [[ $prefix != CHECK ]]; then options+=' profile=true'; fi
  if [[ $prefix == IR ]]; then options+=' marshal=ir'; fi
  lower "$test_root/batch-matmul-to-esp.mlir" "$options" |
    "$check" "$test_root/batch-matmul-to-esp.mlir" --check-prefix="$prefix"
  printf 'PASS: lowering %s\n' "$prefix"
done

sequential="$test_root/batch-matmul-to-esp-sequential.mlir"
lower "$sequential" '' | "$check" "$sequential"
lower "$sequential" 'profile=true' | "$check" "$sequential" --check-prefix=PROF
if lower "$sequential" 'marshal=ir' > "$test_tmp/two-ir.log" 2>&1; then
  printf 'FAIL: marshal=ir accepted multiple offloads\n' >&2
  exit 1
fi
"$check" "$sequential" --check-prefix=IR-ERROR < "$test_tmp/two-ir.log"
printf 'PASS: sequential runtime offloads; IR offloads rejected\n'

for mode in runtime ir; do
  "${opt[@]}" "$test_root/batch-matmul-to-esp-invalid.mlir" \
    --split-input-file --verify-diagnostics \
    --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp{marshal=$mode})" \
    > /dev/null
  printf 'PASS: invalid operands rejected (%s)\n' "$mode"
done
"${opt[@]}" "$test_root/batch-matmul-to-esp-ir-invalid.mlir" \
  --split-input-file --verify-diagnostics \
  --pass-pipeline='builtin.module(sodap-linalg-batch-matmul-to-esp{marshal=ir})' \
  > /dev/null
printf 'PASS: IR dynamic shapes and nested offloads rejected\n'
lower "$test_root/batch-matmul-to-esp-ir-invalid.mlir" '' |
  "$check" "$test_root/batch-matmul-to-esp-ir-invalid.mlir" --check-prefix=RUNTIME
printf 'PASS: runtime dynamic shapes and nested sequential offloads\n'

runner="$test_root/run-batch-matmul-esp.mlir"
lower "$runner" '' |
  "$llvm_tools/mlir-opt" -convert-linalg-to-loops -convert-scf-to-cf \
    --canonicalize --cse --finalize-memref-to-llvm --convert-math-to-llvm \
    --convert-math-to-libm -arith-expand -memref-expand --convert-arith-to-llvm \
    --convert-func-to-llvm --reconcile-unrealized-casts |
  "$llvm_tools/mlir-cpu-runner" -e main -entry-point-result=void \
    -shared-libs="$mock" \
    -shared-libs="$llvm_libs/libmlir_runner_utils.so" \
    -shared-libs="$llvm_libs/libmlir_c_runner_utils.so" |
  "$check" "$runner"
printf 'PASS: batch=1 mock runner ABI and layout arguments\n'

"${CC:-cc}" -O2 -Wall -Wextra \
  -I "$plugin_root/include/sodap/ExecutionEngine" \
  "$test_root/test_esp_prof.c" \
  "$plugin_root/lib/sodap/ExecutionEngine/esp_prof.c" \
  -o "$test_tmp/test_esp_prof"
"$test_tmp/test_esp_prof"
printf 'PASS: all targeted ESP checks\n'
