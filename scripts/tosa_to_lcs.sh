#!/bin/bash
# Every pass from TOSA to the LLVM IR of a load/compute/store accelerator and of the
# RISC-V program that drives it. The IR after each pass, whole modules, all goes to
# one file: <outdir>/ir_after_each_pass.mlir. Run it with pixi, in the devcontainer.
#
# Usage: ./scripts/tosa_to_lcs.sh <input_tosa.mlir> [<outdir>]
# The output directory defaults to the input directory. Set LCS_PROFILE=true to instrument
# 05_kernel_riscv.ll and 05_cpu_baseline.ll with matching whole-kernel regions.
#
# Writes in <outdir>:
#   ir_after_each_pass.mlir         the IR after each pass
#   05_hw_acc.ll                    for Bambu, with architecture.xml
#   architecture.xml
#   05_kernel_riscv.ll              for the RISC-V core
#   05_cpu_baseline.ll              matching CPU-only baseline
#   ac_channel_specializations.ll   needed to build 05_hw_acc.ll
#   forward_kernel_testbench.c      written by soda-opt, not used here

set -eou pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <input_tosa.mlir> [<outdir>]" >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/scripts/check_docker.sh"
INPUT=$(realpath "$1")
OUT_DIR="${2:-$(dirname "$1")}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

MLIR_PLUGIN_PATH="${MLIR_PLUGIN_PATH:-$REPO_DIR/examples/soda-plugins/build/lib/SODAPlugin.so}"
PLUGIN="--load-dialect-plugin=$MLIR_PLUGIN_PATH --load-pass-plugin=$MLIR_PLUGIN_PATH"
TRANSFORM="${LCS_TRANSFORM:-$REPO_DIR/scripts/lcs/transform.mlir}"
TOP=forward_kernel
PROFILE=${LCS_PROFILE:-false}
WORD_STRIDE=${LCS_DMA_WORD_STRIDE:-1}
case "$WORD_STRIDE" in 1|2) ;; *) echo "LCS_DMA_WORD_STRIDE must be 1 or 2" >&2; exit 1 ;; esac
printf 'dma_word_stride=%s\nprofile=%s\n' "$WORD_STRIDE" "$PROFILE" > layout.txt
case "$PROFILE" in true|false) ;; *) echo "LCS_PROFILE must be true or false" >&2; exit 1 ;; esac
DUMP=ir_after_each_pass.mlir
: > "$DUMP"

FRONTEND_DUMP=ir_after_each_pass_frontend.mlir
: > "$FRONTEND_DUMP"

HOST_DUMP=ir_after_each_pass_host.mlir
: > "$HOST_DUMP"

ACC_DUMP=ir_after_each_pass_acc.mlir
: > "$ACC_DUMP"


# Run a tool on stdin, append its IR after each pass to the dump. The dump also gets
# any error, so show its end when the tool fails.
step() {
  local where=$1
  shift
  $DOCKER_RUN "$@" --mlir-disable-threading --mlir-print-ir-after-all \
    --mlir-print-ir-module-scope 2> >(tee -a "$DUMP" "$where" > /dev/null) \
    || { tail -20 "$where" >&2; return 1; }
}

section() {
  local where=$1
  shift
  printf '\n// ========== %s ==========\n\n' "$1" | tee -a "$DUMP" "$where" > /dev/null
}

section "$FRONTEND_DUMP" "Front end and tiling, mlir-opt"
IR=$(step "$FRONTEND_DUMP" mlir-opt "$INPUT" $PLUGIN --pass-pipeline="builtin.module(
    func.func(
      tosa-to-arith,
      tosa-to-tensor,
      tosa-to-linalg-named,
      tosa-to-linalg,
      sodap-linalg-rank-reduce-contractions
    ),
    linalg-fold-unit-extent-dims,
    linalg-fuse-elementwise-ops,
    canonicalize,
    transform-preload-library{transform-library-paths=$TRANSFORM},
    transform-interpreter{entry-point=tile},
    one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},
    canonicalize,
    cse,
    canonicalize,
    buffer-results-to-out-params{hoist-static-allocs=true},
    loop-invariant-code-motion,
    fold-memref-alias-ops
  )")

section "$FRONTEND_DUMP" "Outlining forward_kernel, soda-opt"
OUTLINED=$(step "$FRONTEND_DUMP" soda-opt \
  --convert-all-to-soda -soda-outline-bambu-code <<< "$IR")
printf '%s\n' "$OUTLINED" > outlined.mlir
python3 "$REPO_DIR/scripts/lcs/argument_order.py" outlined.mlir > argument_order.txt
KERNEL=$(step "$FRONTEND_DUMP" soda-opt \
  -soda-extract-arguments-to-c-testbench=using-bare-ptr \
  -soda-generate-bambu-accelcode=no-aa <<< "$OUTLINED")
printf '%s\n' "$KERNEL" > kernel.mlir

section "$ACC_DUMP" "Accelerator: to dataflow, mlir-opt"
IR=$(step "$ACC_DUMP" mlir-opt $PLUGIN --pass-pipeline="builtin.module(
    canonicalize,
    transform-preload-library{transform-library-paths=$TRANSFORM},
    transform-interpreter{entry-point=promote},
    func.func(buffer-loop-hoisting),
    transform-interpreter{entry-point=multibuffer},
    transform-interpreter{entry-point=split},
    sodap-dataflow-copy-to-dma{top-func=$TOP dma-word-stride=$WORD_STRIDE},
    func.func(
      sodap-dataflow-tasks-to-nodes
    ),
    sodap-dataflow-nodes-to-func,
    func.func(
      convert-linalg-to-loops
    ),
    sodap-dataflow-infer-top-interface{top-func=$TOP},
    sodap-emit-bambu-architecture{top-func=$TOP file=architecture.xml},
    sodap-convert-dataflow-to-llvm,
    func.func(
      promote-buffers-to-stack{max-alloc-size-in-bytes=1048576}
    ),
    lower-affine,
    convert-scf-to-cf,
    convert-math-to-llvm,
    expand-strided-metadata,
    lower-affine,
    finalize-memref-to-llvm,
    convert-func-to-llvm{use-bare-ptr-memref-call-conv=true},
    convert-index-to-llvm,
    convert-cf-to-llvm,
    reconcile-unrealized-casts
  )" <<< "$KERNEL")

$DOCKER_RUN mlir-translate --mlir-to-llvmir <<< "$IR" \
  | $DOCKER_RUN llvm-link -S - ac_channel_specializations.ll \
  | $DOCKER_RUN opt -passes=always-inline -S -o 05_hw_acc.ll

section "$HOST_DUMP" "Host: the ESP calls, soda-opt"
IR=$(step "$HOST_DUMP" soda-opt $PLUGIN \
  --pass-pipeline="builtin.module(sodap-kernel-to-esp-accelerator{profile=$PROFILE dma-word-stride=$WORD_STRIDE})" <<< "$KERNEL")

if [ "$PROFILE" = true ]; then
  IR=$(step "$HOST_DUMP" soda-opt $PLUGIN \
    --pass-pipeline="builtin.module(sodap-profile-kernel)" <<< "$IR")
fi
section "$HOST_DUMP" "Host: to LLVM, soda-opt"
IR=$(step "$HOST_DUMP" soda-opt \
  --lower-all-to-llvm="use-bare-ptr-memref-call-conv" \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts <<< "$IR")

$DOCKER_RUN mlir-translate --mlir-to-llvmir -o 05_kernel_riscv.ll <<< "$IR"

# The same outlined function and argument order, executed entirely by the CPU.
# This is a baseline for this tiling/fusion schedule, not a tuned BLAS baseline.
IR=$KERNEL
if [ "$PROFILE" = true ]; then
  IR=$(step "$HOST_DUMP" soda-opt $PLUGIN \
    --pass-pipeline="builtin.module(sodap-profile-kernel)" <<< "$IR")
fi
IR=$(step "$HOST_DUMP" soda-opt \
  --lower-all-to-llvm="use-bare-ptr-memref-call-conv" \
  --convert-func-to-llvm --reconcile-unrealized-casts <<< "$IR")
$DOCKER_RUN mlir-translate --mlir-to-llvmir -o 05_cpu_baseline.ll <<< "$IR"

echo "IR after each pass: $(pwd)/$DUMP"
