#!/bin/bash
# Every pass from TOSA to the LLVM IR of a load/compute/store accelerator, of the
# RISC-V program that drives it, and of a CPU-only baseline (the flow of
# docs/LinalgToLCSPlan.md). The IR after each pass, whole modules, all goes to
# one file: <outdir>/ir_after_each_pass.mlir. Run it with pixi, in the devcontainer.
#
#   1. The kernel into a tile (SODATosaPlugin): transposes climb above
#      elementwise ops toward the arguments (sodap-tosa-raise-transposes), the
#      transposes of constants fold (sodap-tosa-fold-constant-transposes), then
#      sodap-tosa-outline-kernel moves the computation into
#      module @forward_kernel and leaves @forward, the host, the transposes of
#      its arguments and results (run by the CPU) and a dataflow.launch. Any
#      other transpose stops the script -> outlined.mlir.
#   2. In the tile: front end; compute tasks on whole tensors
#      (transform.sodap.create_tasks), each tiled on its own
#      (transform.sodap.tile_and_fuse); bufferization.
#   3. Accelerator, in the tile: load and store tasks (sodap-dataflow-load-store),
#      every task on one loop nest built on demand (sodap-dataflow-schedule),
#      ping-pong buffers and depth-1 tokens (sodap-dataflow-ping-pong), DMA,
#      nodes, LLVM; sodap-tiles-to-llvm writes the tile's LLVM IR -> 05_hw_acc.ll.
#   4. Host, @forward: TOSA to linalg, bufferization, the ESP calls
#      (sodap-launches-to-esp) -> 05_kernel_riscv.ll.
#   5. CPU baseline, a pipeline of its own from the same input: front end, no
#      tasks and no tiling, bufferization, soda-opt's outlining -> 05_cpu_baseline.ll.
#      Its argument order must be the host's, or the script stops.
#
# Usage: ./scripts/tosa_to_lcs.sh <input_tosa.mlir> [<outdir>]
# The output directory defaults to the input directory.
#
# Environment:
#   LCS_TRANSFORM        schedules, default scripts/lcs/transform.mlir
#   LCS_ENTRY            the partition to run, default tasks (ops chained through
#                        their inits share a task); tasks_trmm is trmm's from the plan
#   LCS_DMA_WORD_STRIDE  32-bit words per element in shared memory, 1 (default) or 2
#   LCS_PROFILE=true     instrument 05_kernel_riscv.ll and 05_cpu_baseline.ll with
#                        matching whole-kernel regions
#   LCS_OUTSTANDING_LOADS=false  no outstanding_load task: the load task sends each
#                        read request and waits for its data (default true)
#   LCS_SEQUENTIAL=true  the ablation with no dataflow: after the load and store
#                        tasks, every task merged into one (sodap-dataflow-merge-tasks),
#                        so it loads everything, computes everything, then stores
#                        everything, on whole buffers; no schedule, ping-pong or
#                        outstanding_load (default false)
#   LCS_TILE_SIZES       tile sizes instead of the cost model's: one list per
#                        task result, one size per loop of the op producing it
#                        (0 = untiled), e.g. '[[10, 10, 10], [10, 10]]' on gemm
#                        SMALL. tile_sizes.txt has the sizes of a run, in this form.
#   LCS_PLM_BYTES        the cost model (transform.sodap.tile_and_fuse): the tile
#   LCS_DMA_REQUEST_CYCLES  sizes with the fewest DMA cycles whose buffers fit in
#   LCS_DMA_WORD_CYCLES  LCS_PLM_BYTES (default 65536), with a request per tile
#                        row costing LCS_DMA_REQUEST_CYCLES (64) and a word
#                        LCS_DMA_WORD_CYCLES (1)
#
# Writes in <outdir>:
#   ir_after_each_pass.mlir         the IR after each pass (also split in
#                                   ir_after_each_pass_{frontend,acc,host,cpu}.mlir)
#   outlined.mlir                   the tile and the host, on TOSA
#   tasks.mlir, bufferized.mlir     after the partition and tiling, after bufferization
#   argument_order.txt              forward_kernel's arguments, as forward's numbers
#   dataflow.mlir                   after load/store, schedule and ping-pong (or the
#                                   merge, with LCS_SEQUENTIAL=true)
#   05_hw_acc.ll                    for Bambu, with architecture.xml (copies of
#   architecture.xml                tiles/forward_kernel/'s, with acc.ll)
#   host.mlir                       the host with the ESP calls
#   05_kernel_riscv.ll              for the RISC-V core
#   05_cpu_baseline.ll              the CPU-only baseline
#   cpu/                            the baseline's bufferized.mlir, outlined.mlir, kernel.mlir
#   layout.txt                      the options above
#   tile_sizes.txt                  the tile sizes used, as LCS_TILE_SIZES takes them
#   tile_cost.txt                   the cost model's DMA cycles and buffer bytes for them
#   transform_tile_sizes.mlir       with any LCS_TILE_SIZES or model variable: the
#                                   schedules, with those sizes or that model

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
# Never in the same call as $PLUGIN: both hold the Dataflow dialect.
TOSA_PLUGIN="--load-pass-plugin=${SODAP_TOSA_PLUGIN:-$(dirname "$MLIR_PLUGIN_PATH")/SODATosaPlugin.so}"
TRANSFORM="${LCS_TRANSFORM:-$REPO_DIR/scripts/lcs/transform.mlir}"
ENTRY="${LCS_ENTRY:-tasks}"
TOP=forward_kernel
TILE=tiles/$TOP
PROFILE=${LCS_PROFILE:-false}
OUTSTANDING_LOADS=${LCS_OUTSTANDING_LOADS:-true}
SEQUENTIAL=${LCS_SEQUENTIAL:-false}
WORD_STRIDE=${LCS_DMA_WORD_STRIDE:-1}
case "$WORD_STRIDE" in 1|2) ;; *) echo "LCS_DMA_WORD_STRIDE must be 1 or 2" >&2; exit 1 ;; esac
case "$PROFILE" in true|false) ;; *) echo "LCS_PROFILE must be true or false" >&2; exit 1 ;; esac
case "$OUTSTANDING_LOADS" in true|false) ;; *) echo "LCS_OUTSTANDING_LOADS must be true or false" >&2; exit 1 ;; esac
case "$SEQUENTIAL" in true|false) ;; *) echo "LCS_SEQUENTIAL must be true or false" >&2; exit 1 ;; esac
[ "$SEQUENTIAL" = true ] && OUTSTANDING_LOADS=false
printf 'dma_word_stride=%s\nprofile=%s\noutstanding_loads=%s\nsequential=%s\n' \
  "$WORD_STRIDE" "$PROFILE" "$OUTSTANDING_LOADS" "$SEQUENTIAL" > layout.txt
# LCS_TILE_SIZES or the model's variables replace the attributes of
# @tile_tasks's tile_and_fuse, the only one on %tasks.
TILE_ATTRS=
if [ -n "${LCS_TILE_SIZES:-}" ]; then
  TILE_ATTRS="tile_sizes = $LCS_TILE_SIZES"
elif [ -n "${LCS_PLM_BYTES:-}${LCS_DMA_REQUEST_CYCLES:-}${LCS_DMA_WORD_CYCLES:-}" ]; then
  TILE_ATTRS="plm_bytes = ${LCS_PLM_BYTES:-65536} : i64, dma_request_cycles = ${LCS_DMA_REQUEST_CYCLES:-64} : i64, dma_word_cycles = ${LCS_DMA_WORD_CYCLES:-1} : i64"
fi
if [ -n "$TILE_ATTRS" ]; then
  TILE_PATTERN='tile_and_fuse %tasks {[^}]*}'
  if ! grep -q "$TILE_PATTERN" "$TRANSFORM"; then
    echo "LCS_TILE_SIZES/LCS_PLM_BYTES/...: no '$TILE_PATTERN' in $TRANSFORM" >&2
    exit 1
  fi
  sed "s|$TILE_PATTERN|tile_and_fuse %tasks {$TILE_ATTRS}|" \
    "$TRANSFORM" > transform_tile_sizes.mlir
  TRANSFORM=$PWD/transform_tile_sizes.mlir
fi
OUTSTANDING_LOAD=
[ "$OUTSTANDING_LOADS" = true ] && OUTSTANDING_LOAD="func.func(sodap-dataflow-outstanding-load), canonicalize,"
# Between the load and store tasks and the DMA: the dataflow schedule, or one
# sequential task.
if [ "$SEQUENTIAL" = true ]; then
  STAGES="func.func(sodap-dataflow-merge-tasks)"
else
  STAGES="func.func(sodap-dataflow-schedule),
    fold-memref-alias-ops,
    func.func(sodap-dataflow-ping-pong)"
fi
DUMP=ir_after_each_pass.mlir
FRONTEND_DUMP=ir_after_each_pass_frontend.mlir
ACC_DUMP=ir_after_each_pass_acc.mlir
HOST_DUMP=ir_after_each_pass_host.mlir
CPU_DUMP=ir_after_each_pass_cpu.mlir
for dump in "$DUMP" "$FRONTEND_DUMP" "$ACC_DUMP" "$HOST_DUMP" "$CPU_DUMP"; do
  : > "$dump"
done

# Run a tool, append its IR after each pass to the dump and to $1. The dumps also
# get any error, followed by the IR of the failed pass, so show the diagnostics
# (or the end of $1 if there are none) when the tool fails.
step() {
  local where=$1
  shift
  $DOCKER_RUN "$@" --mlir-disable-threading --mlir-print-ir-after-all \
    --mlir-print-ir-module-scope 2> >(tee -a "$DUMP" "$where" > /dev/null) \
    || { grep -E "(error|note):" "$where" >&2 || tail -20 "$where" >&2; return 1; }
}

section() {
  local where=$1
  shift
  printf '\n// ========== %s ==========\n\n' "$1" | tee -a "$DUMP" "$where" > /dev/null
}

# soda-opt's outlining of $2 (memrefs, no results) in directory $3: outlined.mlir,
# argument_order.txt and kernel.mlir. Dumps to $1.
outline() {
  local where=$1 input=$2 dir=$3
  section "$where" "Outlining forward_kernel ($dir), soda-opt"
  step "$where" soda-opt "$input" $PLUGIN --convert-all-to-soda -soda-outline-bambu-code \
    -o "$dir/outlined_raw.mlir"
  step "$where" soda-opt "$dir/outlined_raw.mlir" $PLUGIN \
    --pass-pipeline="builtin.module(sodap-order-kernel-arguments)" -o "$dir/outlined.mlir"
  python3 "$REPO_DIR/scripts/lcs/argument_order.py" "$dir/outlined.mlir" > "$dir/argument_order.txt"
  step "$where" soda-opt "$dir/outlined.mlir" $PLUGIN \
    -soda-extract-arguments-to-c-testbench=using-bare-ptr \
    -soda-generate-bambu-accelcode=no-aa -o "$dir/kernel.mlir"
}

FRONT_END="func.func(
      tosa-to-arith,
      tosa-to-tensor,
      tosa-to-linalg-named,
      tosa-to-linalg,
      sodap-linalg-rank-reduce-contractions
    ),
    linalg-fold-unit-extent-dims,
    linalg-fuse-elementwise-ops,
    func.func(sodap-linalg-constants-to-indices),
    canonicalize"
BUFFERIZE="one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},
    canonicalize,
    cse,
    canonicalize,
    buffer-results-to-out-params"

section "$FRONTEND_DUMP" "The kernel into a tile, the transposes to the host, mlir-opt"
step "$FRONTEND_DUMP" mlir-opt "$INPUT" $TOSA_PLUGIN --pass-pipeline="builtin.module(
    func.func(
      sodap-tosa-raise-transposes,
      sodap-tosa-fold-constant-transposes
    ),
    sodap-tosa-outline-kernel
  )" -o outlined.mlir

section "$FRONTEND_DUMP" "Tile: front end, compute tasks and tiling ($ENTRY), mlir-opt"
step "$FRONTEND_DUMP" mlir-opt outlined.mlir $PLUGIN --pass-pipeline="builtin.module(
    transform-preload-library{transform-library-paths=$TRANSFORM},
    builtin.module(
      $FRONT_END,
      transform-interpreter{entry-point=$ENTRY}
    )
  )" -o tasks.mlir
# tile_and_fuse's remark: the sizes it used.
{ grep -o 'remark: tile_sizes = .*' "$FRONTEND_DUMP" || true; } \
  | sed 's/^remark: tile_sizes = //' > tile_sizes.txt
{ grep -o 'remark: dma_cycles = .*' "$FRONTEND_DUMP" || true; } \
  | sed 's/^remark: //' > tile_cost.txt

section "$FRONTEND_DUMP" "Tile: bufferization, mlir-opt"
step "$FRONTEND_DUMP" mlir-opt tasks.mlir $PLUGIN --pass-pipeline="builtin.module(
    builtin.module(
      $BUFFERIZE
    )
  )" -o bufferized.mlir

section "$ACC_DUMP" "Tile: load and store tasks, $([ "$SEQUENTIAL" = true ] && echo "merged into one" || echo "schedule, ping-pong"), mlir-opt"
step "$ACC_DUMP" mlir-opt bufferized.mlir $PLUGIN --pass-pipeline="builtin.module(
    builtin.module(
      canonicalize,
      func.func(sodap-dataflow-load-store),
      canonicalize,
      $STAGES,
      canonicalize
    )
  )" -o dataflow.mlir

# The passes taking top-func find the kernel by the tile's name, and write
# their files in $TILE.
section "$ACC_DUMP" "Tile: to nodes and LLVM, mlir-opt"
step "$ACC_DUMP" mlir-opt dataflow.mlir $PLUGIN --pass-pipeline="builtin.module(
    builtin.module(
      sodap-dataflow-copy-to-dma{dma-word-stride=$WORD_STRIDE},
      $OUTSTANDING_LOAD
      func.func(
        sodap-dataflow-tasks-to-nodes
      ),
      sodap-dataflow-nodes-to-func,
      func.func(
        convert-linalg-to-loops
      ),
      sodap-dataflow-infer-top-interface,
      sodap-emit-bambu-architecture{file=architecture.xml},
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
    ),
    sodap-tiles-to-llvm
  )" -o acc.mlir

$DOCKER_RUN llvm-link -S - "$TILE/ac_channel_specializations.ll" < "$TILE/acc.ll" \
  | $DOCKER_RUN opt -passes=always-inline -S -o "$TILE/05_hw_acc.ll"
cp "$TILE/05_hw_acc.ll" "$TILE/architecture.xml" .

section "$HOST_DUMP" "Host: TOSA to linalg, bufferization, the ESP calls, mlir-opt"
step "$HOST_DUMP" mlir-opt acc.mlir $PLUGIN --pass-pipeline="builtin.module(
    func.func(
      tosa-to-arith,
      tosa-to-tensor,
      tosa-to-linalg-named,
      tosa-to-linalg
    ),
    $BUFFERIZE,
    sodap-launches-to-esp{host=$TOP profile=$PROFILE dma-word-stride=$WORD_STRIDE}
  )" -o host.mlir
# forward_kernel's arguments are @forward's, then its results.
N_ARGS=$(grep -m1 -oE "func.func @$TOP\([^)]*\)" host.mlir | grep -oE '%arg[0-9]+:' | wc -l)
seq -s ' ' 0 $((N_ARGS - 1)) > argument_order.txt
HOST=host.mlir
if [ "$PROFILE" = true ]; then
  step "$HOST_DUMP" soda-opt host.mlir $PLUGIN \
    --pass-pipeline="builtin.module(sodap-profile-kernel)" -o host_profiled.mlir
  HOST=host_profiled.mlir
fi
section "$HOST_DUMP" "Host: to LLVM, soda-opt"
step "$HOST_DUMP" soda-opt "$HOST" --lower-all-to-llvm="use-bare-ptr-memref-call-conv" \
  --convert-func-to-llvm --reconcile-unrealized-casts -o host_llvm.mlir
$DOCKER_RUN mlir-translate --mlir-to-llvmir -o 05_kernel_riscv.ll host_llvm.mlir

# The CPU baseline: a pipeline of its own, with no tasks and no tiling.
mkdir -p cpu
section "$CPU_DUMP" "CPU baseline: front end and bufferization, mlir-opt"
step "$CPU_DUMP" mlir-opt "$INPUT" $PLUGIN --pass-pipeline="builtin.module(
    $FRONT_END,
    $BUFFERIZE
  )" -o cpu/bufferized.mlir
outline "$CPU_DUMP" cpu/bufferized.mlir cpu
if ! cmp -s argument_order.txt cpu/argument_order.txt; then
  echo "The CPU baseline's arguments are in another order: $(cat cpu/argument_order.txt)," \
    "not the host's $(cat argument_order.txt)" >&2
  exit 1
fi
CPU_KERNEL=cpu/kernel.mlir
if [ "$PROFILE" = true ]; then
  step "$CPU_DUMP" soda-opt cpu/kernel.mlir $PLUGIN \
    --pass-pipeline="builtin.module(sodap-profile-kernel)" -o cpu/kernel_profiled.mlir
  CPU_KERNEL=cpu/kernel_profiled.mlir
fi
section "$CPU_DUMP" "CPU baseline: to LLVM, soda-opt"
step "$CPU_DUMP" soda-opt "$CPU_KERNEL" --lower-all-to-llvm="use-bare-ptr-memref-call-conv" \
  --convert-func-to-llvm --reconcile-unrealized-casts -o cpu/kernel_llvm.mlir
$DOCKER_RUN mlir-translate --mlir-to-llvmir -o 05_cpu_baseline.ll cpu/kernel_llvm.mlir

echo "IR after each pass: $(pwd)/$DUMP"
