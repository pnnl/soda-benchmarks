#!/bin/bash
# The accelerator's LLVM IR with a second (register) tiling of the matmul, by
# hand. Re-runs tosa_to_lcs.sh's "Tile: to nodes and LLVM" step on a build's
# dataflow.mlir, with micro.mlir applied between sodap-dataflow-nodes-to-func and
# convert-linalg-to-loops. Run it in the devcontainer. Keep the pipeline below in
# sync with tosa_to_lcs.sh: without MR and NR, accelerator.ll must be identical
# to the build's tiles/forward_kernel/05_hw_acc.ll.
#
# Usage: micro_tiling.sh <build> <outdir> [MR NR]
#   build   a tosa_to_lcs.sh output directory (dataflow.mlir, layout.txt,
#           tiles/forward_kernel/ac_channel_specializations.ll)
#   MR NR   register block, dividing the memory tile; omit both for none
# Writes <outdir>/accelerator.ll and <outdir>/architecture.xml, for Bambu.
set -euo pipefail
[ "$#" -eq 2 ] || [ "$#" -eq 4 ] || { echo "Usage: $0 <build> <outdir> [MR NR]" >&2; exit 1; }
BUILD=$(cd "$1" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
P=${MLIR_PLUGIN_PATH:-$REPO/examples/soda-plugins/build/lib/SODAPlugin.so}
PLUGIN="--load-dialect-plugin=$P --load-pass-plugin=$P"
STRIDE=$(sed -n 's/^dma_word_stride=//p' "$BUILD/layout.txt")
OUTSTANDING_LOAD=
grep -q '^outstanding_loads=true' "$BUILD/layout.txt" \
  && OUTSTANDING_LOAD="func.func(sodap-dataflow-outstanding-load), canonicalize,"
mkdir -p "$2/tiles/forward_kernel"
cd "$2"
cp "$BUILD/tiles/forward_kernel/ac_channel_specializations.ll" tiles/forward_kernel/

mlir-opt "$BUILD/dataflow.mlir" $PLUGIN --pass-pipeline="builtin.module(builtin.module(
    sodap-dataflow-copy-to-dma{dma-word-stride=$STRIDE},
    $OUTSTANDING_LOAD
    func.func(sodap-dataflow-tasks-to-nodes),
    sodap-dataflow-nodes-to-func))" -o nodes.mlir
if [ "$#" -eq 4 ]; then
  sed "s/@MR@/$3/g; s/@NR@/$4/g" "$HERE/micro.mlir" > micro.mlir
  mlir-opt nodes.mlir $PLUGIN --pass-pipeline="builtin.module(
      transform-preload-library{transform-library-paths=$PWD/micro.mlir},
      builtin.module(transform-interpreter{entry-point=micro}, canonicalize, cse))" \
    -o nodes_micro.mlir
else
  cp nodes.mlir nodes_micro.mlir
fi
mlir-opt nodes_micro.mlir $PLUGIN --pass-pipeline="builtin.module(
    builtin.module(
      func.func(convert-linalg-to-loops),
      sodap-dataflow-infer-top-interface,
      sodap-emit-bambu-architecture{file=architecture.xml},
      sodap-convert-dataflow-to-llvm,
      func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=1048576}),
      lower-affine, convert-scf-to-cf, convert-math-to-llvm, expand-strided-metadata,
      lower-affine, finalize-memref-to-llvm,
      convert-func-to-llvm{use-bare-ptr-memref-call-conv=true},
      convert-index-to-llvm, convert-cf-to-llvm, reconcile-unrealized-casts),
    sodap-tiles-to-llvm)" -o acc.mlir
llvm-link -S - tiles/forward_kernel/ac_channel_specializations.ll < tiles/forward_kernel/acc.ll \
  | opt -passes=always-inline -S -o accelerator.ll
cp tiles/forward_kernel/architecture.xml .
