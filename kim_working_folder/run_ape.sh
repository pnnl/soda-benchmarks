#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_ape.sh <file.mlir> linalg_ape [output_file]
    - Run only soda-linalg-ape-analysis and print resulting IR.

  ./run_ape.sh <file.mlir> affine_ape [output_file]
    - Run linalg analysis, lower to affine loops, then run insertion.
    - Prints resulting IR with @issue_APE_request callsites.

  ./run_ape.sh <file.mlir> run [output_file]
    - Run linalg analysis + affine insertion, lower to LLVM, and execute.
    - Uses mlir-cpu-runner with APE runtime support library.
EOF
}

if [[ $# -ge 1 && ( "$1" == "help" || "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 1
fi

INFILE="$1"
MODE="$2"
OUTFILE="${3:-/dev/null}"

PLUGIN="/workspaces/builds/soda-plugins/build/lib/SODAPlugin.so"
INSTR_UTILS="/workspaces/builds/soda-plugins/build/lib/libmlir_sodap_instr_runner_utils.so"
MLIR_UTILS="/opt/llvm-project/lib/libmlir_runner_utils.so"
C_UTILS="/opt/llvm-project/lib/libmlir_c_runner_utils.so"

LOWER_PIPELINE='builtin.module(
  func.func(
    convert-linalg-to-loops,
    lower-affine,
    convert-scf-to-cf,
    convert-arith-to-llvm
  ),
  convert-vector-to-llvm,
  expand-strided-metadata,
  finalize-memref-to-llvm,
  convert-func-to-llvm,
  convert-cf-to-llvm,
  reconcile-unrealized-casts
)'

run_linalg_ape() {
  mlir-opt "$INFILE" \
    -mlir-disable-threading \
    --load-pass-plugin="$PLUGIN" \
    --pass-pipeline='builtin.module(gen-addr-function-pass)' #generate-rank-function-pass)'
    # --pass-pipeline='builtin.module(func.func(soda-linalg-ape-analysis))'

    # --pass-pipeline='builtin.module(GenerateRankFunctionPass)' #soda-linalg-ape-analysis))'
    # --pass-pipeline='builtin.module(func.func(GenerateRankFunctionPass))' #soda-linalg-ape-analysis))'
}

run_affine_ape() {
  mlir-opt "$INFILE" \
    -mlir-disable-threading \
    --load-pass-plugin="$PLUGIN" \
    --pass-pipeline='builtin.module(func.func(soda-linalg-ape-analysis,convert-linalg-to-affine-loops,soda-affine-ape-insertion))'
}

run_ape_exec() {
  # run_affine_ape | \
  run_linalg_ape | \
    mlir-opt \
      -convert-linalg-to-affine-loops \
      -expand-strided-metadata \
      -lower-affine \
      -memref-expand | \
    mlir-opt \
      -convert-linalg-to-affine-loops \
      -expand-strided-metadata \
      -lower-affine \
      -convert-scf-to-cf \
      -convert-complex-to-standard \
      -convert-vector-to-llvm \
      --convert-math-to-llvm \
      --convert-math-to-libm \
      -arith-expand \
      -memref-expand \
      -convert-to-llvm='filter-dialects=memref' \
      -finalize-memref-to-llvm \
      -convert-arith-to-llvm \
      -finalize-memref-to-llvm \
      -convert-complex-to-llvm \
      -convert-func-to-llvm='use-bare-ptr-memref-call-conv=1' \
      --test-lower-to-llvm \
      -convert-cf-to-llvm \
      -reconcile-unrealized-casts \
      -symbol-dce | \
    mlir-opt \
      --pass-pipeline="$LOWER_PIPELINE" | \
    mlir-cpu-runner \
      -O0 -e main -entry-point-result=void \
      -shared-libs="$INSTR_UTILS" \
      -shared-libs="$MLIR_UTILS" \
      -shared-libs="$C_UTILS"
}

case "$MODE" in
  linalg_ape)
    if [[ $# -eq 3 ]]; then
      run_linalg_ape | tee "$OUTFILE"
    else
      run_linalg_ape
    fi
    ;;

  affine_ape)
    if [[ $# -eq 3 ]]; then
      run_affine_ape | tee "$OUTFILE"
    else
      run_affine_ape
    fi
    ;;

  run)
    if [[ $# -eq 3 ]]; then
      run_ape_exec | tee "$OUTFILE"
    else
      run_ape_exec
    fi
    ;;

  *)
    echo "Unknown mode: $MODE"
    usage
    exit 1
    ;;
esac
