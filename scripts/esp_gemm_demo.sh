#!/bin/bash
# End-to-end demo of the cpu and esp backends on PolyBench gemm.
#
# Scaffolds and builds three experiments, each answering a different question:
#
#   1. cpu / baseline        -- does the torch -> TOSA -> linalg -> soda -> LLVM
#                               -> native pipeline compute gemm correctly?
#   2. cpu / esp schedule    -- does the ESP lowering fire, with the right
#                               dimensions, and does it link against the runtime?
#   3. esp / transformed     -- does it cross-compile for the SoC's RISC-V core,
#                               and is the baremetal application complete?
#
# Usage: scripts/esp_gemm_demo.sh [--dataset MINI] [--keep]
#
# Every run scaffolds fresh directories, so it is safe to repeat: sb-cli appends
# a -NNN counter when a name is taken. --keep skips nothing; it is the default
# and is accepted only so the intent reads in a script that calls this one.

set -e -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET="MINI"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --keep) shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

BENCH="PolyBenchPyTorch.linear_algebra.blas.gemm"
SB="pixi run --manifest-path $REPO_DIR/pixi.toml sb-cli"
MAKE="pixi run --manifest-path $REPO_DIR/pixi.toml make"

banner() {
  echo
  echo "==============================================================="
  echo "  $*"
  echo "==============================================================="
}

# sb-cli prints "Symlink: <path> -> ..."; that path is the experiment directory.
scaffold() {
  local out
  out=$($SB init --benchmark_name "$BENCH" --dataset "$DATASET" --dtype float32 "$@" 2>&1)
  echo "$out" >&2
  echo "$out" | sed -n 's/.*Symlink: \([^ ]*\) ->.*/\1/p'
}

cd "$REPO_DIR"

# --------------------------------------------------------------------------
banner "1/3  cpu backend, baseline flow -- numeric check against PyTorch"
DIR=$(scaffold --flow baseline --backend cpu --stage simulation)
(cd "$DIR" && $MAKE && $MAKE check)
echo "[demo] results: $DIR/output/cpu/baseline/07_results.txt"

# --------------------------------------------------------------------------
banner "2/3  cpu backend, esp schedule -- the ESP call sequence, on the host"
DIR=$(scaffold --flow transformed --instrumentation esp \
        --backend cpu --stage simulation)
(cd "$DIR" && $MAKE)
echo
echo "[demo] The seven esp_* calls above are the offload. The mock runtime does"
echo "[demo] not compute, so the mismatch that follows them is expected -- what"
echo "[demo] this checks is that the pass fired and the runtime ABI matches."
echo "[demo] results: $DIR/output/cpu/transformed/07_results.txt"

# --------------------------------------------------------------------------
banner "3/3  esp backend -- RISC-V object and a buildable baremetal application"
DIR=$(scaffold --flow transformed --backend esp --stage object)
(cd "$DIR" && $MAKE)
OBJ="$DIR/output/esp/transformed/06_kernel_riscv.o"
echo
file "$OBJ"
echo "[demo] undefined ESP symbols the runtime resolves:"
llvm-nm "$OBJ" | grep ' U esp_' || true
echo "[demo] baremetal application staged in:"
ls "$DIR/output/esp/transformed/esp-app"
echo
echo "[demo] Build that directory inside an ESP checkout to get a binary --"
echo "[demo] see its README.md, or set ESP_ROOT and ask for --stage binary."
