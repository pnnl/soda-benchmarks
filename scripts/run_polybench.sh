#!/bin/bash
# Scaffold and build every PolyBench kernel for one dataset, then print which
# ones built. Run it in the devcontainer.
#
# Usage: scripts/run_polybench.sh <dataset> [backend] [stage]
#   dataset   TEST, MINI, SMALL, MEDIUM, LARGE or EXTRALARGE
#   backend   default esp
#   stage     default llvm
#
# Each kernel gets benches/experiments/<kernel>-<dataset>-<backend>-<stage>.
# An experiment that already exists is rebuilt rather than scaffolded again, so
# a second run only redoes what make thinks is stale. The make output of each
# kernel goes to make.log in its experiment directory.

set -u -o pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 <dataset> [backend] [stage]" >&2
  exit 1
fi

DATASET=$1
BACKEND=${2:-esp}
STAGE=${3:-llvm}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXI="pixi run --manifest-path $REPO_DIR/pixi.toml"
EXPERIMENTS="$REPO_DIR/benches/experiments"

KERNELS=$($PIXI sb-cli list | awk '{print $1}')
[ -n "$KERNELS" ] || { echo "sb-cli list returned no kernels" >&2; exit 1; }

PASSED=()
PASSED_OUTSTANDING=()
FAILED=()
FAILED_OUTSTANDING=()
for KERNEL in $KERNELS; do
  NAME="$KERNEL-$DATASET-$BACKEND-$STAGE"
  DIR="$EXPERIMENTS/$NAME"
  echo "=== $NAME"

  if [ ! -e "$DIR" ] && ! $PIXI sb-cli init --benchmark_name "$KERNEL" \
      --dataset "$DATASET" --backend "$BACKEND" --stage "$STAGE" \
      --output_dir "$NAME" --device xc7vx485t-2ffg1761 --clock_period 20.0 ; then
    FAILED+=("$KERNEL (init)")
    continue
  fi

  echo "===   Running LCS_OUTSTANDING_LOADS=true"
  if (cd "$DIR" && LCS_OUTSTANDING_LOADS=true $PIXI make > make.log 2>&1); then
    mv $DIR/output/esp $DIR/output/esp_outstanding_loads_true
    PASSED_OUTSTANDING+=("$KERNEL")
  else
    tail -5 "$DIR/make.log" >&2
    FAILED_OUTSTANDING+=("$KERNEL")
  fi

  echo "===   Running LCS_OUTSTANDING_LOADS=false"
  if (cd "$DIR" && LCS_OUTSTANDING_LOADS=false $PIXI make > make.log 2>&1); then
    mv $DIR/output/esp $DIR/output/esp_outstanding_loads_false
    PASSED+=("$KERNEL")
  else
    tail -5 "$DIR/make.log" >&2
    FAILED+=("$KERNEL")
  fi
done

echo
echo "Passed (${#PASSED[@]}): ${PASSED[*]:-}"
echo "Failed (${#FAILED[@]}): ${FAILED[*]:-}"
echo
echo "Passed outstanding (${#PASSED_OUTSTANDING[@]}): ${PASSED_OUTSTANDING[*]:-}"
echo "Failed outstanding (${#FAILED_OUTSTANDING[@]}): ${FAILED_OUTSTANDING[*]:-}"
[ "${#FAILED[@]}" -eq 0 ]
