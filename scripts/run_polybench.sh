#!/bin/bash
# Scaffold and build every PolyBench kernel for one dataset, then print which
# ones built. Run it in the devcontainer.
#
# Usage: scripts/run_polybench.sh [--sequential] <dataset> [backend] [stage]
#   --sequential  also build the sequential ablation (LCS_SEQUENTIAL=true, see
#                 tosa_to_lcs.sh): no dataflow, whole buffers, small datasets only
#   dataset       TEST, MINI, SMALL, MEDIUM, LARGE or EXTRALARGE
#   backend       default esp
#   stage         default llvm
#
# Each kernel gets benches/experiments/<kernel>-<dataset>-<backend>-<stage>.
# An experiment that already exists is rebuilt rather than scaffolded again.
# Each run builds in output/esp, then moves it to output/esp_outstanding_loads_
# {true,false} or output/esp_sequential; a second run moves that back first, so
# make only redoes what is stale. The make output of each run goes to make.log
# in its experiment directory.

set -u -o pipefail

SEQUENTIAL=false
if [ "${1:-}" = --sequential ]; then
  SEQUENTIAL=true
  shift
fi

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 [--sequential] <dataset> [backend] [stage]" >&2
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

# Build $DIR with the environment "$@" into output/$1.
build() {
  local out="$DIR/output/$1" esp="$DIR/output/esp"
  shift
  [ -d "$out" ] && [ ! -e "$esp" ] && mv -T "$out" "$esp"
  if ! (cd "$DIR" && env "$@" $PIXI make > make.log 2>&1); then
    tail -5 "$DIR/make.log" >&2
    return 1
  fi
  # A leftover output/esp (a failed run) kept the old $out in place.
  mv -T "$esp" "$out" || { echo "built, but $out is in the way: the build is in $esp" >&2; return 1; }
}

PASSED=()
PASSED_OUTSTANDING=()
PASSED_SEQUENTIAL=()
FAILED=()
FAILED_OUTSTANDING=()
FAILED_SEQUENTIAL=()
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
  if build esp_outstanding_loads_true LCS_OUTSTANDING_LOADS=true; then
    PASSED_OUTSTANDING+=("$KERNEL")
  else
    FAILED_OUTSTANDING+=("$KERNEL")
  fi

  echo "===   Running LCS_OUTSTANDING_LOADS=false"
  if build esp_outstanding_loads_false LCS_OUTSTANDING_LOADS=false; then
    PASSED+=("$KERNEL")
  else
    FAILED+=("$KERNEL")
  fi

  [ "$SEQUENTIAL" = true ] || continue
  echo "===   Running LCS_SEQUENTIAL=true"
  if build esp_sequential LCS_SEQUENTIAL=true LCS_OUTSTANDING_LOADS=false; then
    PASSED_SEQUENTIAL+=("$KERNEL")
  else
    FAILED_SEQUENTIAL+=("$KERNEL")
  fi
done

echo
echo "Passed (${#PASSED[@]}): ${PASSED[*]:-}"
echo "Failed (${#FAILED[@]}): ${FAILED[*]:-}"
echo
echo "Passed outstanding (${#PASSED_OUTSTANDING[@]}): ${PASSED_OUTSTANDING[*]:-}"
echo "Failed outstanding (${#FAILED_OUTSTANDING[@]}): ${FAILED_OUTSTANDING[*]:-}"
if [ "$SEQUENTIAL" = true ]; then
  echo
  echo "Passed sequential (${#PASSED_SEQUENTIAL[@]}): ${PASSED_SEQUENTIAL[*]:-}"
  echo "Failed sequential (${#FAILED_SEQUENTIAL[@]}): ${FAILED_SEQUENTIAL[*]:-}"
fi
[ "${#FAILED[@]}" -eq 0 ]
