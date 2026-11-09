#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
source "$here/env.sh"
kind=${1:?Usage: replay-operator.sh conv|add|gemm [fresh-run-root] [hardware-dir] [model-dir]}
case "$kind" in conv|add|gemm) ;; *) echo 'Expected conv, add or gemm' >&2; exit 2 ;; esac
if [[ -n ${2:-} ]]; then
  run=$(realpath -m "$2")
  mkdir "$run" # Refuse an existing result directory.
else
  mkdir -p "$repo/build"
  run=$(mktemp -d "$repo/build/replay-$kind.XXXXXX")
fi
hardware=${3:-$SODA_VALIDATED/build/resnet18/hardware-lcs}
model=${4:-$SODA_VALIDATED/build/resnet18/operators/$kind}
trap 'echo "Replay failed; inspect $run/software.log and $run/system.log" >&2' ERR
cp -a "$model/esp-app" "$run/app"
printf 'Rebuilding software and running %s; logs: %s\n' "$kind" "$run"
bash "$repo/examples/esp-validation/reusable-nn/build-software.sh" \
  "$run/app" "$ESP_ROOT" "$ESP_SOC" > "$run/software.log" 2>&1
bash "$repo/examples/esp-validation/reusable-nn/run-system.sh" \
  "$hardware" "$run/app" "$ESP_SOC" "$run/system" > "$run/system.log" 2>&1
cat "$run/system/full-system/validation.json"
