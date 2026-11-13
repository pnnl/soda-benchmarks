#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Run from the configured Emilia shell; minicom records independently.
set -Eeuo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)
campaign=${CAMPAIGN:-$repo/local-results/vc707-20261104-buffered}
apps=${APPS:-$repo/local-results/vc707-20261104-startup}
wait_seconds=${WAIT_SECONDS:-15}
repeats=${REPEATS:-1}
program_timeout=${PROGRAM_TIMEOUT:-180}
load_timeout=${LOAD_TIMEOUT:-60}
dry_run=false
if [[ ${1:-} == --dry-run ]]; then dry_run=true; shift; fi
if [[ ${1:-} == --help ]]; then
  echo "Usage: $0 [--dry-run] [gemm syr2k trmm symm syrk gesummv doitgen]"
  echo 'Defaults: all seven kernels; REPEATS=1; WAIT_SECONDS=15 after each app.'
  echo 'Command limits: PROGRAM_TIMEOUT=180 and LOAD_TIMEOUT=60 seconds.'
  echo 'Uses the existing FPGA environment and independent minicom recording.'
  exit 0
fi
kernels=("$@")
if ((${#kernels[@]} == 0)); then kernels=(gemm syr2k trmm symm syrk gesummv doitgen); fi
[[ $wait_seconds =~ ^[1-9][0-9]*$ && $repeats =~ ^[1-9][0-9]*$ ]] || {
  echo 'WAIT_SECONDS and REPEATS must be positive integers.' >&2; exit 1;
}
[[ $program_timeout =~ ^[1-9][0-9]*$ && $load_timeout =~ ^[1-9][0-9]*$ ]] || {
  echo 'PROGRAM_TIMEOUT and LOAD_TIMEOUT must be positive integers.' >&2; exit 1;
}
if ! "$dry_run"; then
  for variable in FPGA_HOST XIL_HW_SERVER_PORT ESPLINK_IP ESPLINK_PORT; do
    [[ -n ${!variable:-} ]] || { echo "Missing exported $variable in this shell." >&2; exit 1; }
  done
  for tool in make vivado riscv64-unknown-elf-objcopy timeout stdbuf; do
    command -v "$tool" >/dev/null || { echo "Missing $tool in PATH." >&2; exit 1; }
  done
fi
# Check all inputs before programming any image.
for kernel in "${kernels[@]}"; do
  case "$kernel" in gemm|syr2k|trmm|symm|syrk|gesummv|doitgen) ;;
    *) echo "Unknown kernel: $kernel" >&2; exit 1 ;; esac
  for file in "$campaign/$kernel/esp/socs/vc707/Makefile" \
              "$campaign/$kernel/esp/socs/vc707/top.bit" \
              "$apps/$kernel/deploy/$kernel-cpu.exe" "$apps/$kernel/deploy/$kernel.exe"; do
    [[ -s $file ]] || { echo "Missing input: $file" >&2; exit 1; }
  done
  (cd "$apps/$kernel/deploy" && sha256sum --status -c SHA256SUMS)
done
mkdir -p "$repo/local-results"
logs=$(mktemp -d "$repo/local-results/manual-board-$(date +%Y%m%d-%H%M%S).XXXXXX")
run_label=RUN
if "$dry_run"; then run_label='DRY RUN'; fi
echo "Command logs: $logs"
echo "${#kernels[@]} bitstreams; $repeats CPU/offload pair(s) each; ${wait_seconds}s after each app."
trap 'echo "Stopped: command failed. Inspect $logs and the minicom recording." >&2' ERR

run_step() {
  local log=$1
  local limit=$2
  shift 2
  printf '\n[%s] %s: ' "$(date --iso-8601=seconds)" "$run_label" | tee -a "$log"
  printf '%q ' "$@" | tee -a "$log"
  printf '\n' | tee -a "$log"
  if ! "$dry_run"; then
    # Bound the entire make process group, including an ESPLink retry loop.
    # Line buffering exposes the reset/download stage immediately through tee.
    timeout --verbose --kill-after=5s "${limit}s" stdbuf -oL -eL "$@" 2>&1 | tee -a "$log"
  fi
}

for kernel in "${kernels[@]}"; do
  soc="$campaign/$kernel/esp/socs/vc707"
  deploy="$apps/$kernel/deploy"
  mkdir -p "$logs/$kernel"
  run_step "$logs/$kernel/program.log" "$program_timeout" make -C "$soc" fpga-program
  # ESP's Vivado make recipe can hide the tool's nonzero exit status.
  if grep -Eq '(^|[[:space:]])ERROR:' "$logs/$kernel/program.log"; then
    echo "Programming failed for $kernel; stopping." >&2; exit 1
  fi
  if ! "$dry_run" && ! grep -q 'End of startup status: HIGH' "$logs/$kernel/program.log"; then
    echo "No successful Vivado programming confirmation for $kernel; stopping." >&2; exit 1
  fi
  for ((repeat=1; repeat<=repeats; repeat++)); do
    for mode in cpu esp; do
      exe="$deploy/$kernel.exe"
      if [[ $mode == cpu ]]; then exe="$deploy/$kernel-cpu.exe"; fi
      run_step "$logs/$kernel/$mode-$repeat.log" "$load_timeout" make -C "$soc" fpga-run "TEST_PROGRAM=$exe"
      echo "Waiting ${wait_seconds}s for $kernel / $mode / run $repeat to finish and print."
      if ! "$dry_run"; then sleep "$wait_seconds"; fi
    done
  done
done
echo "Sequence finished. Check minicom's UART recording for numerical results and profiling."
echo "Command logs: $logs"
