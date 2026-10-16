#!/usr/bin/env bash
# Scaffold the PolyBench MINI HLS matrix and emit GNU parallel job files.
#
# Every configuration is scaffolded ONCE, at --stage simulation. The generated
# Makefile also carries `simulation` and `gds` aliases, so run-simulation.sh and
# run-synthesis.sh drive the same build tree instead of two copies of it.
#
# Usage:
#   ./setup.sh                    # defaults below
#   BASE_DIR=/data/run ./setup.sh
#
# Re-running is safe: existing experiments are skipped but still contribute
# their job lines, so the job files are always complete.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# --- knobs -----------------------------------------------------------------
# BASE_DIR must match the one the runner scripts use: they read the job files
# from it, and `parallel --resume` keys off the joblog stored beside them.
BASE_DIR="${BASE_DIR:-$HOME/soda-runs/polybench-mini-none-simple}"
PIXI_CMD="${PIXI_CMD:-pixi run --manifest-path $REPO_ROOT/pixi.toml}"

DATASET="${DATASET:-MINI}"        # alo_soda schedules exist for MINI only
DTYPE="${DTYPE:-float32}"
CLOCK_PERIOD="${CLOCK_PERIOD:-5}"
FLOW="${FLOW:-transformed}"       # the only flow that reads transform.mlir
BACKEND="${BACKEND:-bambu}"
STAGE="${STAGE:-simulation}"

# Available options
# KERNELS=(atax bicg doitgen gemm gemver gesummv mvt symm syr2k syrk threemm trmm twomm)
# DEVICES=(nangate45 xcu280-2Lfsvh2892-VVD)
# INSTRUMENTATIONS=(none assert change-location change-location-exposed-ports hw-counters)
# SCHEDULES=(simple alo_soda)

KERNELS=(atax bicg doitgen gemm gemver gesummv mvt symm syr2k syrk threemm trmm twomm)
DEVICES=(nangate45)
INSTRUMENTATIONS=(none)
SCHEDULES=(simple)

# Only these reach OpenROAD. scripts/ll_to_verilog.sh writes an OpenROAD config
# for asap7*/nangate45 and nothing else, and scripts/to_copy/ carries no Xilinx
# files, so xcu280 has no synthesis path in this repo -- it stops at simulation.
SYNTHESIZABLE_DEVICES=(nangate45)

# --- setup -----------------------------------------------------------------
EXP_ROOT="$BASE_DIR/experiments"
SIM_JOBS="$BASE_DIR/jobs-simulation.txt"
SYN_JOBS="$BASE_DIR/jobs-synthesis.txt"
mkdir -p "$EXP_ROOT" "$BASE_DIR/logs"
: >"$SIM_JOBS"
: >"$SYN_JOBS"

device_tag() { printf '%s' "${1%%-*}"; }   # xcu280-2Lfsvh2892-VVD -> xcu280

is_synthesizable() {
  local want=$1 dev
  for dev in "${SYNTHESIZABLE_DEVICES[@]}"; do
    [[ $want == "$dev" ]] && return 0
  done
  return 1
}

n_new=0 n_skip=0 n_fail=0

# Sequential on purpose: `sb-cli init` read-modify-writes experiments/registry.py
# and creates a symlink, so parallelising it would race both. 260 inits are cheap
# next to the builds they set up.
for kernel in "${KERNELS[@]}"; do
  for device in "${DEVICES[@]}"; do
    for instr in "${INSTRUMENTATIONS[@]}"; do
      for ts in "${SCHEDULES[@]}"; do
        name="${kernel}__$(device_tag "$device")__${instr}__${ts}"
        exp="$EXP_ROOT/$name"

        if [[ -e $exp || -L $exp ]]; then
          n_skip=$((n_skip + 1))
        elif $PIXI_CMD sb-cli init \
              --base_dir "$BASE_DIR" --output_dir "$name" \
              --benchmark_name "$kernel" --dataset "$DATASET" --dtype "$DTYPE" \
              --device "$device" --clock_period "$CLOCK_PERIOD" \
              --flow "$FLOW" --backend "$BACKEND" --stage "$STAGE" \
              --instrumentation "$instr" --default_ts "$ts" \
              >"$BASE_DIR/logs/init-$name.log" 2>&1; then
          n_new=$((n_new + 1))
        else
          echo "[setup] FAILED init: $name (logs/init-$name.log)" >&2
          n_fail=$((n_fail + 1))
          continue
        fi

        # PIXI_CMD is expanded here, at generation time, so the job lines carry
        # no '$' for the shell parallel spawns to re-expand.
        printf 'cd %q && %s make simulation > build-simulation.log 2>&1\n' \
          "$exp" "$PIXI_CMD" >>"$SIM_JOBS"
        if is_synthesizable "$device"; then
          printf 'cd %q && %s make gds > build-gds.log 2>&1\n' \
            "$exp" "$PIXI_CMD" >>"$SYN_JOBS"
        fi
      done
    done
  done
done

echo "[setup] scaffolded=$n_new skipped=$n_skip failed=$n_fail"
echo "[setup] $(wc -l <"$SIM_JOBS") simulation jobs -> $SIM_JOBS"
echo "[setup] $(wc -l <"$SYN_JOBS") synthesis jobs  -> $SYN_JOBS"
echo "[setup] next: BASE_DIR=$BASE_DIR ./run-simulation.sh"
