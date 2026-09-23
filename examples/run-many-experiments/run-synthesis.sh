#!/usr/bin/env bash
# Run every synthesis (GDS) job from setup.sh in parallel.
#
# Usage:
#   ./run-synthesis.sh
#   JOBS=2 BASE_DIR=/data/run ./run-synthesis.sh
#
# RUN ./run-simulation.sh FIRST. Both `make simulation` and `make gds` reach
# Bambu through scripts/ll_to_verilog.sh, and the simulation run already copies
# forward_kernel.v to 06_verilog.v and writes the OpenROAD scripts for the
# device. So simulation-then-gds invokes Bambu once; gds-then-simulation invokes
# it twice AND leaves 06_verilog.v newer than the GDS, forcing a re-synthesis.
#
# nangate45 only. xcu280-2Lfsvh2892-VVD falls through the device case in
# scripts/ll_to_verilog.sh, so no OpenROAD config is ever generated for it and
# scripts/to_copy/ carries no Xilinx equivalent -- there is no Vivado here
# either. setup.sh therefore emits no synthesis jobs for that device.
set -euo pipefail

BASE_DIR="${BASE_DIR:-$HOME/soda-runs/polybench-mini}"   # must match setup.sh
# OpenROAD parallelises internally, so keep this lower than the simulation run.
JOBS="${JOBS:-2}"

JOBFILE="$BASE_DIR/jobs-synthesis.txt"
JOBLOG="$BASE_DIR/jobs-synthesis.joblog"

if [[ ! -s $JOBFILE ]]; then
  echo "No synthesis jobs at $JOBFILE -- run ./setup.sh first." >&2
  exit 1
fi

opts=(--will-cite --jobs "$JOBS" --joblog "$JOBLOG" --bar)
# --resume keys off the input line number; do not reorder the job file.
if [[ -f $JOBLOG && ${RESUME:-1} == 1 ]]; then
  opts+=(--resume)
fi

echo "[run-synthesis] $(wc -l <"$JOBFILE") jobs, $JOBS at a time"
rc=0
parallel "${opts[@]}" <"$JOBFILE" || rc=$?

# Column 7 of a joblog is Exitval; column 9 onward is the command.
awk 'NR > 1 && $7 != 0 { $1=$2=$3=$4=$5=$6=$8=""; print "  FAILED: " $0 }' "$JOBLOG"
echo "[run-synthesis] parallel exit=$rc  joblog=$JOBLOG"
exit "$rc"
