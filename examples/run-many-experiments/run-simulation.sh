#!/usr/bin/env bash
# Run every simulation job from setup.sh in parallel.
#
# Usage:
#   ./run-simulation.sh                      # defaults below
#   JOBS=8 BASE_DIR=/data/run ./run-simulation.sh
#   RESUME=0 ./run-simulation.sh             # re-run everything from scratch
#
# Each job cd's into its experiment and runs `make simulation`, writing
# build-simulation.log there. Failures do not stop the sweep; they are listed at
# the end and re-run on the next invocation.
set -euo pipefail

BASE_DIR="${BASE_DIR:-$HOME/soda-runs/polybench-mini}"   # must match setup.sh
# Bambu runs Verilator with --verilator-parallel, so each job is already
# multi-threaded; a low job count beats oversubscribing the machine.
JOBS="${JOBS:-4}"

JOBFILE="$BASE_DIR/jobs-simulation.txt"
JOBLOG="$BASE_DIR/jobs-simulation.joblog"

if [[ ! -s $JOBFILE ]]; then
  echo "No simulation jobs at $JOBFILE -- run ./setup.sh first." >&2
  exit 1
fi

opts=(--will-cite --jobs "$JOBS" --joblog "$JOBLOG" --bar)
# --resume skips entries the joblog already records. It keys off the input LINE
# NUMBER, so never reorder or hand-edit the job file between runs.
if [[ -f $JOBLOG && ${RESUME:-1} == 1 ]]; then
  opts+=(--resume)
fi

echo "[run-simulation] $(wc -l <"$JOBFILE") jobs, $JOBS at a time"
rc=0
parallel "${opts[@]}" <"$JOBFILE" || rc=$?

# Column 7 of a joblog is Exitval; column 9 onward is the command.
awk 'NR > 1 && $7 != 0 { $1=$2=$3=$4=$5=$6=$8=""; print "  FAILED: " $0 }' "$JOBLOG"
echo "[run-simulation] parallel exit=$rc  joblog=$JOBLOG"
echo "[run-simulation] next: BASE_DIR=$BASE_DIR ./run-synthesis.sh"
exit "$rc"
