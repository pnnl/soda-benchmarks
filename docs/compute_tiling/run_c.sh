#!/bin/bash
# Synthesize and simulate mm.c with the host's Bambu (/opt/panda), Verilator.
# Usage: run_c.sh <outdir> <fp_units> [-DMR=.. -DNR=.. -DPA=.. -DPB=.. -DPC=.. -DREP=..]
#   fp_units  instances of Bambu's float multiplier and of its adder
# Prints the II of each pipelined loop, the simulated cycles, the testbench's
# mismatches and Bambu's area estimate of mm.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1; P=$2; shift 2
mkdir -p "$OUT" && cd "$OUT"
source /opt/panda/settings.sh
bambu "$HERE/mm.c" --top-fname=mm --compiler=I386_CLANG19 --clock-period=20.0 \
  --device=xc7vx485t-2ffg1761 -v3 --no-clean "$@" \
  -C="__float_mule8m23b_127nih=$P,__float_adde8m23b_127nih=$P" \
  --simulate --simulator=VERILATOR --generate-tb="$HERE/tb.c" > bambu.log 2>&1
echo "rc=$? $(grep -o 'II=[0-9]* solution' bambu.log | tr '\n' ' ')"
grep -h -o -E 'Total cycles *: [0-9]+|mismatches [0-9]+' bambu.log | sort -u
grep -A12 'Module binding information for function mm:' bambu.log | grep -o 'Total estimated area: [0-9]*'
