#!/bin/bash

set -e -o pipefail

# Define output zip name with current date and time
ZIP_NAME="artifacts/eels_asic_artifacts_$(date +'%Y%m%d_%H%M%S').zip"

# Collect files
zip -j "$ZIP_NAME" \
  output/bambu/baseline/input.ll \
  output/bambu/baseline/bambu-log \
  output/bambu/baseline/06_verilog.v \
  output/bambu/baseline/07_results.txt \
  output/frozen_graph.pbtxt \
  output/bambu/baseline/forward_kernel.v \
  output/forward_kernel_testbench.c \
  tfscript.py