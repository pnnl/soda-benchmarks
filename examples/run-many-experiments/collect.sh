#!/bin/bash

set -euo pipefail

# atax
# bicg
# doitgen
# gemm
# gemver
# gesummv
# mvt
# symm
# syr2k
# syrk
# threemm
# trmm
# twomm

KERNELS=(atax bicg doitgen gemm gemver gesummv mvt symm syr2k syrk threemm trmm twomm)
# Updte bash variables to use the curly braces for the kernel variable in the rsync commands. This ensures that the variable is correctly interpreted and avoids any potential issues with variable expansion.
for kernel in "${KERNELS[@]}"; do
    mkdir -p /workspaces/soda/soda-benchmarks/examples/run-many-experiments/results/mini-none-alo_soda/logs/${kernel}
    cd /workspaces/soda/soda-benchmarks/examples/run-many-experiments/results/mini-none-alo_soda/logs/${kernel}
    rsync -av /home/developer/soda-runs/polybench-mini-none/experiments/${kernel}__nangate45__none__simple/output/bambu/transformed/HLS_output/Synthesis/bash_flow/openroad/reports/nangate45/forward_kernel/base .
    rsync -av /home/developer/soda-runs/polybench-mini-none/experiments/${kernel}__nangate45__none__simple/*.log .

    # also rsync the results:
    rsync -av /home/developer/soda-runs/polybench-mini-none/experiments/${kernel}__nangate45__none__simple/output/bambu/transformed/HLS_output/Synthesis/bash_flow/openroad/results/nangate45/forward_kernel/base/6_1_fill.odb .
    rsync -av /home/developer/soda-runs/polybench-mini-none/experiments/${kernel}__nangate45__none__simple/output/bambu/transformed/HLS_output/Synthesis/bash_flow/openroad/results/nangate45/forward_kernel/base/6_1_fill.sdc .
    rsync -av /home/developer/soda-runs/polybench-mini-none/experiments/${kernel}__nangate45__none__simple/output/bambu/transformed/HLS_output/Synthesis/bash_flow/openroad/results/nangate45/forward_kernel/base/6_1_merged.gds .
    rsync -av /home/developer/soda-runs/polybench-mini-none/experiments/${kernel}__nangate45__none__simple/output/bambu/transformed/HLS_output/Synthesis/bash_flow/openroad/results/nangate45/forward_kernel/base/6_final.v .
done
