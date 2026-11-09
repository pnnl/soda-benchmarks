# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# vivado -mode batch -source synthesize.tcl -tclargs hardware-dir fresh-run-dir
set generated [file normalize [file join [lindex $argv 0] combined]]
set run [file normalize [lindex $argv 1]]
if {[file exists $run]} {error "Use a fresh synthesis directory: $run"}
set adapter [file normalize [file join [file dirname [info script]] ../generated-gemm/sqr_bambu_basic_dma64.sv]]
file mkdir $run
cd $run
set_param general.maxThreads 4
foreach memory [glob -nocomplain $generated/*.mem] {file copy $memory .}
set_property include_dirs [list $generated] [current_fileset]
set_property verilog_define {SODAP_LCS_GENERIC} [current_fileset]
read_verilog -sv [list $generated/operator_engines.v $generated/forward_kernel.sv $adapter]
synth_design -top sqr_bambu_basic_dma64 -part xc7vx485tffg1761-2 -mode out_of_context
create_clock -name clk -period 20.0 [get_ports clk]
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt
write_checkpoint synthesized.dcp
