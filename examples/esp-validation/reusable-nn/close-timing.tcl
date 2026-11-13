# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Refine an existing routed implementation without changing RTL or clocks.
if {$argc != 2} {error "Usage: close-timing.tcl ROUTED.dcp FRESH-OUTPUT"}
set checkpoint [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
if {[file exists $out]} {error "Use a fresh output directory: $out"}
file mkdir $out/reports
set_param general.maxThreads 4
open_checkpoint $checkpoint
phys_opt_design -directive AggressiveExplore
route_design -directive Explore
report_timing_summary -max_paths 10 -report_unconstrained -file $out/reports/timing.rpt
report_utilization -file $out/reports/utilization.rpt
report_utilization -hierarchical -hierarchical_depth 12 -file $out/reports/hierarchy.rpt
report_clocks -file $out/reports/clocks.rpt
report_drc -file $out/reports/drc.rpt
set setup [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
set report [open $out/reports/timing.tsv w]
puts $report "metric\tvalue"
puts $report "STATS.WNS\t$setup"
puts $report "STATS.WHS\t$hold"
close $report
set clocks [open $out/reports/system_clock.tsv w]
puts $clocks "connection\tclock\tperiod_ns"
foreach {net expected} {chip_refclk 20.0 clkm 10.0} {
    set clock [get_clocks -of_objects [get_nets $net]]
    if {[llength $clock] != 1} {error "Expected one clock on $net"}
    set period [get_property PERIOD $clock]
    puts $clocks "$net\t$clock\t$period"
    if {abs($period-$expected)>0.001} {error "Unexpected period on $net"}
}
close $clocks
write_checkpoint $out/top_routed.dcp
if {$setup<0 || $hold<0} {error "Timing still fails: setup=$setup ns hold=$hold ns"}
write_bitstream $out/top.bit
puts "NN_FPGA_TIMING_PASS setup=$setup hold=$hold bit=$out/top.bit"
