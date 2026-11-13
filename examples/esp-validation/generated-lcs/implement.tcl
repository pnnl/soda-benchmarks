# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
if {$argc != 3} {error "Usage: implement.tcl PROJECT.xpr REPORTS JOBS"}
set project [file normalize [lindex $argv 0]]
set reports [file normalize [lindex $argv 1]]
set jobs [lindex $argv 2]
file mkdir $reports
set_param general.maxThreads $jobs
open_project $project
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    error "Synthesis did not complete: [get_property STATUS [get_runs synth_1]]"
}
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "Implementation did not complete: [get_property STATUS [get_runs impl_1]]"
}
open_run impl_1
report_timing_summary -max_paths 10 -report_unconstrained -file $reports/timing.rpt
report_utilization -file $reports/utilization.rpt
report_utilization -hierarchical -hierarchical_depth 12 -file $reports/hierarchy.rpt
report_clocks -file $reports/clocks.rpt
report_drc -file $reports/drc.rpt
set check [open $reports/check_timing.rpt w]
puts $check [check_timing -verbose -return_string]
close $check
set timing [open $reports/timing.tsv w]
puts $timing "metric\tvalue"
foreach name {STATS.WNS STATS.TNS STATS.WHS STATS.THS} {
    puts $timing "$name\t[get_property $name [get_runs impl_1]]"
}
close $timing
set clocks_out [open $reports/system_clock.tsv w]
puts $clocks_out "connection\tclock\tperiod_ns"
foreach {net expected} {chip_refclk 20.0 clkm 10.0} {
    set clocks [get_clocks -of_objects [get_nets $net]]
    if {[llength $clocks] != 1} {error "Expected one clock on $net"}
    set period [get_property PERIOD $clocks]
    puts $clocks_out "$net\t$clocks\t$period"
    if {abs($period - $expected) > 0.001} {error "Unexpected $net period: $period"}
}
set accelerator [get_cells -hierarchical -filter {REF_NAME == sqr_bambu_basic_dma64}]
if {[llength $accelerator] != 1} {error "Expected one accelerator wrapper"}
# Rebuilt hierarchy can remove the wrapper clock port. Check the clocks of
# retained registers instead. A literal prefix handles brackets in tile names.
set sequential {}
foreach cell [get_cells -hierarchical -filter {IS_SEQUENTIAL == 1}] {
    if {[string first "$accelerator/" $cell] == 0} {lappend sequential $cell}
}
if {![llength $sequential]} {error "No accelerator sequential cells found"}
set clocks [get_clocks -of_objects [get_pins -of_objects $sequential -filter {IS_CLOCK == 1}]]
if {[llength $clocks] != 1} {error "Expected one accelerator clock"}
set period [get_property PERIOD $clocks]
puts $clocks_out "accelerator_sequential_cells\t$clocks\t$period"
if {abs($period - 20.0) > 0.001} {error "Expected 50 MHz accelerator clock: $period ns"}
close $clocks_out
close_project
