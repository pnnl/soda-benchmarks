# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Use a board-specific hw_server. Never select the first of several devices.
if {$argc != 3} {error "Usage: program-board.tcl HOST PORT BITSTREAM"}
set host [lindex $argv 0]
set port [lindex $argv 1]
set bit [file normalize [lindex $argv 2]]
if {![file isfile $bit]} {error "Missing bitstream: $bit"}
if {[catch {
    open_hw_manager
    connect_hw_server -url $host:$port
    set targets [get_hw_targets]
    if {[llength $targets] != 1} {error "Expected one reserved-board JTAG target: $targets"}
    set target [lindex $targets 0]
    open_hw_target $target
    set devices [get_hw_devices]
    if {[llength $devices] != 1} {error "Expected one FPGA on reserved target: $devices"}
    set device [lindex $devices 0]
    set part [get_property PART $device]
    if {![string match -nocase "xc7vx485t*" $part]} {error "Unexpected FPGA: $part"}
    puts "LCS_FPGA_TARGET server=$host:$port target=$target device=$device part=$part"
    set_property PROGRAM.FILE $bit $device
    program_hw_devices $device
    refresh_hw_device $device
    set done unknown
    if {[lsearch -exact [list_property $device] REGISTER.IR.BIT5_DONE] >= 0} {
        set done [get_property REGISTER.IR.BIT5_DONE $device]
        if {$done ne "1"} {error "FPGA DONE is not asserted: $done"}
    }
    puts "LCS_FPGA_PROGRAMMED bit=$bit done=$done"
    close_hw_target
    disconnect_hw_server
    close_hw_manager
} message]} {
    puts stderr "LCS_FPGA_ERROR: $message"
    exit 1
}
exit 0
