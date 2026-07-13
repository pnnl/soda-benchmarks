//===----------------------------------------------------------------------===//
//
// Part of the SODA Benchmarks Project
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

// Single-active-counter instrumentation IP.
//
// Unlike `sodaInstrHWCounters` (many concurrent start/stop counters), only
// one counter is ever running: location 0 counts cycles starting from
// reset, and every call to this IP just switches which location is
// currently accumulating cycles ("change location"). This avoids the
// separate start/stop state that `sodaInstrHWCounters` needs.
//
// Reporting: because only one counter runs at a time there is no natural
// "stop" event to print on, so this IP never prints per-call. Instead, the
// compiler emits one extra finalize call before each `func.return` with the
// report-sentinel location (all bits set, i.e. `location[63] == 1`); when
// that sentinel is observed, every location's final count is printed once
// via `$display`, avoiding the per-iteration log flood of naive counters.
module sodaInstrChangeLocation #(
    parameter LOC_WIDTH = 4,       // Number of bits for location tracking
    parameter COUNTER_WIDTH = 32   // Width of each counter
) (
    input wire              clock,
    input wire              reset,
    input wire              start_port,   // handshake start
    output reg              done_port,    // handshake done (2-cycle latency)
    input wire [63:0]       location,     // location identifier or report sentinel
    output reg [COUNTER_WIDTH-1:0] count  // current count for the active location
);

    // Bit 63 of `location` is reserved as the "print final report" sentinel;
    // it is never produced by real (small, incrementally-assigned) loop ids.
    wire report_trigger = location[63];
    wire [LOC_WIDTH-1:0] loc_idx = location[LOC_WIDTH-1:0];

    // One counter per location, but only `current_loc` accumulates at a time.
    reg [COUNTER_WIDTH-1:0] counters [0:(1<<LOC_WIDTH)-1];
    reg [LOC_WIDTH-1:0] current_loc;

    // Two-cycle latency for done_port, mirroring the shared IP template.
    reg done_port_reg;
    reg report_pending_reg;

    always @(posedge clock) begin
        if (!reset) begin
            done_port_reg <= 0;
        end else begin
            done_port_reg <= start_port;
        end
    end

    always @(posedge clock) begin
        if (!reset) begin
            report_pending_reg <= 0;
        end else begin
            report_pending_reg <= report_trigger && start_port;
        end
    end

    always @(posedge clock) begin
        integer k;
        if (!reset) begin
            done_port <= 0;
        end else begin
            done_port <= done_port_reg;
            if (done_port_reg && report_pending_reg) begin
                for (k = 0; k < (1<<LOC_WIDTH); k = k + 1) begin
                    $display("[HW] sodaInstrChangeLocation: FINAL location %0d count %d",
                             k, counters[k]);
                end
            end
        end
    end

    always @(posedge clock or negedge reset) begin
        integer i;
        if (!reset) begin
            for (i = 0; i < (1<<LOC_WIDTH); i = i + 1)
                counters[i] <= 0;
            current_loc <= 0;
            count <= 0;
        end else begin
            // Location 0 counts cycles from reset: whichever location is
            // currently active always accumulates, every cycle.
            counters[current_loc] <= counters[current_loc] + 1;
            // A non-report call switches the active location.
            if (start_port && !report_trigger)
                current_loc <= loc_idx;
            count <= counters[current_loc];
        end
    end

endmodule
