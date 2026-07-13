//===----------------------------------------------------------------------===//
//
// Part of the SODA Benchmarks Project
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//


module sodaInstrHWCounters #(
    parameter LOC_WIDTH = 4,       // Number of bits for location tracking
    parameter COUNTER_WIDTH = 32,  // Width of each counter
    parameter PRINT_ON_STOP = 1    // 1 = print each location on every stop (legacy,
                                    // floods the sim log); 0 = only print via
                                    // the report-sentinel finalize call below.
) (
    input wire              clock,
    input wire              reset,
    input wire              start_port,   // handshake start
    output reg              done_port,    // handshake done (2-cycle latency)
    input wire              action,       // true=start, false=stop (per-location)
    input wire [63:0]       location,     // location identifier, or report sentinel
    output reg [COUNTER_WIDTH-1:0] count  // current count for selected location
);

    // Bit 63 of `location` is reserved as the "print final report" sentinel
    // (all bits set, i.e. `location == -1` from the compiler side); it is
    // never produced by real (small, incrementally-assigned) loop ids. When
    // asserted, every location's final count is printed once via `$display`,
    // instead of the per-stop print below -- this avoids the simulation log
    // flood from printing on every loop iteration/stop.
    wire report_trigger = location[63];

    // Truncate location to LOC_WIDTH bits (simple truncation)
    wire [LOC_WIDTH-1:0] loc_idx;
    assign loc_idx = location[LOC_WIDTH-1:0];

    // Counter array for each location
    reg [COUNTER_WIDTH-1:0] counters [0:(1<<LOC_WIDTH)-1];
    reg running [0:(1<<LOC_WIDTH)-1];

    // Two-cycle latency for done_port
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
            if (done_port_reg) begin
                if (report_pending_reg) begin
                    // Print-at-simulation-end: report every location once.
                    for (k = 0; k < (1<<LOC_WIDTH); k = k + 1) begin
                        $display("[HW] sodaInstrHWCounters: FINAL location %0d count %d",
                                 k, counters[k]);
                    end
                end else if (PRINT_ON_STOP) begin
                    // Legacy behavior (default): print on every stop.
                    $display("[HW] sodaInstrHWCounters: location %h count %d", location, counters[loc_idx]);
                end
            end
        end
    end

    always @(posedge clock or negedge reset) begin
        integer i;
        integer j;
        if (!reset) begin
            for (i = 0; i < (1<<LOC_WIDTH); i = i + 1) begin
                counters[i] <= 0;
                running[i] <= 0;
            end
            count <= 0;
        end else if (!report_trigger) begin
            // Update running state for this location (report calls are not
            // real locations and must not perturb counter state).
            running[loc_idx] <= action;

            // Increment counter for locations that are running and start_port is asserted
            for (j = 0; j < (1<<LOC_WIDTH); j = j + 1) begin
                if (running[j] && start_port)
                    counters[j] <= counters[j] + 1;
            end

            count <= counters[loc_idx];
        end
    end

endmodule
