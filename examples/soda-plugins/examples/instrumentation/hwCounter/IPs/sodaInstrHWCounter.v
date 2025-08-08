//===----------------------------------------------------------------------===//
//
// Part of the SODA Benchmarks Project
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//


module sodaInstrHWCounter #(
    parameter LOC_WIDTH = 8,      // Number of bits for location tracking
    parameter COUNTER_WIDTH = 4   // Width of each counter
) (
    input wire              clock,
    input wire              reset,
    input wire              start_port,   // handshake start
    output reg              done_port,    // handshake done (2-cycle latency)
    input wire              action,       // true=start, false=stop (per-location)
    input wire [63:0]       location,     // location identifier
    output reg [COUNTER_WIDTH-1:0] count  // current count for selected location
);

    // Truncate location using a simple hash (XOR folding)
    wire [LOC_WIDTH-1:0] loc_idx;
    assign loc_idx = ^location[63:LOC_WIDTH] ^ location[LOC_WIDTH-1:0];

    // Counter array for each location
    reg [COUNTER_WIDTH-1:0] counters [0:(1<<LOC_WIDTH)-1];
    reg running [0:(1<<LOC_WIDTH)-1];

    // Two-cycle latency for done_port
    reg done_port_reg;

    always @(posedge clock) begin
        if (!reset) begin
            done_port_reg <= 0;
        end else begin
            done_port_reg <= start_port;
        end
    end

    always @(posedge clock) begin
        if (!reset) begin
            done_port <= 0;
        end else begin
            done_port <= done_port_reg;
            if (done_port_reg) begin
                $display("sodaInstrHWCounter: location %h count %d", location, counters[loc_idx]);
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
        end else begin
            // Update running state for this location
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
