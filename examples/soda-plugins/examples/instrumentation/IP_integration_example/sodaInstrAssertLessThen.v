//===----------------------------------------------------------------------===//
//
// Part of the SODA Benchmarks Project
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//


module sodaInstrAssertLessThen
  (input wire        clock,
   input wire        reset,

   input wire        start_port,
   output reg        done_port,

   input wire [63:0] it,
   input wire [63:0] max);

   reg        done_port_reg;
   wire       result;

   assign result = (it < max);

   //----------------------------------------------------------------
   // Simulate processing on input
   //----------------------------------------------------------------

   always @(posedge clock) begin
      if (!reset) begin
         done_port_reg <= 0;
      end
      else begin
         done_port_reg <= start_port;
      end
   end

   //----------------------------------------------------------------
   // Outputs, two cycle latency
   //----------------------------------------------------------------

   always @(posedge clock) begin
      if (!reset) begin
         done_port <= 0;
      end
      else begin
         done_port <= done_port_reg;
         if (done_port_reg) begin
            $display("sodaInstrAssertLessThen: %h < %h ? %s", it, max, result ? "true" : "false");
         end
      end
   end

endmodule
