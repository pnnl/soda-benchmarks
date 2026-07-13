//===----------------------------------------------------------------------===//
//
// Part of the SODA Benchmarks Project
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

// Fixed-size vector dot-product HW module.
//
// Unlike the other instrumentation IPs (assertions/counters), this module
// does not just observe the kernel -- it *replaces* a matched
// `linalg.dot` reduction outright (see `sodap-swap-op-to-hw`), reading both
// operand vectors from memory and writing the reduced scalar result back to
// memory. It is modeled on the `IP_integration_example/module1.v`
// memory-master pattern: a small FSM drives the shared RAM bus directly to
// read operands, and reuses the `__builtin_memstore` block to write the
// result, exactly like `module1`/`module2` do for their outputs.
//
// Simplification: the accumulate datapath below is a plain integer
// multiply-accumulate over the raw memory words (`Mout_Wdata_ram`/
// `M_Rdata_ram` bit patterns), standing in for a real floating-point MAC.
// A production IP would replace the multiply/add in `S_ACCUM` with a
// floating-point core; the surrounding handshake/memory-master/report
// structure would stay the same. This module has been lint-checked
// (`verilator --lint-only`) but not run through Bambu HLS/Verilator
// end-to-end.
`timescale 1ns / 1ps
module sodaVectorDot
  (clock, reset, start_port, done_port,
   a_addr, b_addr, len, out_addr,
   Min_oe_ram, Mout_oe_ram, Min_we_ram, Mout_we_ram,
   Min_addr_ram, Mout_addr_ram, M_Rdata_ram,
   Min_Wdata_ram, Mout_Wdata_ram,
   Min_data_ram_size, Mout_data_ram_size, M_DataRdy);

  parameter BITSIZE_a_addr = 32,
            BITSIZE_b_addr = 32,
            BITSIZE_out_addr = 32,
            BITSIZE_Min_addr_ram = 32,
            BITSIZE_Mout_addr_ram = 32,
            BITSIZE_M_Rdata_ram = 32,
            BITSIZE_Min_Wdata_ram = 32,
            BITSIZE_Mout_Wdata_ram = 32,
            BITSIZE_Min_data_ram_size = 6,
            BITSIZE_Mout_data_ram_size = 6;

  // IN
  input clock;
  input reset;
  input start_port;
  input [BITSIZE_a_addr-1:0] a_addr;
  input [BITSIZE_b_addr-1:0] b_addr;
  input [31:0] len;                    // number of f32 elements
  input [BITSIZE_out_addr-1:0] out_addr;
  input Min_oe_ram;
  input Min_we_ram;
  input [BITSIZE_Min_addr_ram-1:0] Min_addr_ram;
  input [BITSIZE_M_Rdata_ram-1:0] M_Rdata_ram;
  input [BITSIZE_Min_Wdata_ram-1:0] Min_Wdata_ram;
  input [BITSIZE_Min_data_ram_size-1:0] Min_data_ram_size;
  input M_DataRdy;

  // OUT
  output reg done_port;
  output reg Mout_oe_ram;
  output reg Mout_we_ram;
  output reg [BITSIZE_Mout_addr_ram-1:0] Mout_addr_ram;
  output reg [BITSIZE_Mout_Wdata_ram-1:0] Mout_Wdata_ram;
  output reg [BITSIZE_Mout_data_ram_size-1:0] Mout_data_ram_size;

  localparam ELEM_BYTES = 4; // sizeof(float)

  // -------------------------------------------------------------------
  // Result-store sub-FSM, reusing __builtin_memstore exactly like
  // IP_integration_example/module1.v does for its outputs.
  // -------------------------------------------------------------------
  reg        start_port_memstore;
  wire       done_port_memstore;
  reg [BITSIZE_Mout_Wdata_ram-1:0] store_data_int;
  reg [BITSIZE_out_addr-1:0]       store_addr_int;
  reg [6:0]                        store_size_int;
  reg        Min_oe_ram_int;
  reg        Min_we_ram_int;
  reg [BITSIZE_Min_addr_ram-1:0]   Min_addr_ram_int;
  reg [BITSIZE_Min_Wdata_ram-1:0]  Min_Wdata_ram_int;
  reg [BITSIZE_Min_data_ram_size-1:0] Min_data_ram_size_int;

  __builtin_memstore #(
      .BITSIZE_data(BITSIZE_Mout_Wdata_ram), .BITSIZE_addr(BITSIZE_out_addr),
      .BITSIZE_size(7), .BITSIZE_Min_addr_ram(BITSIZE_Min_addr_ram),
      .BITSIZE_Mout_addr_ram(BITSIZE_Mout_addr_ram),
      .BITSIZE_M_Rdata_ram(BITSIZE_M_Rdata_ram),
      .BITSIZE_Min_Wdata_ram(BITSIZE_Min_Wdata_ram),
      .BITSIZE_Mout_Wdata_ram(BITSIZE_Mout_Wdata_ram),
      .BITSIZE_Min_data_ram_size(BITSIZE_Min_data_ram_size),
      .BITSIZE_Mout_data_ram_size(BITSIZE_Mout_data_ram_size)
  ) my_result_store (
      .clock(clock), .reset(reset),
      .start_port(start_port_memstore), .data(store_data_int),
      .addr(store_addr_int), .size(store_size_int),
      .done_port(done_port_memstore),
      .Min_oe_ram(Min_oe_ram_int), .Mout_oe_ram(Mout_oe_ram),
      .Min_we_ram(Min_we_ram_int), .Mout_we_ram(Mout_we_ram),
      .Min_addr_ram(Min_addr_ram_int), .Mout_addr_ram(Mout_addr_ram),
      .M_Rdata_ram(M_Rdata_ram), .Min_Wdata_ram(Min_Wdata_ram_int),
      .Mout_Wdata_ram(Mout_Wdata_ram), .Min_data_ram_size(Min_data_ram_size_int),
      .Mout_data_ram_size(Mout_data_ram_size), .M_DataRdy(M_DataRdy));

  // -------------------------------------------------------------------
  // Main FSM: sequentially read a[i]/b[i] directly off the shared RAM
  // bus (simplified single-request/wait-for-DataRdy protocol), multiply-
  // accumulate, then hand off to the store sub-FSM for the final write.
  // -------------------------------------------------------------------
  parameter [3:0]
    S_IDLE       = 4'd0,
    S_REQ_A      = 4'd1,
    S_WAIT_A     = 4'd2,
    S_REQ_B      = 4'd3,
    S_WAIT_B     = 4'd4,
    S_ACCUM      = 4'd5,
    S_STORE      = 4'd6,
    S_STORE_WAIT = 4'd7,
    S_DONE       = 4'd8;

  reg [3:0] state, next_state;
  reg [31:0] idx;
  reg [BITSIZE_Mout_Wdata_ram-1:0] acc;
  reg [BITSIZE_Mout_Wdata_ram-1:0] a_val;

  always @(posedge clock or negedge reset)
    if (!reset) state <= S_IDLE;
    else        state <= next_state;

  always @(posedge clock or negedge reset) begin
    if (!reset) begin
      idx <= 0;
      acc <= 0;
      a_val <= 0;
    end else begin
      case (state)
        S_IDLE: begin
          idx <= 0;
          acc <= 0;
        end
        S_WAIT_A: if (M_DataRdy) a_val <= M_Rdata_ram;
        S_ACCUM: begin
          // Simplified integer MAC standing in for a floating-point core.
          acc <= acc + (a_val * M_Rdata_ram);
          idx <= idx + 1;
        end
        default: ;
      endcase
    end
  end

  always @(*) begin
    next_state = state;
    Mout_oe_ram = 1'b0;
    Mout_we_ram = 1'b0;
    Mout_addr_ram = {BITSIZE_Mout_addr_ram{1'b0}};
    Min_oe_ram_int = Min_oe_ram;
    Min_we_ram_int = Min_we_ram;
    Min_addr_ram_int = Min_addr_ram;
    Min_Wdata_ram_int = Min_Wdata_ram;
    Min_data_ram_size_int = Min_data_ram_size;
    start_port_memstore = 1'b0;
    store_data_int = acc;
    store_addr_int = out_addr;
    store_size_int = 32;
    done_port = 1'b0;

    case (state)
      S_IDLE: begin
        if (start_port)
          next_state = (len == 0) ? S_STORE : S_REQ_A;
      end
      S_REQ_A: begin
        Mout_oe_ram = 1'b1;
        Mout_addr_ram = a_addr + idx * ELEM_BYTES;
        next_state = S_WAIT_A;
      end
      S_WAIT_A: begin
        Mout_oe_ram = 1'b1;
        Mout_addr_ram = a_addr + idx * ELEM_BYTES;
        next_state = M_DataRdy ? S_REQ_B : S_WAIT_A;
      end
      S_REQ_B: begin
        Mout_oe_ram = 1'b1;
        Mout_addr_ram = b_addr + idx * ELEM_BYTES;
        next_state = S_WAIT_B;
      end
      S_WAIT_B: begin
        Mout_oe_ram = 1'b1;
        Mout_addr_ram = b_addr + idx * ELEM_BYTES;
        next_state = M_DataRdy ? S_ACCUM : S_WAIT_B;
      end
      S_ACCUM: begin
        next_state = ((idx + 1) >= len) ? S_STORE : S_REQ_A;
      end
      S_STORE: begin
        start_port_memstore = 1'b1;
        next_state = S_STORE_WAIT;
      end
      S_STORE_WAIT: begin
        start_port_memstore = 1'b1;
        next_state = done_port_memstore ? S_DONE : S_STORE_WAIT;
      end
      S_DONE: begin
        done_port = 1'b1;
        next_state = S_IDLE;
      end
      default: next_state = S_IDLE;
    endcase
  end

endmodule
