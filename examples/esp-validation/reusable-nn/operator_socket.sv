// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// One instance of each reusable engine behind the existing four DMA streams.
// The socket reads the magic slot, then the selected engine rereads/latches its
// complete descriptor. All engines retain state across calls and reinitialize
// their local working data at each launch. Arbitration is sequential.
module forward_kernel (
  input clock, reset, start_port,
  output reg done_port,
  output idle_port,
  output [63:0] P0_TDATA, output P0_TVALID, input P0_TREADY,
  input [31:0] P1_TDATA, input P1_TVALID, output P1_TREADY,
  output [63:0] P2_TDATA, output P2_TVALID, input P2_TREADY,
  output [31:0] P3_TDATA, output P3_TVALID, input P3_TREADY
);
  `include "operator_abi.svh"
  localparam IDLE=0, REQUEST=1, MAGIC=2, PAD=3, START=4, RUN=5;
  reg [2:0] state;
  reg [1:0] selected;
  wire [2:0] starts, dones, read_valids, read_readies, write_valids, data_valids;
  wire [63:0] read_commands[0:2], write_commands[0:2];
  wire [31:0] write_data[0:2];
  wire [2:0] active;
  genvar i;
  generate for(i=0;i<3;i=i+1) begin
    assign active[i] = (state==START || state==RUN) && selected==i;
    assign starts[i] = state==START && selected==i;
  end endgenerate
  assign idle_port = state==IDLE;
  assign P0_TDATA = state==REQUEST ? {32'd2,32'd0} : read_commands[selected];
  assign P0_TVALID = state==REQUEST || ((state==START || state==RUN) && read_valids[selected]);
  assign P1_TREADY = (state==MAGIC || state==PAD) || ((state==START || state==RUN) && read_readies[selected]);
  assign P2_TDATA = write_commands[selected];
  assign P2_TVALID = (state==START || state==RUN) && write_valids[selected];
  assign P3_TDATA = write_data[selected];
  assign P3_TVALID = (state==START || state==RUN) && data_valids[selected];

  conv_kernel u_conv (
    .clock(clock),.reset(reset),.start_port(starts[0]),.done_port(dones[0]),.idle_port(),
    .P0_TDATA(read_commands[0]),.P0_TVALID(read_valids[0]),.P0_TREADY(P0_TREADY && active[0]),
    .P1_TDATA(P1_TDATA),.P1_TVALID(P1_TVALID && active[0]),.P1_TREADY(read_readies[0]),
    .P2_TDATA(write_commands[0]),.P2_TVALID(write_valids[0]),.P2_TREADY(P2_TREADY && active[0]),
    .P3_TDATA(write_data[0]),.P3_TVALID(data_valids[0]),.P3_TREADY(P3_TREADY && active[0]));
  add_kernel u_add (
    .clock(clock),.reset(reset),.start_port(starts[1]),.done_port(dones[1]),.idle_port(),
    .P0_TDATA(read_commands[1]),.P0_TVALID(read_valids[1]),.P0_TREADY(P0_TREADY && active[1]),
    .P1_TDATA(P1_TDATA),.P1_TVALID(P1_TVALID && active[1]),.P1_TREADY(read_readies[1]),
    .P2_TDATA(write_commands[1]),.P2_TVALID(write_valids[1]),.P2_TREADY(P2_TREADY && active[1]),
    .P3_TDATA(write_data[1]),.P3_TVALID(data_valids[1]),.P3_TREADY(P3_TREADY && active[1]));
  gemm_kernel u_gemm (
    .clock(clock),.reset(reset),.start_port(starts[2]),.done_port(dones[2]),.idle_port(),
    .P0_TDATA(read_commands[2]),.P0_TVALID(read_valids[2]),.P0_TREADY(P0_TREADY && active[2]),
    .P1_TDATA(P1_TDATA),.P1_TVALID(P1_TVALID && active[2]),.P1_TREADY(read_readies[2]),
    .P2_TDATA(write_commands[2]),.P2_TVALID(write_valids[2]),.P2_TREADY(P2_TREADY && active[2]),
    .P3_TDATA(write_data[2]),.P3_TVALID(data_valids[2]),.P3_TREADY(P3_TREADY && active[2]));

  always @(posedge clock) begin
    if(!reset) begin state<=IDLE; selected<=0; done_port<=0; end
    else begin
      done_port<=0;
      case(state)
        IDLE: if(start_port) state<=REQUEST;
        REQUEST: if(P0_TREADY) state<=MAGIC;
        MAGIC: if(P1_TVALID) begin
          if(P1_TDATA==OP_CONV_MAGIC) selected<=0;
          else if(P1_TDATA==OP_ADD_MAGIC) selected<=1;
          else if(P1_TDATA==GEMM_MAGIC) selected<=2;
          // synthesis translate_off
          else $fatal(1,"Unknown reusable operator descriptor magic");
          // synthesis translate_on
          state<=PAD;
        end
        PAD: if(P1_TVALID) state<=START;
        START: state<=RUN;
        RUN: if(dones[selected]) begin done_port<=1; state<=IDLE; end
        default: state<=IDLE;
      endcase
    end
  end
endmodule
