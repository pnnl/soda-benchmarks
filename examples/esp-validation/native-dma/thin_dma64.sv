// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Native ESP interface adapter: 32/64-bit packing, request unit conversion,
// and lifecycle adaptation. No write-chunk payload buffer or admission policy.
`timescale 1ns/1ps
module sqr_bambu_basic_dma64 #(
  parameter CHECK_ARCHIVE_GEMM=0,
  parameter TIMEOUT_CYCLES=1000000
) (
  input clk, rst, conf_done,
  output reg acc_done,
  output [31:0] debug,
  output dma_read_ctrl_valid,
  output [31:0] dma_read_ctrl_data_index, dma_read_ctrl_data_length,
  output [2:0] dma_read_ctrl_data_size,
  output [5:0] dma_read_ctrl_data_user,
  input dma_read_ctrl_ready,
  input dma_read_chnl_valid,
  input [63:0] dma_read_chnl_data,
  output dma_read_chnl_ready,
  output dma_write_ctrl_valid,
  output [31:0] dma_write_ctrl_data_index, dma_write_ctrl_data_length,
  output [2:0] dma_write_ctrl_data_size,
  output [5:0] dma_write_ctrl_data_user,
  input dma_write_ctrl_ready,
  output dma_write_chnl_valid,
  output [63:0] dma_write_chnl_data,
  input dma_write_chnl_ready
);
  wire [63:0] read_request,write_request;
  wire read_request_valid,read_request_ready,write_request_valid,write_request_ready;
  wire core_done,core_read_ready,core_write_valid,core_write_ready;
  wire [31:0] core_write_data;
  reg core_start,done_pending;
  reg read_active,read_full,read_half;
  reg [31:0] read_words_left;
  reg [63:0] read_buffer;
  reg write_active,write_full,write_half;
  reg [31:0] write_beats_left;
  reg [63:0] write_buffer;

  // Keep the original startup alignment with the legacy ESP DMA CONFIG state.
  always @(posedge clk) if(!rst) core_start<=0; else core_start<=conf_done;
  forward_kernel u_core (
    .clock(clk),.reset(rst),.start_port(core_start),.done_port(core_done),.idle_port(),
    .P0_TDATA(read_request),.P0_TVALID(read_request_valid),.P0_TREADY(read_request_ready),
    .P1_TDATA(read_half?read_buffer[63:32]:read_buffer[31:0]),
    .P1_TVALID(read_full),.P1_TREADY(core_read_ready),
    .P2_TDATA(write_request),.P2_TVALID(write_request_valid),.P2_TREADY(write_request_ready),
    .P3_TDATA(core_write_data),.P3_TVALID(core_write_valid),.P3_TREADY(core_write_ready));

  wire [32:0] rounded_read_words={1'b0,read_request[63:32]}+{32'b0,read_request[0]}+33'd1;
  assign dma_read_ctrl_valid=read_request_valid&&!read_active;
  assign read_request_ready=dma_read_ctrl_ready&&!read_active;
  assign dma_read_ctrl_data_index=read_request[31:0]>>1;
  assign dma_read_ctrl_data_length=rounded_read_words[32:1];
  assign dma_read_ctrl_data_size=3'd3;
  assign dma_read_ctrl_data_user=0;
  assign dma_read_chnl_ready=read_active&&!read_full;

  wire allow_write_request;
  // The test-only hold matches the original adapter. No gate in synthesis.
`ifdef SYNTHESIS
  assign allow_write_request=1'b1;
`else
  integer stall_until=0,cycles=0;
  initial if(!$value$plusargs("gemm_stall=%d",stall_until)) stall_until=0;
  always @(posedge clk) if(!rst||conf_done) cycles<=0; else cycles<=cycles+1;
  assign allow_write_request=cycles>=stall_until;
`endif
  assign dma_write_ctrl_valid=write_request_valid&&!write_active&&allow_write_request;
  assign write_request_ready=dma_write_ctrl_ready&&!write_active&&allow_write_request;
  assign dma_write_ctrl_data_index=write_request[31:0]>>1;
  assign dma_write_ctrl_data_length=write_request[63:32]>>1;
  assign dma_write_ctrl_data_size=3'd3;
  assign dma_write_ctrl_data_user=0;
  assign core_write_ready=write_active&&!write_full;
  assign dma_write_chnl_valid=write_full;
  assign dma_write_chnl_data=write_buffer;
  assign debug={26'b0,done_pending,write_active,write_full,write_half,read_active,read_full};

  always @(posedge clk) begin
    if(!rst) begin
      read_active<=0;read_full<=0;read_half<=0;read_words_left<=0;read_buffer<=0;
      write_active<=0;write_full<=0;write_half<=0;write_beats_left<=0;write_buffer<=0;
      done_pending<=0;acc_done<=0;
    end else begin
      acc_done<=0;
      if(dma_read_ctrl_valid&&dma_read_ctrl_ready) begin
        read_active<=1;read_words_left<=read_request[63:32];read_half<=read_request[0];
      end
      if(dma_read_chnl_valid&&dma_read_chnl_ready) begin read_buffer<=dma_read_chnl_data;read_full<=1;end
      if(read_full&&core_read_ready) begin
        read_words_left<=read_words_left-1;
        if(read_words_left==1) begin read_active<=0;read_full<=0;read_half<=0;end
        else if(read_half) begin read_full<=0;read_half<=0;end
        else read_half<=1;
      end
      if(dma_write_ctrl_valid&&dma_write_ctrl_ready) begin
        write_active<=1;write_beats_left<=write_request[63:32]>>1;
      end
      if(core_write_valid&&core_write_ready) begin
        if(!write_half) begin write_buffer[31:0]<=core_write_data;write_half<=1;end
        else begin write_buffer[63:32]<=core_write_data;write_half<=0;write_full<=1;end
      end
      if(dma_write_chnl_valid&&dma_write_chnl_ready) begin
        write_full<=0;write_beats_left<=write_beats_left-1;
        if(write_beats_left==1) write_active<=0;
      end
      if(core_done) done_pending<=1;
      if(done_pending&&!write_active&&!write_full&&!write_half&&!read_active) begin
        acc_done<=1;done_pending<=0;
      end
    end
  end

  // Passive checks only. These never supply data or alter the synthesized IP.
  // synthesis translate_off
  integer invocation=0,nr=0,nw=0,read_beats=0,write_beats=0;
  reg monitoring_run=0,rr_held=0,wr_held=0,wd_held=0;
  reg [63:0] rr_previous,wr_previous,wd_previous;
  always @(posedge clk) begin
    if(conf_done) begin
      invocation=invocation+1;monitoring_run=1;nr=0;nw=0;read_beats=0;write_beats=0;
      $display("THIN_BOUNDARY START invocation=%0d stall_until=%0d",invocation,stall_until);
    end
    if(rst&&invocation>0) begin
      if(monitoring_run&&cycles==TIMEOUT_CYCLES) $fatal(1,"THIN hardware timeout");
      if(rr_held&&(!dma_read_ctrl_valid||{dma_read_ctrl_data_length,dma_read_ctrl_data_index}!==rr_previous)) $fatal(1,"read request changed while stalled");
      if(wr_held&&(!dma_write_ctrl_valid||{dma_write_ctrl_data_length,dma_write_ctrl_data_index}!==wr_previous)) $fatal(1,"write request changed while stalled");
      if(wd_held&&(!dma_write_chnl_valid||dma_write_chnl_data!==wd_previous)) $fatal(1,"write data changed while stalled");
      rr_held=dma_read_ctrl_valid&&!dma_read_ctrl_ready;wr_held=dma_write_ctrl_valid&&!dma_write_ctrl_ready;
      wd_held=dma_write_chnl_valid&&!dma_write_chnl_ready;
      rr_previous={dma_read_ctrl_data_length,dma_read_ctrl_data_index};
      wr_previous={dma_write_ctrl_data_length,dma_write_ctrl_data_index};wd_previous=dma_write_chnl_data;
      if(dma_read_ctrl_valid&&dma_read_ctrl_ready) begin
        nr++;if(!read_request[63:32]) $fatal(1,"empty read");
      end
      if(dma_write_ctrl_valid&&dma_write_ctrl_ready) begin
        nw++;if(write_request[0]||write_request[32]||!write_request[63:32]) $fatal(1,"unaligned or empty write");
      end
      if(dma_read_chnl_valid&&dma_read_chnl_ready) read_beats++;
      if(dma_write_chnl_valid&&dma_write_chnl_ready) write_beats++;
      if(acc_done) begin
        monitoring_run=0;
        $display("THIN_BOUNDARY RESULT invocation=%0d requests=%0d/%0d beats=%0d/%0d cycles=%0d",invocation,nr,nw,read_beats,write_beats,cycles);
      end
    end
  end
  // synthesis translate_on
endmodule
