// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Component test: generated IP + width adapter + backpressured 64-bit memory.
// This is not a full-system ESP simulation.
`timescale 1ns/1ps
module lcs_dma64_tb #(parameter SERIAL_DMA = 1, parameter WRITE_BUFFER_BEATS = 16);
  `include "parameters.svh"
  reg clk=0; always #5 clk=!clk;
  reg rst=0, start=0, hold_read=0;
  wire done, rv, rr, rdready, wv, wr, wdvalid;
  wire [31:0] ri,rl,wi,wl;
  wire [63:0] wd;
  reg [31:0] mem[0:WORDS-1], gold[0:WORDS-1];
  reg [3:0] mask[0:WORDS-1];
  integer raddr=0,rleft=0,waddr=0,wleft=0,cycles=0,i,errors=0,run_id;
  real error_value;
  wire rdvalid=rleft>0 && (hold_read || cycles%5!=0);
  wire wdready=wleft>0 && cycles%7!=0;
  // The native ESP socket services one read or write transaction at a time.
  // Independent channels hide an early write command that waits for data
  // whose computation still requires DMA reads.
  assign rr=rleft==0 && (!SERIAL_DMA || wleft==0) && cycles%7!=0;
  assign wr=wleft==0 && (!SERIAL_DMA || (rleft==0 && !rv)) && cycles%11!=0;
  sqr_bambu_basic_dma64 #(.CHECK_ARCHIVE_GEMM(0), .TIMEOUT_CYCLES(1000000),
                         .WRITE_BUFFER_BEATS(WRITE_BUFFER_BEATS)) dut(
    .clk(clk),.rst(rst),.conf_done(start),.acc_done(done),.debug(),
    .dma_read_ctrl_valid(rv),.dma_read_ctrl_ready(rr),.dma_read_ctrl_data_index(ri),.dma_read_ctrl_data_length(rl),
    .dma_read_ctrl_data_size(),.dma_read_ctrl_data_user(),
    .dma_read_chnl_valid(rdvalid),.dma_read_chnl_ready(rdready),.dma_read_chnl_data({mem[2*raddr+1],mem[2*raddr]}),
    .dma_write_ctrl_valid(wv),.dma_write_ctrl_ready(wr),.dma_write_ctrl_data_index(wi),.dma_write_ctrl_data_length(wl),
    .dma_write_ctrl_data_size(),.dma_write_ctrl_data_user(),
    .dma_write_chnl_valid(wdvalid),.dma_write_chnl_ready(wdready),.dma_write_chnl_data(wd));
  function real fp32(input [31:0] value);
    if (value[30:23] == 0)
      fp32 = (value[31] ? -1.0 : 1.0) * value[22:0] * (2.0 ** (-149));
    else
      fp32 = (value[31] ? -1.0 : 1.0) * (1.0 + value[22:0] / 8388608.0) *
             (2.0 ** (integer'(value[30:23]) - 127));
  endfunction
  always @(posedge clk) if(rst) begin
    cycles<=cycles+1;
    if(cycles>2000000) $fatal(1,"adapter timeout");
    hold_read <= rdvalid && !rdready;
    if(rv&&rr) begin
      if(rl==0 || 2*(ri+rl)>WORDS) $fatal(1,"read bounds");
      raddr<=ri; rleft<=rl;
    end
    if(rdvalid&&rdready) begin raddr<=raddr+1; rleft<=rleft-1; end
    if(wv&&wr) begin
      if(wl==0 || 2*(wi+wl)>WORDS) $fatal(1,"write bounds");
      waddr<=wi; wleft<=wl;
    end
    if(wdvalid&&wdready) begin
      mem[2*waddr]<=wd[31:0]; mem[2*waddr+1]<=wd[63:32];
      waddr<=waddr+1; wleft<=wleft-1;
    end
  end
  initial begin
    $readmemh("expected.hex",gold);
    $readmemh("mask.hex",mask);
    repeat(5) @(negedge clk); rst=1;
    for(run_id=0; run_id<2; run_id=run_id+1) begin
      $readmemh("initial.hex",mem);
      repeat(4) @(negedge clk);
      start=1; @(negedge clk); start=0;
      wait(done); @(negedge clk);
      if(rleft || wleft) $fatal(1,"done before DMA drained");
      $writememh($sformatf("actual-%0d.hex",run_id),mem);
      for(i=0;i<WORDS;i=i+1) begin
        if(mask[i] == 2) begin
          error_value = fp32(mem[i]) - fp32(gold[i]);
          if(error_value < 0) error_value = -error_value;
          if($isunknown(mem[i]) || mem[i][30:23]==255 || error_value>TOL) begin
            if(errors<8) $display("mismatch word=%0d got=%h expected=%h",i,mem[i],gold[i]);
            errors++;
          end
        end else if(mem[i] !== gold[i]) $fatal(1,"input/padding changed at %0d",i);
      end
      if(errors) $fatal(1,"numerical errors=%0d",errors);
      $display("LCS_DMA64 RESULT invocation=%0d errors=0",run_id);
      wait(!done);
    end
    $display("LCS_DMA64 PASS errors=0"); $finish;
  end
endmodule
