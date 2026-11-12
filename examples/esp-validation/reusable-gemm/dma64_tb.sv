// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Replay descriptors captured from the generated host against one unchanged
// synthesized core. Only the descriptor changes between launches: intermediate
// tensor data must come from the preceding hardware invocation.
`timescale 1ns/1ps
module reusable_gemm_tb #(parameter TIMEOUT_CYCLES=2000000, parameter real TOL=0.001);
  `include "parameters.svh"
  reg clk=0; always #5 clk=!clk;
  reg rst=0,start=0,hold_read=0;
  wire done,rv,rr,rdready,wv,wr,wdvalid;
  wire [31:0] ri,rl,wi,wl;
  wire [63:0] wd;
  reg [31:0] mem[0:WORDS-1],gold[0:WORDS-1],next_input[0:WORDS-1],before_mem[0:WORDS-1];
  reg [3:0] mask[0:WORDS-1];
  integer raddr=0,rleft=0,waddr=0,wleft=0,cycles=0,i,errors=0,invocation;
  real difference;
  wire rdvalid=rleft>0 && (hold_read || cycles%5!=0);
  wire wdready=wleft>0 && cycles%7!=0;
  assign rr=rleft==0 && wleft==0 && cycles%7!=0;
  assign wr=wleft==0 && rleft==0 && !rv && cycles%11!=0;
  sqr_bambu_basic_dma64 #(.CHECK_ARCHIVE_GEMM(0),.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) dut(
    .clk(clk),.rst(rst),.conf_done(start),.acc_done(done),.debug(),
    .dma_read_ctrl_valid(rv),.dma_read_ctrl_ready(rr),.dma_read_ctrl_data_index(ri),.dma_read_ctrl_data_length(rl),
    .dma_read_ctrl_data_size(),.dma_read_ctrl_data_user(),
    .dma_read_chnl_valid(rdvalid),.dma_read_chnl_ready(rdready),.dma_read_chnl_data({mem[2*raddr+1],mem[2*raddr]}),
    .dma_write_ctrl_valid(wv),.dma_write_ctrl_ready(wr),.dma_write_ctrl_data_index(wi),.dma_write_ctrl_data_length(wl),
    .dma_write_ctrl_data_size(),.dma_write_ctrl_data_user(),
    .dma_write_chnl_valid(wdvalid),.dma_write_chnl_ready(wdready),.dma_write_chnl_data(wd));
  function real fp32(input [31:0] value);
    if (value[30:23]==0)
      fp32=(value[31] ? -1.0 : 1.0)*value[22:0]*(2.0**(-149));
    else
      fp32=(value[31] ? -1.0 : 1.0)*(1.0+value[22:0]/8388608.0)*
           (2.0**(integer'(value[30:23])-127));
  endfunction
  always @(posedge clk) if(rst) begin
    cycles<=cycles+1;
    if(cycles>INVOCATIONS*TIMEOUT_CYCLES+1000) $fatal(1,"adapter timeout");
    hold_read<=rdvalid&&!rdready;
    if($test$plusargs("trace_config") && cycles<120 && dut.read_full && dut.core_read_ready)
      $display("CONFIG_READ cycle=%0d half=%0d data=%h",cycles,dut.read_half,
               dut.read_half ? dut.read_buffer[63:32] : dut.read_buffer[31:0]);
    if(rv&&rr) begin
      if(rl==0 || 2*(ri+rl)>WORDS) $fatal(1,"read bounds");
      raddr<=ri;rleft<=rl;
    end
    if(rdvalid&&rdready) begin raddr<=raddr+1;rleft<=rleft-1;end
    if(wv&&wr) begin
      if(wl==0 || 2*(wi+wl)>WORDS) $fatal(1,"write bounds");
      waddr<=wi;wleft<=wl;
    end
    if(wdvalid&&wdready) begin
      mem[2*waddr]<=wd[31:0];mem[2*waddr+1]<=wd[63:32];
      waddr<=waddr+1;wleft<=wleft-1;
    end
  end
  initial begin
    $readmemh("initial-0.hex",mem);
    repeat(5) @(negedge clk);rst=1;
    for(invocation=0;invocation<INVOCATIONS;invocation=invocation+1) begin
      $readmemh($sformatf("initial-%0d.hex",invocation),next_input);
      $readmemh($sformatf("expected-%0d.hex",invocation),gold);
      $readmemh($sformatf("mask-%0d.hex",invocation),mask);
      // H remains exactly where the first hardware launch wrote it.
      for(i=0;i<HEADER_WORDS;i=i+1) mem[i]=next_input[i];
      for(i=0;i<WORDS;i=i+1) before_mem[i]=mem[i];
      repeat(4) @(negedge clk);
      start=1;@(negedge clk);start=0;
      wait(done);@(negedge clk);
      if(rleft||wleft) $fatal(1,"done before DMA completion");
      for(i=0;i<WORDS;i=i+1) begin
        if(mask[i]==2) begin
          difference=fp32(mem[i])-fp32(gold[i]);
          if(difference<0) difference=-difference;
          if($isunknown(mem[i]) || mem[i][30:23]==255 || difference>TOL) begin
            if(errors<8) $display("mismatch word=%0d got=%h expected=%h",i,mem[i],gold[i]);
            errors++;
          end
        end else if(mem[i]!==before_mem[i])
          $fatal(1,"input/header/padding changed at word %0d",i);
      end
      if(errors) $fatal(1,"numerical errors=%0d",errors);
      $display("REUSABLE_GEMM_RTL invocation=%0d errors=0 cycles=%0d",invocation,cycles);
      wait(!done);
    end
    $display("REUSABLE_GEMM_RTL PASS invocations=%0d errors=0",INVOCATIONS);$finish;
  end
endmodule
