// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Component-level adapter test; the full-system test is run separately.
`timescale 1ns/1ps
module dma64_tb;
  reg clk=0; always #5 clk=!clk;
  reg rst=0, start=0;
  wire done, rv, rr, rdready, wv, wr, wdvalid;
  wire [31:0] ri,rl,wi,wl;
  wire [63:0] wd;
  reg [31:0] mem[0:513], gold[0:95];
  integer raddr=0,rleft=0,waddr=0,wleft=0,cycles=0,i,errors=0;
  wire rdvalid=rleft>0;
  assign rr=rleft==0 && rv;
  assign wr=wleft==0 && wv;
  sqr_bambu_basic_dma64 dut(
    .clk(clk),.rst(rst),.conf_done(start),.acc_done(done),.debug(),
    .dma_read_ctrl_valid(rv),.dma_read_ctrl_ready(rr),.dma_read_ctrl_data_index(ri),.dma_read_ctrl_data_length(rl),
    .dma_read_ctrl_data_size(),.dma_read_ctrl_data_user(),
    .dma_read_chnl_valid(rdvalid),.dma_read_chnl_ready(rdready),.dma_read_chnl_data({mem[2*raddr+1],mem[2*raddr]}),
    .dma_write_ctrl_valid(wv),.dma_write_ctrl_ready(wr),.dma_write_ctrl_data_index(wi),.dma_write_ctrl_data_length(wl),
    .dma_write_ctrl_data_size(),.dma_write_ctrl_data_user(),
    .dma_write_chnl_valid(wdvalid),.dma_write_chnl_ready(wleft>0),.dma_write_chnl_data(wd));
  always @(posedge clk) if(rst) begin
    cycles<=cycles+1;
    if(cycles>100000) $fatal(1,"adapter timeout read_valid=%b ready=%b active=%b",rv,rr,dut.read_active);
    if(rv&&rr) begin raddr<=ri; rleft<=rl; end
    if(rdvalid&&rdready) begin raddr<=raddr+1; rleft<=rleft-1; end
    if(wv&&wr) begin waddr<=wi; wleft<=wl; end
    if(wdvalid&&wleft>0) begin
      mem[2*waddr]<=wd[31:0]; mem[2*waddr+1]<=wd[63:32];
      waddr<=waddr+1; wleft<=wleft-1;
    end
  end
  string data_dir;
  initial begin
    if (!$value$plusargs("data_dir=%s",data_dir)) data_dir="../../../checks";
    $readmemh({data_dir,"/input_1.hex"},mem);
    $readmemh({data_dir,"/gold_1.hex"},gold);
    for(i=0;i<96;i++) if($isunknown(gold[i])) $fatal(1,"Missing golden fixture");
    repeat(5) @(negedge clk);
    rst=1;
    repeat(1000) @(negedge clk);
    start=1; @(negedge clk); start=0;
    wait(done); @(negedge clk);
    for(i=0;i<96;i++) if(mem[416+i]!==gold[i]) errors++;
    if(errors) $fatal(1,"adapter numerical errors=%0d",errors);
    @(posedge clk); #1;
    $display("GEMM_DMA64 PASS errors=0");
    $finish;
  end
endmodule
