// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Native CPU / real generated RTL co-simulation, using ESP's DMA64 adapter.
#include "Voperator.h"
#include "verilated.h"
#include "sodap/ExecutionEngine/ReusableOperatorABI.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

static double simulationTime=0;
double sc_time_stamp() { return simulationTime; }

namespace {
[[noreturn]] void fail(const char *reason) {
  std::fprintf(stderr,"REUSABLE_OP RTL FAILED: %s\n",reason);
  std::abort();
}
class Simulator {
  Voperator dut;
  uint32_t *memory=nullptr;
  uint64_t beats=0, readAt=0, writeAt=0, readLeft=0, writeLeft=0;
  uint64_t outputBegin=0,outputEnd=0,cycles=0,reads=0,writes=0;
  uint64_t readyAt=0,latency=0;
  unsigned invocation=0,stallMask=0;
  uint64_t maxCycles=2000000000ull;
  void tick() {
    bool idle=!readLeft && !writeLeft;
    bool available=!stallMask || (cycles & stallMask)!=0;
    dut.clk=0;
    simulationTime=double(cycles)*2;
    dut.dma_read_ctrl_ready=idle && available;
    dut.dma_write_ctrl_ready=0;
    dut.dma_read_chnl_valid=readLeft && available && cycles>=readyAt;
    dut.dma_read_chnl_data=readLeft ? uint64_t(memory[readAt*2]) | (uint64_t(memory[readAt*2+1])<<32) : 0;
    dut.dma_write_chnl_ready=writeLeft && available && cycles>=readyAt;
    dut.eval();
    dut.dma_write_ctrl_ready=idle && available && !dut.dma_read_ctrl_valid;
    dut.eval();
    bool rc=dut.dma_read_ctrl_valid && dut.dma_read_ctrl_ready;
    bool rd=dut.dma_read_chnl_valid && dut.dma_read_chnl_ready;
    bool wc=dut.dma_write_ctrl_valid && dut.dma_write_ctrl_ready;
    bool wd=dut.dma_write_chnl_valid && dut.dma_write_chnl_ready;
    uint64_t ra=dut.dma_read_ctrl_data_index, rn=dut.dma_read_ctrl_data_length;
    uint64_t wa=dut.dma_write_ctrl_data_index, wn=dut.dma_write_ctrl_data_length;
    uint64_t data=dut.dma_write_chnl_data;
    if(rc && (dut.dma_read_ctrl_data_size!=3 || !rn || ra+rn>beats)) fail("invalid DMA read");
    if(wc && (dut.dma_write_ctrl_data_size!=3 || !wn || wa+wn>beats || wa<outputBegin || wa+wn>outputEnd))
      fail("DMA write outside current output tensor");
    dut.clk=1;
    simulationTime+=1;
    dut.eval();
    if(rd) { ++readAt; --readLeft; ++reads; }
    if(wd) {
      if(data>>32) fail("nonzero DMA64 padding");
      memory[writeAt*2]=uint32_t(data); memory[writeAt*2+1]=uint32_t(data>>32);
      ++writeAt; --writeLeft; ++writes;
    }
    if(rc) { readAt=ra; readLeft=rn; readyAt=cycles+latency; }
    if(wc) { writeAt=wa; writeLeft=wn; readyAt=cycles+latency; }
    ++cycles;
    if(Verilated::gotFinish()) fail("RTL simulation terminated unexpectedly");
  }
public:
  Simulator() {
    if(auto *value=std::getenv("SODAP_RTL_STALL_MASK")) stallMask=std::strtoul(value,nullptr,0);
    if(auto *value=std::getenv("SODAP_RTL_DMA_LATENCY")) latency=std::strtoull(value,nullptr,0);
    if(auto *value=std::getenv("SODAP_RTL_MAX_CYCLES")) maxCycles=std::strtoull(value,nullptr,0);
    dut.rst=0; dut.conf_done=0;
    for(unsigned i=0;i<8;++i) tick();
    dut.rst=1;
    for(unsigned i=0;i<4;++i) tick();
  }
  void run(uint32_t *arena,uint64_t bytes) {
    if(readLeft || writeLeft) fail("previous invocation has outstanding DMA");
    memory=arena; beats=bytes/8;
    auto f=[&](unsigned index)->uint64_t{return arena[index*2];};
    uint64_t elements=0;
    if(f(0)==OP_CONV_MAGIC) { outputBegin=f(CONV_C)/2; elements=f(CONV_N)*f(CONV_OH)*f(CONV_OW)*f(CONV_OC); }
    else if(f(0)==OP_ADD_MAGIC) { outputBegin=f(ADD_C)/2; elements=f(ADD_COUNT); }
    else if(f(0)==GEMM_MAGIC) { outputBegin=f(GEMM_FIELD_C)/2; elements=f(GEMM_FIELD_M)*f(GEMM_FIELD_N); }
    else fail("unknown descriptor magic");
    outputEnd=outputBegin+elements;
    uint64_t start=cycles,oldReads=reads,oldWrites=writes;
    dut.conf_done=1; tick(); dut.conf_done=0;
    do {
      tick();
      if(cycles-start>maxCycles) fail("invocation exceeded cycle budget");
    } while(!dut.acc_done);
    if(readLeft || writeLeft) fail("completion precedes DMA drain");
    if(writes-oldWrites!=elements) fail("incorrect number of output elements");
    uint64_t elapsed=cycles-start;
    for(unsigned i=0;i<4;++i) tick();
    std::printf("NN_RTL invocation=%u cycles=%llu read_beats=%llu write_beats=%llu\n",
      invocation++,(unsigned long long)elapsed,(unsigned long long)(reads-oldReads),(unsigned long long)(writes-oldWrites));
    std::fflush(stdout);
  }
};
}
extern "C" void esp_op_execute_rtl(uint32_t *arena,uint64_t bytes) {
  static const bool initialized=[]() {
    const char *args[]={"reusable-nn-cosim"};
    Verilated::commandArgs(1,args);
    return true;
  }();
  (void)initialized;
  static Simulator simulator;
  simulator.run(arena,bytes);
}
