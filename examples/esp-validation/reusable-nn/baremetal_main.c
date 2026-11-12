// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <stdint.h>
#include <stdio.h>
#include "model.h"
extern const float nn_input[], nn_parameters[], nn_reference[];
extern void forward_kernel(const float *, const float *, float *);
static float output[NN_OUTPUT_ELEMENTS];
static uint64_t cycle(void) {
  uint64_t result;
  __asm__ volatile("csrr %0, mcycle" : "=r"(result));
  return result;
}
extern void tohost_exit(uintptr_t code) __attribute__((noreturn));
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  (void)regs;
  printf("NN_TRAP cause=%lu epc=%lx\n",(unsigned long)cause,(unsigned long)epc);
  tohost_exit(1337);
}
int main(void) {
  printf("NN_BEGIN outputs=%u arena_bytes=%u\n", NN_OUTPUT_ELEMENTS, NN_ARENA_BYTES);
#ifdef SODAP_OP_CPU_ONLY
  printf("NN_MODE execution=CPU_ONLY arena=cacheable\n");
#endif
  uint64_t start=cycle();
  forward_kernel(nn_input, nn_parameters, output);
  uint64_t elapsed=cycle()-start;
  unsigned errors=0, top=0, gold_top=0;
  float max_error=0;
  for(unsigned i=0;i<NN_OUTPUT_ELEMENTS;++i) {
    float error=output[i]-nn_reference[i];
    if(error<0) error=-error;
    float scale=nn_reference[i]<0 ? -nn_reference[i] : nn_reference[i];
    if(!(error<=0.001f+0.0001f*scale)) ++errors;
    if(error>max_error) max_error=error;
    if(output[i]>output[top]) top=i;
    if(nn_reference[i]>nn_reference[gold_top]) gold_top=i;
  }
  double scaled=(double)max_error*1e9+0.5;
  if(scaled>4000000000.0) scaled=4000000000.0;
  printf("NN_RESULT outputs=%u errors=%u max_error_e9=%u top1=%u reference_top1=%u cycles=%lu\n",
    NN_OUTPUT_ELEMENTS,errors,(unsigned)scaled,top,gold_top,(unsigned long)elapsed);
  printf(errors ? "TEST FAILED\n" : "TEST PASSED\n");
  return errors ? 1 : 0;
}
