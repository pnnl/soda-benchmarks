// Measurement harness around the unchanged generated host object/runtime.
#include <stdint.h>
#include <stdio.h>
#include <esp_probe.h>
#include "esp_prof.h"
#include "fixture.h"
extern void tohost_exit(uintptr_t) __attribute__((noreturn));
static uint64_t discovery;
static uint64_t cycle(void){uint64_t c;__asm__ volatile("csrr %0, mcycle":"=r"(c)::"memory");return c;}
int pb_driver_probe(struct esp_device **,unsigned,unsigned,const char*);
int probe(struct esp_device **dev,unsigned vendor,unsigned id,const char *name){
 uint64_t start=cycle();int n=pb_driver_probe(dev,vendor,id,name);discovery+=cycle()-start;return n;
}
static uint32_t bits(float x){union{float f;uint32_t u;}v={.f=x};return v.u;}
uintptr_t handle_trap(uintptr_t cause,uintptr_t epc,uintptr_t regs[32]){
 (void)regs;printf("PB_TRAP cause=%lu epc=%lx\n",(unsigned long)cause,(unsigned long)epc);tohost_exit(1337);
}
int main(void){
#ifdef PB_SIMULATION
 // Match the manual baseline's compact simulation UART fixture. The board
 // application retains the standard baud divider and clock configuration.
 *(volatile uint32_t*)UART_SCALER=1;
 __asm__ volatile("fence iorw, iorw":::"memory");
#endif
 printf("\nPB_BEGIN kernel=%s dataset=%s dtype=float32 mode=esp\n",PB_NAME,PB_DATASET);
 printf("PB_IMPLEMENTATION variant=integration_thin_candidate guard_check=0\n");
 for(unsigned i=0;i<PB_OUTPUT_ELEMENTS;++i)pb_output[i]=0;
 esp_prof_reset();discovery=0;uint64_t start=cycle();PB_CALL();uint64_t raw=cycle()-start;
 uint64_t alloc=esp_prof_cycles(ESP_PROF_ALLOC);
 if(raw<discovery || alloc<discovery){printf("PB_ERROR profiling\nTEST FAILED\n");return 1;}
 unsigned errors=0;float max_abs=0,max_rel=0;
 for(unsigned i=0;i<PB_OUTPUT_ELEMENTS;++i){
  float diff=pb_output[i]-pb_reference[i];if(diff<0)diff=-diff;
  float scale=pb_reference[i]<0 ? -pb_reference[i] : pb_reference[i];
  if(!(diff<=0.001f+0.0001f*scale)){
   if(errors<4)printf("PB_MISMATCH index=%u actual=%08x reference=%08x\n",i,bits(pb_output[i]),bits(pb_reference[i]));
   ++errors;
  }
  if(diff>max_abs)max_abs=diff;float rel=diff/(scale>0.001f?scale:0.001f);if(rel>max_rel)max_rel=rel;
 }
 printf("PB_RAW_TIMING total=%lu discovery=%lu\n",(unsigned long)raw,(unsigned long)discovery);
 printf("PB_TIMING total=%lu alloc=%lu pack=%lu config=0 accel=%lu unpack=%lu discovery=%lu\n",
  (unsigned long)(raw-discovery),(unsigned long)(alloc-discovery),
  (unsigned long)esp_prof_cycles(ESP_PROF_PACK),(unsigned long)esp_prof_cycles(ESP_PROF_ACCEL),
  (unsigned long)esp_prof_cycles(ESP_PROF_UNPACK),(unsigned long)discovery);
 // The unchanged runtime returns after DONE. CPU output validation is complete;
 // this harness does not add red zones to its shared-arena layout.
 printf("PB_RESULT outputs=%u errors=%u guard_check=0 max_abs_bits=%08x max_rel_bits=%08x runtime_returned=1\n",
  PB_OUTPUT_ELEMENTS,errors,bits(max_abs),bits(max_rel));
 printf(errors?"TEST FAILED\n":"TEST PASSED\n");return errors?1:0;
}
