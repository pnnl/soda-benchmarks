// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/ExecutionEngine/ReusableOperatorABI.h"
#include <cassert>
#include <cstring>
#include <cstdio>
struct Vector { float *allocated,*aligned; int64_t offset,size,stride; };
int main(int argc,char **argv) {
  auto ctx=esp_op_open(256,0);
  float x[]={-4,2,-0.5f},w=2,b=1,out[3]={};
  Vector xv{x,x,0,3,1},wv{&w,&w,0,1,1},bv{&b,&b,0,1,1},ov{out,out,0,3,1};
  esp_op_copy_in(ctx,0,1,&xv); esp_op_copy_in(ctx,32,1,&wv); esp_op_copy_in(ctx,40,1,&bv);
  auto run=[&](int mode,uint32_t slope) {
    esp_op_conv_activation(ctx,0,32,40,64,1,1,3,1,1,1,1,1,3,1,1,0,0,1,1,mode,slope);
    esp_op_copy_out(ctx,64,1,&ov);
  };
  if(argc>1) {
    if(!std::strcmp(argv[1],"mode")) run(3,0);
    if(!std::strcmp(argv[1],"nan")) run(2,0x7fc00000);
    if(!std::strcmp(argv[1],"negative")) run(2,0xbf000000);
    if(!std::strcmp(argv[1],"large")) run(2,0x40000000);
    return 2;
  }
  run(2,0x3f000000); assert(out[0]==-3.5f && out[1]==5 && out[2]==0);
  run(1,0); assert(out[0]==0 && out[1]==5 && out[2]==0);
  run(0,0); assert(out[0]==-7 && out[1]==5 && out[2]==0);
  run(2,0x3e000000); assert(out[0]==-0.875f && out[1]==5 && out[2]==0);
  esp_op_close(ctx);
  std::puts("Convolution activation reuse PASS");
}
