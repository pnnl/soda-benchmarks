// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/ExecutionEngine/ReusableOperatorABI.h"
#include <cassert>
#include <cstdio>
#include <cstring>
struct Vector { float *allocated,*aligned; int64_t offset,size,stride; };
int main(int argc,char **argv) {
  if(argc>1 && !std::strcmp(argv[1],"handle")) esp_op_close(42);
  auto ctx=esp_op_open(512,0);
  if(argc>1) {
    if(!std::strcmp(argv[1],"duplicate")) esp_op_open(512,0);
    if(!std::strcmp(argv[1],"bounds")) esp_op_add(ctx,0,128,256,INT64_MAX,0);
    if(!std::strcmp(argv[1],"overlap")) esp_op_add(ctx,0,128,0,4,0);
    if(!std::strcmp(argv[1],"alignment")) esp_op_add(ctx,1,128,256,4,0);
    if(!std::strcmp(argv[1],"flag")) esp_op_add(ctx,0,128,256,4,2);
    if(!std::strcmp(argv[1],"permutation")) esp_op_transpose(ctx,0,128,2,2,2,1,1,0,0,2,3);
    return 2;
  }
  float a[]={-1,2,3,-4},b[]={2,3,-4,-5},out[4]={};
  Vector av{a,a,0,4,1},bv{b,b,0,4,1},ov{out,out,0,4,1};
  esp_op_copy_in(ctx,0,1,&av); esp_op_copy_in(ctx,128,1,&bv);
  esp_op_add(ctx,0,128,256,4,1);
  esp_op_copy_out(ctx,256,1,&ov);
  const float expected[]={1,5,0,0}; assert(!std::memcmp(out,expected,sizeof out));
  // Reuse the same engine with a smaller shape and disabled activation.
  esp_op_add(ctx,0,128,256,3,0);
  esp_op_copy_out(ctx,256,1,&ov);
  assert(out[0]==1 && out[1]==5 && out[2]==-1 && out[3]==0);
  esp_op_close(ctx);
  // A closed CPU context releases its arena for subsequent inferences. With
  // the test's 1024-byte static heap, two live allocations would not fit.
  ctx=esp_op_open(512,0);
  esp_op_close(ctx);
  std::puts("Reusable operator runtime PASS");
}
