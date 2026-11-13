// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/ExecutionEngine/ReusableGemmABI.h"
#include <cassert>
#include <cstdio>
#include <cstring>

struct Vector {
  float *allocated, *aligned;
  int64_t offset, size, stride;
};
template <unsigned N> Vector view(float (&x)[N]) { return {x, x, 0, N, 1}; }

int main(int argc, char **argv) {
  if (argc > 1 && !std::strcmp(argv[1], "handle")) esp_gemm_close(42);
  constexpr int64_t a=128,b=192,c=264,w=328,y=344,bytes=360;
  int64_t context = esp_gemm_open(bytes, 0);
  if (argc > 1) {
    if (!std::strcmp(argv[1], "duplicate")) esp_gemm_open(bytes, 0);
    if (!std::strcmp(argv[1], "bounds"))
      esp_gemm_invoke(context,a,b,c,2,2,INT64_MAX,32,24,32,1,0);
    if (!std::strcmp(argv[1], "overlap"))
      esp_gemm_invoke(context,a,b,a,2,2,3,32,24,32,1,0);
    if (!std::strcmp(argv[1], "alignment"))
      esp_gemm_invoke(context,a+1,b,c,2,2,3,32,24,32,1,0);
    return 2;
  }
  float A[]={1,2,3,99,4,5,6,99};
  float B[]={1,2,99,3,4,99,5,6,99};
  float C[]={2,2,99,99,2,2,99,99};
  float W[]={1,2}, Y[]={0,99};
  auto av=view(A),bv=view(B),cv=view(C),wv=view(W),yv=view(Y);
  esp_gemm_copy_in(context,a,1,&av);
  esp_gemm_copy_in(context,b,1,&bv);
  esp_gemm_copy_in(context,c,1,&cv);
  esp_gemm_copy_in(context,w,1,&wv);
  esp_gemm_copy_in(context,y,1,&yv);
  // Noncontiguous rows, nontrivial alpha/beta, then a dependent different shape.
  esp_gemm_invoke(context,a,b,c,2,2,3,32,24,32,0.5f,2);
  esp_gemm_invoke(context,c,w,y,1,1,2,32,8,16,1,0);
  for (unsigned i=0;i<8;++i)
    esp_gemm_invoke(context,c,w,y,1,1,2,32,8,16,1,1);
  esp_gemm_copy_out(context,c,1,&cv);
  esp_gemm_copy_out(context,y,1,&yv);
  const float expected[]={15,18,99,99,28.5,36,99,99};
  assert(!std::memcmp(C,expected,sizeof(C)));
  assert(Y[0]==459 && Y[1]==99);
  esp_gemm_copy_out(context,a,1,&av);
  esp_gemm_copy_out(context,b,1,&bv);
  const float originalA[]={1,2,3,99,4,5,6,99};
  const float originalB[]={1,2,99,3,4,99,5,6,99};
  assert(!std::memcmp(A,originalA,sizeof(A)) && !std::memcmp(B,originalB,sizeof(B)));
  esp_gemm_close(context);
  std::puts("Reusable GEMM runtime PASS");
}
