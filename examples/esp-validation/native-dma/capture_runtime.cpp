// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Native validation shim. Executes the actual sequential tile schedule and
// captures the actual generated host's DMA layout, including its transposes.
#include "sodap/ExecutionEngine/ESPRuntime.h"
#include "tile_signature.h"
#include <vector>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static uint32_t *shared;
static size_t words;
static std::vector<uint32_t> before,after,mask;
extern "C" int64_t esp_alloc_shared(int64_t bytes) {
  assert(!shared && bytes>0 && bytes%4==0);
  words=bytes/4; shared=(uint32_t*)calloc(words,4);
  return (int64_t)shared;
}
extern "C" void esp_free_shared(int64_t h) { assert(h==(int64_t)shared);free(shared);shared=nullptr; }
extern "C" void esp_copy_to_shared_f32_strided(int64_t rank,void *p,int64_t h,int64_t offset,int64_t stride) {
  assert(h==(int64_t)shared);
  assert(sodap::copyMemRefF32(rank,p,shared,words,offset,true,stride));
}
extern "C" void esp_copy_from_shared_f32_strided(int64_t h,int64_t offset,int64_t rank,void *p,int64_t stride) {
  assert(h==(int64_t)shared);
  auto view=sodap::decodeMemRefF32(rank,p);
  assert(sodap::copyMemRefF32(rank,p,shared,words,offset,false,stride));
  for(int64_t i=0;i<view.numElements;i++) mask.at(offset+i*stride)=2;
}
extern "C" void esp_accel_start() {
  before.assign(shared,shared+words); mask.assign(words,1);
  std::vector<std::vector<float>> args;
  size_t offset=0;
  for(size_t count:tile_sizes) {
    args.emplace_back(count);
    for(size_t i=0;i<count;i++) memcpy(&args.back()[i],&shared[offset+2*i],4);
    offset+=2*count;
  }
  assert(offset==words);
  CALL_REFERENCE_TILE(args);
  offset=0;
  for(auto &arg:args) {
    for(size_t i=0;i<arg.size();i++) memcpy(&shared[offset+2*i],&arg[i],4);
    offset+=2*arg.size();
  }
  after.assign(shared,shared+words);
}
extern "C" void esp_accel_wait() {}
extern "C" void esp_prof_begin(unsigned) {}
extern "C" void esp_prof_end(unsigned) {}
extern "C" void esp_prof_report() {}
extern "C" void capture_vectors(const char *directory) {
  for(auto entry:{std::make_pair("initial",&before),std::make_pair("expected",&after),std::make_pair("mask",&mask)}) {
    char path[4096];snprintf(path,sizeof(path),"%s/%s.hex",directory,entry.first);
    FILE *f=fopen(path,"w");assert(f);
    for(uint32_t w:*entry.second) fprintf(f,"%08x\n",w);
    fclose(f);
  }
  char path[4096];snprintf(path,sizeof(path),"%s/parameters.svh",directory);
  FILE *f=fopen(path,"w");assert(f);fprintf(f,"localparam WORDS=%zu;\nlocalparam real TOL=0.001;\n",words);fclose(f);
}
