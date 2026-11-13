// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Executable host-ABI regression: the simulated IP increments only element 0.
#include "sodap/ExecutionEngine/ESPRuntime.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#ifndef SLOT_STRIDE
#define SLOT_STRIDE 1
#endif
static float *shared;
static unsigned starts, completed[16];
static bool open[16];
extern "C" int64_t esp_alloc_shared(int64_t bytes) {
  assert(bytes == 16 * SLOT_STRIDE && !shared);
  shared = static_cast<float *>(std::calloc(1, bytes));
  return reinterpret_cast<int64_t>(shared);
}
extern "C" void esp_free_shared(int64_t handle) {
  assert(handle == reinterpret_cast<int64_t>(shared));
  std::free(shared);
  shared = nullptr;
}
extern "C" void esp_copy_to_shared_f32(int64_t rank, void *desc,
                                       int64_t handle, int64_t offset) {
  assert(sodap::copyMemRefF32(rank, desc, reinterpret_cast<void *>(handle),
                            4 * SLOT_STRIDE, offset, true));
}
extern "C" void esp_copy_from_shared_f32(int64_t handle, int64_t offset,
                                         int64_t rank, void *desc) {
  assert(sodap::copyMemRefF32(rank, desc, reinterpret_cast<void *>(handle),
                            4 * SLOT_STRIDE, offset, false));
}
extern "C" void esp_copy_to_shared_f32_strided(int64_t rank, void *desc,
                                               int64_t handle, int64_t offset,
                                               int64_t stride) {
  assert(stride == SLOT_STRIDE);
  assert(sodap::copyMemRefF32(rank, desc, reinterpret_cast<void *>(handle),
                            4 * SLOT_STRIDE, offset, true, stride));
}
extern "C" void esp_copy_from_shared_f32_strided(int64_t handle, int64_t offset,
                                                 int64_t rank, void *desc,
                                                 int64_t stride) {
  assert(stride == SLOT_STRIDE);
  assert(sodap::copyMemRefF32(rank, desc, reinterpret_cast<void *>(handle),
                            4 * SLOT_STRIDE, offset, false, stride));
}
extern "C" void esp_accel_start() {
  assert(open[2] && open[7]);
  assert(shared[0] == 5 + starts);
  assert(shared[SLOT_STRIDE] == 6 && shared[2 * SLOT_STRIDE] == 7 && shared[3 * SLOT_STRIDE] == 8);
  if (SLOT_STRIDE == 2)
    for (unsigned i = 1; i < 8; i += 2)
      assert(shared[i] == 0);
  shared[0] += 1;
  ++starts;
}
extern "C" void esp_accel_wait() {}
extern "C" void esp_prof_begin(unsigned id) {
  assert(id < 16 && !open[id]);
  open[id] = true;
}
extern "C" void esp_prof_end(unsigned id) {
  assert(id < 16 && open[id]);
  open[id] = false;
  ++completed[id];
}
extern "C" void check_result(int64_t rank, void *desc) {
  auto view = sodap::decodeMemRefF32(rank, desc);
  assert(view.numElements == 4);
  assert(view.data[0] == 7 && view.data[1] == 6 &&
         view.data[2] == 7 && view.data[3] == 8);
  for (unsigned id : {1, 2, 3, 5, 7})
    assert(completed[id] == 2 && !open[id]);
  assert(!shared && starts == 2);
  std::puts("kernel inout and profiling PASS");
}
