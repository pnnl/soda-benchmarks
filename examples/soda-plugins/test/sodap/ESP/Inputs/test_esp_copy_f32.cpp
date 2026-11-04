// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/ExecutionEngine/ESPRuntime.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>

template <unsigned Rank>
struct Descriptor {
  sodap::MemRefDescriptorHeadF32 head;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

static uint32_t bits(const float &value) {
  uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

int main() {
  constexpr unsigned words = 24;
  constexpr uint32_t guard = 0x5a5a5a5a;
  int64_t handle = esp_alloc_shared(words * sizeof(float));
  assert(handle);
  auto *shared = reinterpret_cast<uint32_t *>(handle);
  for (unsigned i = 0; i < words; ++i)
    shared[i] = guard;

  float source[16], output[16];
  for (unsigned i = 0; i < 16; ++i) {
    source[i] = static_cast<float>(i) - 8.5f;
    output[i] = -100.0f;
  }
  // Nonzero base offset, transposed/strided source, padded destination.
  Descriptor<2> src{{source, source, 2}, {2, 3}, {1, 4}};
  Descriptor<2> dst{{output, output, 1}, {2, 3}, {8, 2}};
  esp_copy_to_shared_f32(2, &src, handle, 4);
  esp_copy_from_shared_f32(handle, 4, 2, &dst);
  for (unsigned i = 0; i < 6; ++i) {
    unsigned row = i / 3, col = i % 3;
    assert(shared[4 + i] == bits(source[2 + row + 4 * col]));
    assert(bits(output[1 + 8 * row + 2 * col]) == shared[4 + i]);
  }
  for (unsigned i = 0; i < 16; ++i)
    if (i != 1 && i != 3 && i != 5 && i != 9 && i != 11 && i != 13)
      assert(output[i] == -100.0f);
  for (unsigned i = 0; i < words; ++i)
    if (i < 4 || i >= 10)
      assert(shared[i] == guard);

  // Rank zero: preserve representations, including signed zero and NaN payload.
  for (uint32_t value : {0x80000000u, 0x7fc12345u, 0x7f800000u, 0xff800000u}) {
    float scalar, result = 1.0f;
    std::memcpy(&scalar, &value, sizeof(value));
    sodap::MemRefDescriptorHeadF32 in{&scalar, &scalar, 0};
    sodap::MemRefDescriptorHeadF32 out{&result, &result, 0};
    esp_copy_to_shared_f32(0, &in, handle, 23);
    esp_copy_from_shared_f32(handle, 23, 0, &out);
    assert(bits(result) == value);
  }

  Descriptor<1> reversed{{source, source, 6}, {4}, {-2}};
  esp_copy_to_shared_f32(1, &reversed, handle, 0);
  for (unsigned i = 0; i < 4; ++i)
    assert(shared[i] == bits(source[6 - 2 * i]));

  uint32_t before[words];
  std::memcpy(before, shared, sizeof(before));
  Descriptor<2> empty{{nullptr, nullptr, 0}, {0, 100}, {100, 1}};
  assert(sodap::copyMemRefF32(2, &empty, shared, words, words, true));
  esp_copy_to_shared_f32(2, &src, handle, words - 1); // Out of bounds.
  esp_copy_to_shared_f32(2, &src, handle, -1);
  esp_copy_to_shared_f32(2, &src, 0, 0); // Unknown handle.
  assert(!sodap::copyMemRefF32(-1, &src, shared, words, 0, true));
  Descriptor<2> huge{
      {source, source, 0}, {std::numeric_limits<int64_t>::max(), 2}, {2, 1}};
  assert(!sodap::copyMemRefF32(2, &huge, shared, words, 0, true));
  Descriptor<1> overflow{
      {source, source, 1}, {2}, {std::numeric_limits<int64_t>::max()}};
  assert(!sodap::copyMemRefF32(1, &overflow, shared, words, 0, true));
  assert(std::memcmp(before, shared, sizeof(before)) == 0);
  // Padded shared layout, strided host view, and untouched padding guards.
  for (unsigned i = 0; i < words; ++i) shared[i] = guard;
  esp_copy_to_shared_f32_strided(2, &src, handle, 2, 2);
  esp_copy_from_shared_f32_strided(handle, 2, 2, &dst, 2);
  for (unsigned i = 0; i < 6; ++i) {
    assert(shared[2 + 2 * i] == bits(source[2 + i / 3 + 4 * (i % 3)]));
    assert(shared[3 + 2 * i] == guard);
    assert(bits(output[1 + 8 * (i / 3) + 2 * (i % 3)]) == shared[2 + 2 * i]);
  }
  std::memcpy(before, shared, sizeof(before));
  esp_copy_to_shared_f32_strided(2, &src, handle, 14, 2); // Last slot out of bounds.
  esp_copy_to_shared_f32_strided(2, &src, handle, 0, 0);
  assert(std::memcmp(before, shared, sizeof(before)) == 0);
  esp_free_shared(handle);

  // Sequential calls use the same ABI; this mock actually reclaims storage.
  // The ESP RISC-V bump allocator's aligned_free remains a separate limitation.
  handle = esp_alloc_shared(sizeof(float));
  sodap::MemRefDescriptorHeadF32 scalar{source, source, 3};
  esp_copy_to_shared_f32(0, &scalar, handle, 0);
  assert(*reinterpret_cast<uint32_t *>(handle) == bits(source[3]));
  esp_free_shared(handle);
  std::puts("ESP f32 copy PASS");
}
