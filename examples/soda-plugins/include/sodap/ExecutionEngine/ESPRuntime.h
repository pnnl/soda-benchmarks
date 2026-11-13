//===- ESPRuntime.h - ESP accelerator runtime API ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the C runtime wrapper functions for the ESP accelerator that are
// called from MLIR-generated code. These functions mirror the low-level ESP
// invocation steps from esp_invok.h.
//
// Functions that accept memrefs use the MLIR unranked memref ABI:
//   memref<*xT> is passed as (int64_t rank, void *descriptor)
//
// The shared memory buffer is represented as an opaque i64 handle to avoid
// returning memrefs from C (which has a complex struct-return ABI).
//
//===----------------------------------------------------------------------===//
#ifndef SODAP_EXECUTIONENGINE_ESPRUNTIME_H
#define SODAP_EXECUTIONENGINE_ESPRUNTIME_H

#ifdef _WIN32
#ifndef MLIR_ESPRUNNERUTILS_EXPORT
#ifdef mlir_esp_runner_utils_EXPORTS
#define MLIR_ESPRUNNERUTILS_EXPORT __declspec(dllexport)
#else
#define MLIR_ESPRUNNERUTILS_EXPORT __declspec(dllimport)
#endif
#endif
#else
#define MLIR_ESPRUNNERUTILS_EXPORT
#endif

#include <cstdint>
#include <cstring>

namespace sodap {

/// Head of the ranked descriptor an `memref<*xf32>` points at, as MLIR's
/// LLVM lowering lays it out:
///
///     { float *allocated; float *aligned; int64_t offset;
///       int64_t sizes[rank]; int64_t strides[rank]; }
///
/// The two arrays follow this header contiguously, which is why the header is a
/// struct and the arrays are reached by offsetting past it -- letting the
/// compiler place `offset` after the pointers keeps this correct on targets
/// where a pointer is not 64 bits (RISC-V ilp32, which some ESP SoCs use).
struct MemRefDescriptorHeadF32 {
  float *allocated;
  float *aligned;
  int64_t offset;
};

/// A decoded `memref<*xf32>` argument.
struct MemRefViewF32 {
  float *data; ///< aligned pointer, already advanced by `offset`
  int64_t rank;
  const int64_t *sizes;   ///< `rank` entries
  const int64_t *strides; ///< `rank` entries
  int64_t numElements;    ///< product of `sizes`
};

/// Decode the `(int64_t rank, void *descriptor)` pair an `memref<*xf32>`
/// argument arrives as.
///
/// The kernels this runtime serves come out of soda-opt with static, contiguous
/// shapes, so `numElements` is the product of the sizes and the strides are
/// only carried for a caller that wants to check that assumption.
inline MemRefViewF32 decodeMemRefF32(int64_t rank, void *descriptor) {
  auto *head = static_cast<MemRefDescriptorHeadF32 *>(descriptor);
  auto *tail = reinterpret_cast<int64_t *>(static_cast<char *>(descriptor) +
                                           sizeof(MemRefDescriptorHeadF32));

  MemRefViewF32 view;
  view.data = head->aligned + head->offset;
  view.rank = rank;
  view.sizes = tail;
  view.strides = tail + rank;
  view.numElements = 1;
  for (int64_t i = 0; i < rank; ++i)
    view.numElements *= tail[i];
  return view;
}

/// Copy f32 object representations between a strided memref and a dense shared
/// region. Offsets and capacity are in 32-bit elements. Rank zero is one scalar;
/// a zero extent copies nothing. The caller owns descriptor/storage validity.
/// Unlike float/fixed conversion, this preserves NaNs, infinities and -0 bits.
inline bool copyMemRefF32(int64_t rank, void *descriptor, void *shared,
                          int64_t capacity, int64_t offset, bool toShared,
                          int64_t sharedStride = 1) {
  if (rank < 0 || !descriptor || !shared || offset < 0 || offset > capacity ||
      sharedStride < 1)
    return false;
  auto *head = static_cast<MemRefDescriptorHeadF32 *>(descriptor);
  auto *sizes = reinterpret_cast<int64_t *>(static_cast<char *>(descriptor) +
                                          sizeof(MemRefDescriptorHeadF32));
  const int64_t *strides = sizes + rank;
  int64_t count = 1;
  bool empty = false;
  for (int64_t d = 0; d < rank; ++d) {
    if (sizes[d] < 0)
      return false;
    empty |= sizes[d] == 0;
  }
  if (empty)
    return true;
  int64_t available =
      capacity == offset ? 0 : 1 + (capacity - offset - 1) / sharedStride;
  for (int64_t d = 0; d < rank; ++d) {
    if (count > available / sizes[d])
      return false;
    count *= sizes[d];
  }
  if (count > available || !head->aligned)
    return false;

  // Check the descriptor's full index range before modifying either buffer.
  int64_t low = head->offset, high = head->offset;
  for (int64_t d = 0; d < rank; ++d) {
    int64_t span;
    if (__builtin_mul_overflow(sizes[d] - 1, strides[d], &span) ||
        (span < 0 ? __builtin_add_overflow(low, span, &low)
                  : __builtin_add_overflow(high, span, &high)))
      return false;
  }
  auto *bytes = static_cast<unsigned char *>(shared);

  // Most generated host arguments are contiguous, even when the shared
  // layout leaves a padding word between elements. Check the descriptor once
  // instead of reconstructing every element's indices with division/remainder.
  // Singleton dimensions do not constrain the physical layout.
  bool contiguous = true;
  int64_t expectedStride = 1;
  for (int64_t d = rank; d-- > 0;) {
    if (sizes[d] > 1 && strides[d] != expectedStride) {
      contiguous = false;
      break;
    }
    expectedStride *= sizes[d]; // The validated element count bounds this.
  }
  auto *host = head->aligned + head->offset;
  auto *slots = bytes + offset * sizeof(float);
  if (contiguous &&
      ((reinterpret_cast<uintptr_t>(host) |
        reinterpret_cast<uintptr_t>(slots)) % alignof(float)) == 0) {
    // memcpy preserves object representations without aliasing float storage
    // through an integer pointer. Alignment enables word loads/stores on
    // RISC-V instead of four separate byte transfers per float.
    host = static_cast<float *>(__builtin_assume_aligned(host, alignof(float)));
    slots = static_cast<unsigned char *>(
        __builtin_assume_aligned(slots, alignof(float)));
    const uint64_t step = static_cast<uint64_t>(sharedStride) * sizeof(float);
    if (toShared) {
      for (int64_t i = 0; i < count; ++i)
        std::memcpy(slots + i * step, host + i, sizeof(float));
    } else {
      for (int64_t i = 0; i < count; ++i)
        std::memcpy(host + i, slots + i * step, sizeof(float));
    }
    return true;
  }

  for (int64_t linear = 0; linear < count; ++linear) {
    int64_t index = head->offset, remainder = linear;
    for (int64_t d = rank; d-- > 0;) {
      index += (remainder % sizes[d]) * strides[d];
      remainder /= sizes[d];
    }
    void *slot = bytes + (offset + linear * sharedStride) * sizeof(float);
    if (toShared)
      std::memcpy(slot, head->aligned + index, sizeof(float));
    else
      std::memcpy(head->aligned + index, slot, sizeof(float));
  }
  return true;
}

} // namespace sodap

/// Allocate a contiguous shared memory buffer accessible by both the
/// processor and the ESP accelerator. The buffer is zeroed: the pass relies on
/// that for regions it never writes, such as the bias of a bias-less matmul.
/// \param total_bytes  Size of the buffer in bytes.
/// \return Opaque handle to the shared memory allocation.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT int64_t
esp_alloc_shared(int64_t total_bytes);

/// Free a shared memory buffer previously allocated by esp_alloc_shared.
/// \param mem_handle  Opaque handle returned by esp_alloc_shared.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void esp_free_shared(int64_t mem_handle);

/// Copy raw f32 values, with no numeric conversion, to/from dense shared memory.
/// The memref uses the unranked ABI above; its offset and strides are respected.
/// Shared offsets are in f32 elements, not bytes. Rank-zero scalars are supported.
/// The caller must wait for DMA completion before reading or releasing a buffer.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void
esp_copy_to_shared_f32(int64_t rank, void *ptr, int64_t mem_handle, int64_t offset);
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void
esp_copy_from_shared_f32(int64_t mem_handle, int64_t offset, int64_t rank,
                        void *ptr);

/// The same bit-preserving copies with a positive shared-memory element stride.
/// Stride 2 puts each f32 in the low word of a 64-bit slot. Padding is
/// untouched.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void
esp_copy_to_shared_f32_strided(int64_t rank, void *ptr, int64_t mem_handle,
                               int64_t offset, int64_t shared_stride);
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void
esp_copy_from_shared_f32_strided(int64_t mem_handle, int64_t offset,
                                 int64_t rank, void *ptr,
                                 int64_t shared_stride);

/// Convert float data from a memref to fixed-point and copy it into the
/// shared memory buffer at the given element offset, one row at a time.
///
/// The memref is treated as rows x cols with the batch dimension folded in
/// (rows = product of all but the last dimension). Row r of the source lands
/// at offset + r*ld in shared memory, so a K x N operand can be laid into a
/// K x Npad region: ld carries the padded stride and the runtime needs no
/// notion of why it is padded.
/// \param rank        Rank of the source float memref.
/// \param ptr         Descriptor of the source memref<*xf32>.
/// \param mem_handle  Opaque handle of the shared memory buffer.
/// \param offset      Element offset into shared memory.
/// \param ld          Leading dimension: elements between consecutive rows.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void
esp_float2fixed_f32(int64_t rank, void *ptr, int64_t mem_handle,
                    int64_t offset, int64_t ld);

/// Convert fixed-point data from shared memory back to float and store into
/// the destination memref, one row at a time. The inverse of
/// esp_float2fixed_f32: row r is read from offset + r*ld, and only the
/// destination's own width is read, so padding is dropped.
/// \param mem_handle  Opaque handle of the shared memory buffer.
/// \param offset      Element offset into shared memory.
/// \param ld          Leading dimension: elements between consecutive rows.
/// \param rank        Rank of the destination float memref.
/// \param ptr         Descriptor of the destination memref<*xf32>.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void
esp_fixed2float_f32(int64_t mem_handle, int64_t offset, int64_t ld,
                    int64_t rank, void *ptr);

/// Write one accelerator register. The pass emits one call per register with
/// the offset and the value it computed; the runtime has no register map and
/// no idea what any offset means. That is what makes it accelerator-agnostic.
/// \param offset  Byte offset of the register from the device base.
/// \param value   Value to write.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void
esp_accel_write_reg(uint32_t offset, uint32_t value);

/// Flush caches and start the ESP accelerator.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void esp_accel_start();

/// Busy-wait until the ESP accelerator signals completion.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void esp_accel_wait();

#endif // SODAP_EXECUTIONENGINE_ESPRUNTIME_H
