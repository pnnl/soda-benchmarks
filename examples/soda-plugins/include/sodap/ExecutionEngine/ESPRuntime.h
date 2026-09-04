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
  float *data;            ///< aligned pointer, already advanced by `offset`
  int64_t rank;
  const int64_t *sizes;   ///< `rank` entries
  const int64_t *strides; ///< `rank` entries
  int64_t numElements;    ///< product of `sizes`
};

/// Decode the `(int64_t rank, void *descriptor)` pair an `memref<*xf32>`
/// argument arrives as.
///
/// The kernels this runtime serves come out of soda-opt with static, contiguous
/// shapes, so `numElements` is the product of the sizes and the strides are only
/// carried for a caller that wants to check that assumption.
inline MemRefViewF32 decodeMemRefF32(int64_t rank, void *descriptor) {
  auto *head = static_cast<MemRefDescriptorHeadF32 *>(descriptor);
  auto *tail = reinterpret_cast<int64_t *>(
      static_cast<char *>(descriptor) + sizeof(MemRefDescriptorHeadF32));

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

} // namespace sodap

/// Allocate a contiguous shared memory buffer accessible by both the
/// processor and the ESP accelerator.
/// \param total_bytes  Size of the buffer in bytes.
/// \return Opaque handle to the shared memory allocation.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT int64_t
esp_alloc_shared(int64_t total_bytes);

/// Free a shared memory buffer previously allocated by esp_alloc_shared.
/// \param mem_handle  Opaque handle returned by esp_alloc_shared.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void esp_free_shared(int64_t mem_handle);

/// Convert float data from a memref to fixed-point and copy it into the
/// shared memory buffer at the given element offset.
/// \param rank        Rank of the source float memref.
/// \param ptr         Descriptor of the source memref<*xf32>.
/// \param mem_handle  Opaque handle of the shared memory buffer.
/// \param offset      Element offset into shared memory.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void
esp_float2fixed_f32(int64_t rank, void *ptr, int64_t mem_handle,
                    int64_t offset);

/// Convert fixed-point data from shared memory back to float and store
/// into the destination memref.
/// \param mem_handle  Opaque handle of the shared memory buffer.
/// \param offset      Element offset into shared memory.
/// \param rank        Rank of the destination float memref.
/// \param ptr         Descriptor of the destination memref<*xf32>.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void
esp_fixed2float_f32(int64_t mem_handle, int64_t offset, int64_t rank,
                    void *ptr);

/// Configure the ESP accelerator registers for an FFN/matmul operation.
/// Sets dimensions and shared-memory offsets for each operand.
/// \param seq_len  Number of rows (M dimension).
/// \param indim    Shared/reduction dimension (K).
/// \param outdim   Number of columns (N dimension).
/// \param off_in   Element offset of input matrix in shared memory.
/// \param off_w    Element offset of weight matrix in shared memory.
/// \param off_b    Element offset of bias vector in shared memory.
/// \param off_o    Element offset of output matrix in shared memory.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void
esp_accel_cfg_regs(int64_t seq_len, int64_t indim, int64_t outdim,
                   int64_t off_in, int64_t off_w, int64_t off_b, int64_t off_o);

/// Flush caches and start the ESP accelerator.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void esp_accel_start();

/// Busy-wait until the ESP accelerator signals completion.
extern "C" MLIR_ESPRUNNERUTILS_EXPORT void esp_accel_wait();

#endif // SODAP_EXECUTIONENGINE_ESPRUNTIME_H
