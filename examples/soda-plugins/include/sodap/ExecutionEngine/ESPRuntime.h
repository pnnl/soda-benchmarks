//===- EspRuntime.h - ESP accelerator runtime API ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the C runtime wrapper functions for the ESP accelerator that are
// called from MLIR-generated code. These functions mirror the low-level ESP
// invocation steps from esp_invok.h, wrapping them in an MLIR-compatible
// calling convention using unranked memref descriptors.
//
// Function signatures use the MLIR unranked memref ABI:
//   memref<*xT> is passed as (int64_t rank, void *descriptor)
//
//===----------------------------------------------------------------------===//
#ifndef SODAP_ESP_RUNTIME_H
#define SODAP_ESP_RUNTIME_H

#ifdef _WIN32
#ifdef mlir_esp_runner_utils_EXPORTS
#define MLIR_ESPRUNNERUTILS_EXPORT __declspec(dllexport)
#else
#define MLIR_ESPRUNNERUTILS_EXPORT __declspec(dllimport)
#endif
#else
#define MLIR_ESPRUNNERUTILS_EXPORT
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Allocate a contiguous shared memory buffer accessible by both the
/// processor and the ESP accelerator. Returns an unranked memref<*xi8>.
/// \param total_bytes  Size of the buffer in bytes.
MLIR_ESPRUNNERUTILS_EXPORT void esp_alloc_shared(int64_t total_bytes,
                                                 /* out */ int64_t *rank,
                                                 /* out */ void **descriptor);

/// Free a shared memory buffer previously allocated by esp_alloc_shared.
/// \param rank        Rank of the unranked memref (always 0 for this buffer).
/// \param descriptor  Pointer to the memref descriptor.
MLIR_ESPRUNNERUTILS_EXPORT void esp_free_shared(int64_t rank,
                                                void *descriptor);

/// Convert float data from a memref to fixed-point and copy it into the
/// shared memory buffer at the given element offset.
/// \param rank_src     Rank of the source float memref.
/// \param src          Descriptor of the source memref<*xf32>.
/// \param rank_mem     Rank of the shared memory memref.
/// \param mem          Descriptor of the shared memory memref<*xi8>.
/// \param offset       Element offset into shared memory.
MLIR_ESPRUNNERUTILS_EXPORT void esp_float2fixed_f32(int64_t rank_src,
                                                    void *src,
                                                    int64_t rank_mem,
                                                    void *mem,
                                                    int64_t offset);

/// Convert fixed-point data from shared memory back to float and store
/// into the destination memref.
/// \param rank_mem     Rank of the shared memory memref.
/// \param mem          Descriptor of the shared memory memref<*xi8>.
/// \param offset       Element offset into shared memory.
/// \param rank_dst     Rank of the destination float memref.
/// \param dst          Descriptor of the destination memref<*xf32>.
MLIR_ESPRUNNERUTILS_EXPORT void esp_fixed2float_f32(int64_t rank_mem,
                                                    void *mem,
                                                    int64_t offset,
                                                    int64_t rank_dst,
                                                    void *dst);

/// Configure the ESP accelerator registers for an FFN/matmul operation.
/// Sets dimensions and shared-memory offsets for each operand.
/// \param seq_len  Number of rows (M dimension).
/// \param indim    Shared/reduction dimension (K).
/// \param outdim   Number of columns (N dimension).
/// \param off_in   Element offset of input matrix in shared memory.
/// \param off_w    Element offset of weight matrix in shared memory.
/// \param off_b    Element offset of bias vector in shared memory.
/// \param off_o    Element offset of output matrix in shared memory.
MLIR_ESPRUNNERUTILS_EXPORT void esp_accel_cfg_regs(int64_t seq_len,
                                                   int64_t indim,
                                                   int64_t outdim,
                                                   int64_t off_in,
                                                   int64_t off_w,
                                                   int64_t off_b,
                                                   int64_t off_o);

/// Flush caches and start the ESP accelerator.
MLIR_ESPRUNNERUTILS_EXPORT void esp_accel_start();

/// Busy-wait until the ESP accelerator signals completion.
MLIR_ESPRUNNERUTILS_EXPORT void esp_accel_wait();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SODAP_ESP_RUNTIME_H
