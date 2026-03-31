//===- EspRuntimeMock.cpp - Mock ESP runtime for testing --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mock implementation of the ESP accelerator runtime. All functions print
// their invocation to stdout and return safe defaults. This allows testing
// the MLIR-to-runtime lowering without actual ESP hardware.
//
//===----------------------------------------------------------------------===//

#include "sodap/ExecutionEngine/ESPRuntime.h"

#include <iostream>

extern "C" int64_t esp_alloc_shared(int64_t total_bytes) {
  std::cout << "Called: " << __func__ << std::endl;
  std::cout << "\t"
            << "total_bytes=" << total_bytes << std::endl;
  return 0; // opaque handle (mock)
}

extern "C" void esp_free_shared(int64_t mem_handle) {
  std::cout << "Called: " << __func__ << std::endl;
}

extern "C" void esp_float2fixed_f32(int64_t rank, void *ptr,
                                    int64_t mem_handle, int64_t offset) {
  std::cout << "Called: " << __func__ << std::endl;
  std::cout << "\t"
            << "rank=" << rank << ", offset=" << offset << std::endl;
}

extern "C" void esp_fixed2float_f32(int64_t mem_handle, int64_t offset,
                                    int64_t rank, void *ptr) {
  std::cout << "Called: " << __func__ << std::endl;
  std::cout << "\t"
            << "offset=" << offset << ", rank=" << rank << std::endl;
}

extern "C" void esp_accel_cfg_regs(int64_t seq_len, int64_t indim,
                                   int64_t outdim, int64_t off_in,
                                   int64_t off_w, int64_t off_b,
                                   int64_t off_o) {
  std::cout << "Called: " << __func__ << std::endl;
  std::cout << "\t"
            << "seq_len=" << seq_len << ", indim=" << indim
            << ", outdim=" << outdim << std::endl;
  std::cout << "\t"
            << "off_in=" << off_in << ", off_w=" << off_w
            << ", off_b=" << off_b << ", off_o=" << off_o << std::endl;
}

extern "C" void esp_accel_start() {
  std::cout << "Called: " << __func__ << std::endl;
}

extern "C" void esp_accel_wait() {
  std::cout << "Called: " << __func__ << std::endl;
}
