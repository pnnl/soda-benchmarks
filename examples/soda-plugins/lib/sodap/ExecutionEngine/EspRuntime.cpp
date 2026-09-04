//===- EspRuntime.cpp - ESP accelerator runtime -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The hardware implementation of the runtime declared in ESPRuntime.h, driving
// the FFN accelerator ("sld,ffn_sysc_catapult", device 0x074) over its
// memory-mapped registers. EspRuntimeMock.cpp is the other implementation; the
// two are interchangeable from the kernel's point of view, which is what lets
// the cpu backend exercise the lowering on a workstation.
//
// It is built only inside an ESP checkout: <esp_accelerator.h> and <esp_probe.h>
// come from ESP's baremetal support library, along with probe(), aligned_malloc()
// and the register names. See ll_to_riscv.sh, which stages this file into a
// buildable esp-app/ directory, and the CMake option SODAP_ENABLE_ESP_RUNTIME.
//
// The register map and the shared-memory layout are the ones documented in
// .specs/esp_invok.h and exercised by .specs/matmul_test.c; the platform setup
// (page table, coherence, the "is there anything to flush" probe) follows
// examples/bambu-esp-example/polybench_gemm/runtime/esp_gemm.c, which is a
// working runtime measured on a VC707.
//
//===----------------------------------------------------------------------===//

#include "sodap/ExecutionEngine/ESPRuntime.h"

#include <cstdio>
#include <cstdlib>

extern "C" {
#include <esp_accelerator.h>
#include <esp_probe.h>
}

namespace {

// --- Accelerator identity ---------------------------------------------------
constexpr unsigned kDevFFN = 0x074;
constexpr const char *kNameFFN = "sld,ffn_sysc_catapult";

// --- User-defined registers (esp_invok.h) -----------------------------------
constexpr unsigned kFFNSeqLen = 0x58; // rows of the input / output   (M)
constexpr unsigned kFFNInDim = 0x54;  // shared dimension             (K)
constexpr unsigned kFFNOutDim = 0x50; // cols of the weight / output  (N)
constexpr unsigned kFFNAddrI = 0x4c;  // element offset of the input
constexpr unsigned kFFNAddrW = 0x48;  // element offset of the weights
constexpr unsigned kFFNAddrB = 0x44;  // element offset of the bias
constexpr unsigned kFFNAddrO = 0x40;  // element offset of the output

// --- Fixed point ------------------------------------------------------------
// Q16.16, matching matmul_test.c's FX_WL/FX_IL and the accelerator's datapath.
constexpr int kFxFracBits = 16;
constexpr float kFxScale = static_cast<float>(1 << kFxFracBits);

// --- Scatter-gather DMA chunking -------------------------------------------
constexpr int kChunkShift = 20;
constexpr unsigned kChunkSize = 1u << kChunkShift;

unsigned nchunk(unsigned bytes) {
  return (bytes % kChunkSize == 0) ? (bytes / kChunkSize)
                                   : (bytes / kChunkSize) + 1;
}

// --- Per-invocation state ---------------------------------------------------
// The MLIR-facing API is stateless per call -- it has no esp_device* to pass --
// so the device handle, page table and buffer live here, established by
// esp_alloc_shared and torn down by esp_free_shared.
struct EspState {
  struct esp_device *dev = nullptr;
  unsigned **ptable = nullptr;
  token_t *buf = nullptr;
  unsigned memSize = 0;
  unsigned coherence = ACC_COH_NONE;
  bool needFlush = false;
};

EspState g_state;

/// Does this SoC have caches worth flushing?
///
/// esp_flush() prints over the UART and probes for LLC/L2 devices on every call.
/// On an SoC built without ESP caches both probes come back empty and the call
/// does nothing but cost cycles, so ask once. (esp_gemm.c:68-79 measured this at
/// ~576k cycles per invocation.)
bool cachesPresent() {
  struct esp_device *cdev = nullptr;
  int nllc = probe(&cdev, VENDOR_CACHE, DEVID_LLC_CACHE, DEVNAME_LLC_CACHE);
  int nl2 = probe(&cdev, VENDOR_CACHE, DEVID_L2_CACHE, DEVNAME_L2_CACHE);
  return nllc > 0 || nl2 > 0;
}

/// Reject a transfer that would run off the end of the shared buffer.
///
/// The offsets and the size come from two places that have to agree: the pass
/// computed them from M, K and N (ignoring the batch dimension -- correct only
/// while batch is 1, which is what TOSA gives these kernels), and the descriptor
/// carries the operand's real element count. A batch of 2 shows up here as twice
/// the elements the buffer was sized for, so check rather than corrupt memory.
bool fitsInBuffer(const char *what, int64_t offset,
                  const sodap::MemRefViewF32 &view) {
  int64_t capacity = g_state.memSize / sizeof(token_t);
  if (offset >= 0 && offset + view.numElements <= capacity)
    return true;
  std::printf("esp: %s of %lld elements at offset %lld overruns the %lld-element "
              "shared buffer (batch > 1?)\n",
              what, (long long)view.numElements, (long long)offset,
              (long long)capacity);
  return false;
}

} // namespace

extern "C" int64_t esp_alloc_shared(int64_t total_bytes) {
  if (g_state.buf) {
    std::printf("esp_alloc_shared: a buffer is already live; free it first\n");
    return 0;
  }

  struct esp_device *devs = nullptr;
  int ndev = probe(&devs, VENDOR_SLD, kDevFFN, kNameFFN);
  if (ndev == 0) {
    std::printf("esp_alloc_shared: %s not found\n", kNameFFN);
    return 0;
  }
  g_state.dev = &devs[0];

  unsigned memSize = static_cast<unsigned>(total_bytes);
  if (memSize < kChunkSize)
    memSize = kChunkSize; // the DMA engine always goes through the TLB

  if (ioread32(g_state.dev, PT_NCHUNK_MAX_REG) == 0) {
    std::printf("esp_alloc_shared: scatter-gather DMA is disabled\n");
    return 0;
  }
  if (ioread32(g_state.dev, PT_NCHUNK_MAX_REG) < nchunk(memSize)) {
    std::printf("esp_alloc_shared: not enough TLB entries (need %u)\n",
                nchunk(memSize));
    return 0;
  }

  g_state.buf = static_cast<token_t *>(aligned_malloc(memSize));
  if (!g_state.buf) {
    std::printf("esp_alloc_shared: allocation of %u bytes failed\n", memSize);
    return 0;
  }
  g_state.memSize = memSize;

  // Zeroed, and not merely allocated. The pass has no bias to offload, so it
  // points FFN_ADDRB at the output region (ESPPasses.cpp: off_b == off_o); the
  // accelerator still reads N bias words from there, and they have to be zero
  // for C = A*B to come out right.
  for (unsigned i = 0; i < memSize / sizeof(token_t); ++i)
    g_state.buf[i] = 0;

  g_state.ptable = static_cast<unsigned **>(
      aligned_malloc(nchunk(memSize) * sizeof(unsigned *)));
  for (unsigned i = 0; i < nchunk(memSize); ++i)
    g_state.ptable[i] =
        reinterpret_cast<unsigned *>(&g_state.buf[i * (kChunkSize /
                                                       sizeof(token_t))]);

  // ACC_COH_NONE is the one mode available on every SoC; the others need the
  // cache hierarchy to be present.
  g_state.coherence = ACC_COH_NONE;
  g_state.needFlush = cachesPresent();

  iowrite32(g_state.dev, COHERENCE_REG, g_state.coherence);
#ifndef __sparc
  iowrite32(g_state.dev, PT_ADDRESS_REG,
            (unsigned long long)g_state.ptable);
#else
  iowrite32(g_state.dev, PT_ADDRESS_REG, (unsigned)g_state.ptable);
#endif
  iowrite32(g_state.dev, PT_NCHUNK_REG, nchunk(memSize));
  iowrite32(g_state.dev, PT_SHIFT_REG, kChunkShift);
  iowrite32(g_state.dev, SRC_OFFSET_REG, 0x0);
  iowrite32(g_state.dev, DST_OFFSET_REG, 0x0);

  return static_cast<int64_t>(reinterpret_cast<intptr_t>(g_state.buf));
}

extern "C" void esp_free_shared(int64_t mem_handle) {
  if (!g_state.buf)
    return;
  aligned_free(g_state.ptable);
  aligned_free(g_state.buf);
  g_state = EspState();
}

extern "C" void esp_float2fixed_f32(int64_t rank, void *ptr,
                                    int64_t mem_handle, int64_t offset) {
  if (!g_state.buf)
    return;
  sodap::MemRefViewF32 src = sodap::decodeMemRefF32(rank, ptr);
  if (!fitsInBuffer("float2fixed", offset, src))
    return;
  for (int64_t i = 0; i < src.numElements; ++i)
    g_state.buf[offset + i] =
        static_cast<token_t>(src.data[i] * kFxScale);
}

extern "C" void esp_fixed2float_f32(int64_t mem_handle, int64_t offset,
                                    int64_t rank, void *ptr) {
  if (!g_state.buf)
    return;
  sodap::MemRefViewF32 dst = sodap::decodeMemRefF32(rank, ptr);
  if (!fitsInBuffer("fixed2float", offset, dst))
    return;
  for (int64_t i = 0; i < dst.numElements; ++i)
    dst.data[i] =
        static_cast<float>(g_state.buf[offset + i]) / kFxScale;
}

extern "C" void esp_accel_cfg_regs(int64_t seq_len, int64_t indim,
                                   int64_t outdim, int64_t off_in,
                                   int64_t off_w, int64_t off_b,
                                   int64_t off_o) {
  if (!g_state.dev)
    return;
  // Offsets are in token_t elements relative to the start of the buffer, not
  // byte addresses -- the same units the pass computed them in.
  iowrite32(g_state.dev, kFFNSeqLen, (unsigned)seq_len);
  iowrite32(g_state.dev, kFFNInDim, (unsigned)indim);
  iowrite32(g_state.dev, kFFNOutDim, (unsigned)outdim);
  iowrite32(g_state.dev, kFFNAddrI, (unsigned)off_in);
  iowrite32(g_state.dev, kFFNAddrW, (unsigned)off_w);
  iowrite32(g_state.dev, kFFNAddrB, (unsigned)off_b);
  iowrite32(g_state.dev, kFFNAddrO, (unsigned)off_o);
}

extern "C" void esp_accel_start() {
  if (!g_state.dev)
    return;
  if (g_state.needFlush)
    esp_flush(g_state.coherence);
  iowrite32(g_state.dev, CMD_REG, CMD_MASK_START);
}

extern "C" void esp_accel_wait() {
  if (!g_state.dev)
    return;
  unsigned done = 0;
  while (!done) {
    done = ioread32(g_state.dev, STATUS_REG);
    done &= STATUS_MASK_DONE;
  }
  iowrite32(g_state.dev, CMD_REG, 0x0);
}
