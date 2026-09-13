//===- EspRuntime.cpp - ESP accelerator runtime -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The hardware implementation of the runtime declared in ESPRuntime.h.
// EspRuntimeMock.cpp is the other implementation; the two are interchangeable
// from the kernel's point of view, which is what lets the cpu backend exercise
// the lowering on a workstation.
//
// This runtime knows nothing about the accelerator it drives. It has no
// register map, no notion of a vector length or a tile, and no idea what any
// operand is for. The pass computes the layout and the register values; this
// file allocates, copies with the stride it is told, writes the registers it
// is told, and runs the ESP socket protocol -- probe, page table, flush, start,
// wait -- which is the same for every accelerator ESP has ever had.
//
// It is built only inside an ESP checkout: <esp_accelerator.h> and
// <esp_probe.h> come from ESP's baremetal support library, along with probe(),
// aligned_malloc() and the register names. See link_esp_app.sh, which stages
// this file into a buildable esp-app/ directory, and the CMake option
// SODAP_ENABLE_ESP_RUNTIME.
//
//===----------------------------------------------------------------------===//

#include "sodap/ExecutionEngine/ESPRuntime.h"

#include <cstdio>
#include <cstddef>
#include <cstdlib>

extern "C" {
#include <esp_accelerator.h>
#include <esp_probe.h>
}

// ESP applications declare their own token type; esp_accelerator.h does not
// provide one. Q16.16 in a 32-bit word.
typedef int32_t token_t;

namespace {

// --- Accelerator identity ---------------------------------------------------
// The one thing the runtime has to know: which device to probe for. Everything
// else about the accelerator arrives through esp_accel_write_reg.
constexpr unsigned kDevId = 0x074;
constexpr const char *kDevName = "sld,ffn_sysc_catapult";

// --- Fixed point ------------------------------------------------------------
// Q16.16, matching the accelerator's datapath.
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
/// esp_flush() prints over the UART and probes for LLC/L2 devices on every
/// call. On an SoC built without ESP caches both probes come back empty and the
/// call does nothing but cost cycles, so ask once.
bool cachesPresent() {
  struct esp_device *cdev = nullptr;
  int nllc = probe(&cdev, VENDOR_CACHE, DEVID_LLC_CACHE, DEVNAME_LLC_CACHE);
  int nl2 = probe(&cdev, VENDOR_CACHE, DEVID_L2_CACHE, DEVNAME_L2_CACHE);
  return nllc > 0 || nl2 > 0;
}

/// A memref seen as rows x cols, with the batch dimension folded into rows.
/// The kernels this runtime serves have static contiguous shapes, so this is
/// exact; `cols` is the innermost dimension and `rows` everything else.
struct RowMajor {
  const sodap::MemRefViewF32 view;
  int64_t rows;
  int64_t cols;
  explicit RowMajor(int64_t rank, void *ptr)
      : view(sodap::decodeMemRefF32(rank, ptr)) {
    cols = (rank > 0) ? view.sizes[rank - 1] : 1;
    rows = (cols > 0) ? view.numElements / cols : 0;
  }
};

/// Reject a transfer that would run off the end of the shared buffer.
bool fitsInBuffer(const char *what, int64_t offset, int64_t rows, int64_t ld,
                  int64_t cols) {
  int64_t capacity = g_state.memSize / sizeof(token_t);
  int64_t last = (rows > 0) ? offset + (rows - 1) * ld + cols : offset;
  if (offset >= 0 && ld >= cols && last <= capacity)
    return true;
  std::printf("esp: %s of %lldx%lld (ld %lld) at offset %lld overruns the "
              "%lld-element shared buffer\n",
              what, (long long)rows, (long long)cols, (long long)ld,
              (long long)offset, (long long)capacity);
  return false;
}

} // namespace

// --- malloc for the generated kernel ---------------------------------------
// The kernel calls malloc for the intermediates of whatever the pass left on
// the CPU (for gemm: the alpha/beta epilogue). ESP's baremetal build is
// -nostdlib and supplies only aligned_malloc(), which carves from the uncached
// DMA region -- the wrong pool for CPU-side scratch, and a waste of the region
// the accelerator needs. So provide a bump allocator: one static block and a
// pointer that only moves forward. The generated code never frees, so that is
// the whole allocator; MLIR over-allocates by its own alignment and aligns the
// pointer itself, so 8-byte alignment is enough.
//
// The magic guard makes initialisation independent of whether .bss was zeroed:
// malloc runs before any other entry point here, so it cannot rely on something
// else having gone first. Only on the SoC -- on the host, libc's malloc serves.
#ifdef __riscv
namespace {
constexpr unsigned long kBumpBytes = 64ul * 1024ul;
constexpr unsigned long kBumpMagic = 0x45535042ul; // "ESPB"
unsigned char g_bump[kBumpBytes];
unsigned long g_bumpOff;
unsigned long g_bumpMagic;
} // namespace

extern "C" void *malloc(std::size_t n) {
  if (g_bumpMagic != kBumpMagic) {
    g_bumpOff = 0;
    g_bumpMagic = kBumpMagic;
  }
  n = (n + 7ul) & ~7ul;
  if (g_bumpOff + n > kBumpBytes) {
    std::printf("malloc: bump allocator exhausted (%lu of %lu bytes, "
                "wanted %lu)\n",
                g_bumpOff, kBumpBytes, (unsigned long)n);
    return nullptr;
  }
  void *p = &g_bump[g_bumpOff];
  g_bumpOff += n;
  return p;
}
#endif // __riscv

extern "C" int64_t esp_alloc_shared(int64_t total_bytes) {
  if (g_state.buf) {
    std::printf("esp_alloc_shared: a buffer is already live; free it first\n");
    return 0;
  }

  struct esp_device *devs = nullptr;
  int ndev = probe(&devs, VENDOR_SLD, kDevId, kDevName);
  if (ndev == 0) {
    std::printf("esp_alloc_shared: %s not found\n", kDevName);
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

  // Zeroed, and not merely allocated: the pass leaves regions it never writes
  // (the bias of a bias-less matmul, padding columns) and expects them to read
  // as zero. Only the bytes the pass asked for -- the TLB rounding above is a
  // DMA constraint, not something the accelerator will read.
  for (unsigned i = 0; i < static_cast<unsigned>(total_bytes) / sizeof(token_t);
       ++i)
    g_state.buf[i] = 0;

  g_state.ptable = static_cast<unsigned **>(
      aligned_malloc(nchunk(memSize) * sizeof(unsigned *)));
  for (unsigned i = 0; i < nchunk(memSize); ++i)
    g_state.ptable[i] = reinterpret_cast<unsigned *>(
        &g_state.buf[i * (kChunkSize / sizeof(token_t))]);

  // ACC_COH_NONE is the one mode available on every SoC; the others need the
  // cache hierarchy to be present.
  g_state.coherence = ACC_COH_NONE;
  g_state.needFlush = cachesPresent();

  iowrite32(g_state.dev, COHERENCE_REG, g_state.coherence);
#ifndef __sparc
  iowrite32(g_state.dev, PT_ADDRESS_REG, (unsigned long long)g_state.ptable);
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

extern "C" void esp_float2fixed_f32(int64_t rank, void *ptr, int64_t mem_handle,
                                    int64_t offset, int64_t ld) {
  if (!g_state.buf)
    return;
  RowMajor src(rank, ptr);
  if (!fitsInBuffer("float2fixed", offset, src.rows, ld, src.cols))
    return;
  for (int64_t r = 0; r < src.rows; ++r) {
    const float *row = src.view.data + r * src.cols;
    token_t *dst = g_state.buf + offset + r * ld;
    for (int64_t c = 0; c < src.cols; ++c)
      dst[c] = static_cast<token_t>(row[c] * kFxScale);
  }
}

extern "C" void esp_fixed2float_f32(int64_t mem_handle, int64_t offset,
                                    int64_t ld, int64_t rank, void *ptr) {
  if (!g_state.buf)
    return;
  RowMajor dst(rank, ptr);
  if (!fitsInBuffer("fixed2float", offset, dst.rows, ld, dst.cols))
    return;
  for (int64_t r = 0; r < dst.rows; ++r) {
    const token_t *row = g_state.buf + offset + r * ld;
    float *out = dst.view.data + r * dst.cols;
    for (int64_t c = 0; c < dst.cols; ++c)
      out[c] = static_cast<float>(row[c]) / kFxScale;
  }
}

extern "C" void esp_accel_write_reg(uint32_t offset, uint32_t value) {
  if (!g_state.dev)
    return;
  iowrite32(g_state.dev, offset, value);
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
