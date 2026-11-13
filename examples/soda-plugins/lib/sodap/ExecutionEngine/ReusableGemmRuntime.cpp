// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/ExecutionEngine/ReusableGemmABI.h"
#include "sodap/ExecutionEngine/ESPRuntime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __riscv
extern "C" {
#include <esp_accelerator.h>
#include <esp_probe.h>
}
#ifndef SODAP_ESP_DEVICE_ID
#error Define the reusable GEMM ESP device identity when building the runtime
#endif
#ifndef SODAP_ESP_DEVICE_NAME
#error Define SODAP_ESP_DEVICE_NAME
#endif
#endif

namespace {
struct Context {
  uint32_t *arena;
  uint32_t bytes, instance, invocations;
  bool running;
  Context *next;
#ifdef __riscv
  esp_device *device;
  unsigned **pageTable;
  bool flush;
#endif
};
Context *live = nullptr;

#if defined(SODAP_GEMM_TRACE) && !defined(__riscv)
// Optional native fixture capture. It observes the actual compiler-generated
// calls and arena, and is absent from the board runtime.
void trace(const Context &ctx, const char *kind) {
  const char *directory = std::getenv("SODAP_GEMM_TRACE_DIR");
  if (!directory) return;
  char path[4096];
  std::snprintf(path, sizeof(path), "%s/%s-%u.hex", directory, kind, ctx.invocations);
  FILE *file = std::fopen(path, "w");
  if (!file) std::abort();
  for (unsigned i = 0; i < ctx.bytes / 4; ++i)
    std::fprintf(file, "%08x\n", ctx.arena[i]);
  std::fclose(file);
  std::snprintf(path, sizeof(path), "%s/mask-%u.hex", directory, ctx.invocations);
  file = std::fopen(path, "w");
  if (!file) std::abort();
  uint32_t base = ctx.arena[2 * GEMM_FIELD_C];
  uint32_t stride = ctx.arena[2 * GEMM_FIELD_LDC];
  for (unsigned i = 0; i < ctx.bytes / 4; ++i) {
    bool output = i >= base && (i - base) / stride < ctx.arena[2 * GEMM_FIELD_M] &&
        (i - base) % stride < 2 * ctx.arena[2 * GEMM_FIELD_N] && (i - base) % 2 == 0;
    std::fprintf(file, "%x\n", output ? 2 : 0);
  }
  std::fclose(file);
  std::snprintf(path, sizeof(path), "%s/parameters.svh", directory);
  file = std::fopen(path, "w");
  if (!file) std::abort();
  std::fprintf(file, "localparam WORDS=%u;\nlocalparam HEADER_WORDS=%u;\nlocalparam INVOCATIONS=%u;\n",
               ctx.bytes / 4, GEMM_HEADER_BYTES / 4, ctx.invocations + 1);
  std::fclose(file);
}
#else
void trace(const Context &, const char *) {}
#endif

[[noreturn]] void fail(const char *reason) {
  std::printf("REUSABLE_GEMM TEST FAILED: %s\n", reason);
  std::exit(1);
}
void *allocate(unsigned bytes) {
#ifdef __riscv
  void *p = aligned_malloc(bytes);
#else
  void *p = std::malloc(bytes);
#endif
  if (!p) fail("allocation failed");
  return p;
}
void release(void *p) {
#ifdef __riscv
  aligned_free(p);
#else
  std::free(p);
#endif
}
Context &lookup(int64_t handle) {
  for (Context *p = live; p; p = p->next)
    if (reinterpret_cast<intptr_t>(p) == handle) return *p;
  fail("invalid context handle");
}
void fence() {
#ifdef __riscv
  __asm__ volatile("fence iorw, iorw" ::: "memory");
#endif
}
uint64_t endOf(Context &ctx, int64_t offset, int64_t rows,
               int64_t cols, int64_t stride) {
  if (offset < GEMM_HEADER_BYTES || offset % GEMM_SLOT_BYTES ||
      rows <= 0 || cols <= 0 || rows > UINT32_MAX || cols > UINT32_MAX ||
      stride <= 0 || stride % GEMM_SLOT_BYTES || stride > ctx.bytes ||
      uint64_t(stride) < uint64_t(cols) * GEMM_SLOT_BYTES)
    fail("invalid matrix descriptor");
  // Factors are bounded above before multiplication, so this fits uint64_t.
  uint64_t end = uint64_t(offset) + uint64_t(rows - 1) * stride +
                 uint64_t(cols) * GEMM_SLOT_BYTES;
  if (offset > ctx.bytes || end > ctx.bytes) fail("matrix exceeds arena");
  return end;
}
void copy(int64_t handle, int64_t offset, int64_t rank, void *descriptor, bool in) {
  Context &ctx = lookup(handle);
  if (ctx.running || offset < GEMM_HEADER_BYTES || offset % GEMM_SLOT_BYTES ||
      !sodap::copyMemRefF32(rank, descriptor, ctx.arena, ctx.bytes / 4,
                            offset / 4, in, GEMM_WORD_STRIDE))
    fail("invalid shared-memory copy");
}
} // namespace

extern "C" int64_t esp_gemm_open(int64_t bytes, int64_t instance) {
  if (bytes < GEMM_HEADER_BYTES || bytes > (1 << 30) ||
      bytes % GEMM_SLOT_BYTES || instance < 0 || instance > UINT32_MAX)
    fail("invalid arena size or instance");
  for (Context *p = live; p; p = p->next)
    if (p->instance == instance) fail("device instance already owned");
  auto *ctx = static_cast<Context *>(allocate(sizeof(Context)));
  *ctx = {};
  ctx->bytes = bytes;
  ctx->instance = instance;
#ifdef __riscv
  esp_device *devices = nullptr;
  int count = probe(&devices, VENDOR_SLD, SODAP_ESP_DEVICE_ID, SODAP_ESP_DEVICE_NAME);
  if (instance >= count) fail("device instance not found");
  ctx->device = &devices[instance];
  constexpr unsigned shift = 20, chunk = 1u << shift;
  unsigned chunks = (unsigned(bytes) + chunk - 1) / chunk;
  if (chunks > ioread32(ctx->device, PT_NCHUNK_MAX_REG))
    fail("insufficient DMA page table capacity");
  ctx->arena = static_cast<uint32_t *>(allocate(chunks * chunk));
  ctx->pageTable = static_cast<unsigned **>(allocate(chunks * sizeof(unsigned *)));
  for (unsigned i = 0; i < chunks; ++i)
    ctx->pageTable[i] = ctx->arena + i * (chunk / 4);
  esp_device *cache = nullptr;
  int llc = probe(&cache, VENDOR_CACHE, DEVID_LLC_CACHE, DEVNAME_LLC_CACHE);
  int l2 = probe(&cache, VENDOR_CACHE, DEVID_L2_CACHE, DEVNAME_L2_CACHE);
  ctx->flush = llc > 0 || l2 > 0;
  iowrite32(ctx->device, COHERENCE_REG, ACC_COH_NONE);
  iowrite32(ctx->device, PT_ADDRESS_REG, reinterpret_cast<uintptr_t>(ctx->pageTable));
  iowrite32(ctx->device, PT_NCHUNK_REG, chunks);
  iowrite32(ctx->device, PT_SHIFT_REG, shift);
  iowrite32(ctx->device, SRC_OFFSET_REG, 0);
  iowrite32(ctx->device, DST_OFFSET_REG, 0);
#else
  ctx->arena = static_cast<uint32_t *>(allocate(bytes));
#endif
  std::memset(ctx->arena, 0, bytes);
  ctx->next = live;
  live = ctx;
  std::printf("REUSABLE_GEMM open instance=%u arena_bytes=%u abi=%u\n",
              ctx->instance, ctx->bytes, GEMM_VERSION);
  return reinterpret_cast<intptr_t>(ctx);
}

extern "C" void esp_gemm_copy_in(int64_t ctx, int64_t offset,
                                 int64_t rank, void *descriptor) {
  copy(ctx, offset, rank, descriptor, true);
}
extern "C" void esp_gemm_copy_out(int64_t ctx, int64_t offset,
                                  int64_t rank, void *descriptor) {
  copy(ctx, offset, rank, descriptor, false);
}

extern "C" void esp_gemm_invoke(int64_t handle, int64_t a, int64_t b,
    int64_t c, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb,
    int64_t ldc, float alpha, float beta) {
  Context &ctx = lookup(handle);
  if (ctx.running) fail("instance is already running");
  uint64_t ae = endOf(ctx, a, m, k, lda), be = endOf(ctx, b, k, n, ldb);
  uint64_t ce = endOf(ctx, c, m, n, ldc);
  if ((uint64_t(a) < ce && uint64_t(c) < ae) ||
      (uint64_t(b) < ce && uint64_t(c) < be))
    fail("output overlaps an input");
  uint32_t descriptor[GEMM_FIELD_COUNT] = {};
  descriptor[GEMM_FIELD_MAGIC] = GEMM_MAGIC;
  descriptor[GEMM_FIELD_VERSION] = GEMM_VERSION;
  descriptor[GEMM_FIELD_M] = m;
  descriptor[GEMM_FIELD_N] = n;
  descriptor[GEMM_FIELD_K] = k;
  descriptor[GEMM_FIELD_A] = a / 4;
  descriptor[GEMM_FIELD_B] = b / 4;
  descriptor[GEMM_FIELD_C] = c / 4;
  descriptor[GEMM_FIELD_LDA] = lda / 4;
  descriptor[GEMM_FIELD_LDB] = ldb / 4;
  descriptor[GEMM_FIELD_LDC] = ldc / 4;
  std::memcpy(&descriptor[GEMM_FIELD_ALPHA], &alpha, sizeof(float));
  std::memcpy(&descriptor[GEMM_FIELD_BETA], &beta, sizeof(float));
  for (unsigned i = 0; i < GEMM_FIELD_COUNT; ++i) {
    ctx.arena[2 * i] = descriptor[i];
    ctx.arena[2 * i + 1] = 0;
  }
  ctx.running = true;
  trace(ctx, "initial");
  std::printf("REUSABLE_GEMM launch=%u instance=%u M=%u N=%u K=%u A=%u B=%u C=%u\n",
      ctx.invocations, ctx.instance, unsigned(m), unsigned(n), unsigned(k),
      unsigned(a), unsigned(b), unsigned(c));
#ifdef __riscv
  if (ctx.flush) esp_flush(ACC_COH_NONE);
  fence();
  iowrite32(ctx.device, CMD_REG, CMD_MASK_START);
  while (!(ioread32(ctx.device, STATUS_REG) & STATUS_MASK_DONE)) {}
  iowrite32(ctx.device, CMD_REG, 0);
  fence();
#else
  // A native implementation of the same descriptor contract, used to test
  // compiler isolation and arena lifetimes independently of HLS/simulation.
  auto read = [&](int64_t base, int64_t row, int64_t col, int64_t ld) {
    float value;
    std::memcpy(&value, reinterpret_cast<char *>(ctx.arena) + base + row * ld + col * 8, 4);
    return value;
  };
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      float sum = 0;
      for (int64_t q = 0; q < k; ++q)
        sum = sum + read(a, i, q, lda) * read(b, q, j, ldb);
      float value = alpha * sum;
      if (beta != 0) value = value + beta * read(c, i, j, ldc);
      char *slot = reinterpret_cast<char *>(ctx.arena) + c + i * ldc + j * 8;
      std::memcpy(slot, &value, 4);
      std::memset(slot + 4, 0, 4);
    }
#endif
  trace(ctx, "expected");
  ctx.running = false;
  ++ctx.invocations;
  std::printf("REUSABLE_GEMM complete=%u instance=%u\n", ctx.invocations, ctx.instance);
}

extern "C" void esp_gemm_close(int64_t handle) {
  Context &ctx = lookup(handle);
  if (ctx.running) fail("cannot close a running instance");
  std::printf("REUSABLE_GEMM close instance=%u invocations=%u\n", ctx.instance, ctx.invocations);
  Context **link = &live;
  while (*link != &ctx) link = &(*link)->next;
  *link = ctx.next;
#ifdef __riscv
  iowrite32(ctx.device, PT_NCHUNK_REG, 0);
  release(ctx.pageTable);
#endif
  release(ctx.arena);
  release(&ctx);
}
