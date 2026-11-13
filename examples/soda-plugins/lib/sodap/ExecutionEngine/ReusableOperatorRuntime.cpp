// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/ExecutionEngine/ReusableOperatorABI.h"
#include "sodap/ExecutionEngine/ESPRuntime.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#if defined(SODAP_OP_CPU_ONLY) && (defined(SODAP_OP_EXTERNAL_EXECUTOR) || defined(SODAP_OP_VERIFY_RTL))
#error CPU-only execution cannot be combined with an RTL executor
#endif

#if defined(__riscv) && !defined(SODAP_OP_CPU_ONLY)
#define SODAP_OP_ESP_HW
extern "C" {
#include <esp_accelerator.h>
#include <esp_probe.h>
}
#ifndef SODAP_ESP_DEVICE_ID
#error Define SODAP_ESP_DEVICE_ID and SODAP_ESP_DEVICE_NAME for the operator socket
#endif
#endif

namespace {
#ifdef SODAP_OP_CPU_ONLY
constexpr const char *executionPrefix = "CPU_OP";
#else
constexpr const char *executionPrefix = "REUSABLE_OP";
#endif
// Baremetal software inference must use the cacheable lower DRAM half,
// rather than the uncached DMA allocator. The linker bounds this BSS arena.
#ifdef SODAP_OP_CPU_ARENA_BYTES
#ifndef SODAP_OP_CPU_ONLY
#error A CPU arena requires SODAP_OP_CPU_ONLY
#endif
alignas(64) unsigned char cpuArena[SODAP_OP_CPU_ARENA_BYTES];
size_t cpuArenaUsed = 0;
#endif
#if defined(__riscv) && defined(SODAP_OP_CPU_ONLY) && !defined(SODAP_OP_CPU_ARENA_BYTES)
#error Specify SODAP_OP_CPU_ARENA_BYTES for baremetal CPU execution
#endif
struct Context {
  uint32_t *arena;
  uint32_t bytes, instance, invocations;
  bool running;
  int64_t outputOffset;
  uint64_t outputElements;
  Context *next;
#ifdef SODAP_OP_ESP_HW
  esp_device *devices, *device;
  unsigned **pages;
  bool flush;
#endif
};
Context *live = nullptr;
[[noreturn]] void fail(const char *why) {
  std::printf("REUSABLE_OP TEST FAILED: %s\n", why);
  std::exit(1);
}
void *allocate(uint32_t size) {
#ifdef SODAP_OP_CPU_ARENA_BYTES
  size_t aligned = (size_t(size) + 63) & ~size_t(63);
  if (aligned > sizeof(cpuArena) - cpuArenaUsed) fail("CPU arena exhausted");
  void *p = cpuArena + cpuArenaUsed;
  cpuArenaUsed += aligned;
#elif defined(__riscv)
  void *p = aligned_malloc(size);
#else
  void *p = std::malloc(size);
#endif
  if (!p) fail("allocation failed");
  return p;
}
void release(void *p) {
#ifdef SODAP_OP_CPU_ARENA_BYTES
  (void)p; // Reset collectively when the last context is closed.
#elif defined(__riscv)
  aligned_free(p);
#else
  std::free(p);
#endif
}
Context &lookup(int64_t handle) {
  for (auto *p = live; p; p = p->next)
    if (reinterpret_cast<intptr_t>(p) == handle) return *p;
  fail("invalid context handle");
}
uint64_t product(std::initializer_list<int64_t> dimensions) {
  uint64_t count = 1;
  for (int64_t extent : dimensions) {
    if (extent <= 0 || uint64_t(extent) > (1u << 27) / count)
      fail("invalid or oversized tensor dimensions");
    count *= extent;
  }
  return count;
}
uint64_t check(Context &ctx, int64_t offset, uint64_t elements) {
  if (ctx.running || offset < 0 || offset % OP_SLOT_BYTES ||
      uint64_t(offset) > ctx.bytes - OP_HEADER_BYTES ||
      elements > (ctx.bytes - OP_HEADER_BYTES - offset) / OP_SLOT_BYTES)
    fail("tensor exceeds arena or is misaligned");
  return offset + elements * OP_SLOT_BYTES;
}
void disjoint(int64_t a, uint64_t ae, int64_t c, uint64_t ce) {
  if (uint64_t(a) < ce && uint64_t(c) < ae) fail("output overlaps an input");
}
void flag(int64_t value) {
  if (value != 0 && value != 1) fail("invalid boolean parameter");
}
void spatial(std::initializer_list<int64_t> values, bool allowZero = false) {
  for (auto v : values)
    if (v < (allowZero ? 0 : 1) || v > (1 << 20))
      fail("invalid spatial parameter");
}
uint32_t address(int64_t offset) { return (offset + OP_HEADER_BYTES) / 4; }
float read(Context &ctx, int64_t offset, uint64_t index) {
  float value;
#ifdef SODAP_OP_CPU_ONLY
  // GCC 8 otherwise expands memcpy from the arena into byte loads and stack
  // stores. An explicit aligned word access preserves ordinary float loads.
  uint32_t bits = ctx.arena[address(offset) + index * 2];
  std::memcpy(&value, &bits, 4);
#else
  std::memcpy(&value, ctx.arena + address(offset) + index * 2, 4);
#endif
  return value;
}
void write(Context &ctx, int64_t offset, uint64_t index, float value) {
  auto *p = ctx.arena + address(offset) + index * 2;
#ifdef SODAP_OP_CPU_ONLY
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  p[0] = bits;
#else
  std::memcpy(p, &value, 4);
#endif
  p[1] = 0;
}
float activation(float value, int64_t relu) {
  if (!relu) return value;
  return value < 0 ? 0 : value > FLT_MAX ? FLT_MAX : value;
}
void header(Context &ctx, uint32_t magic) {
  std::memset(ctx.arena, 0, OP_HEADER_BYTES);
  ctx.arena[0] = magic;
  ctx.arena[2] = OP_VERSION;
}
void field(Context &ctx, unsigned index, uint32_t value) { ctx.arena[index * 2] = value; }
bool launch(Context &ctx, const char *kind, int64_t outputOffset, uint64_t outputElements) {
  ctx.running = true;
  ctx.outputOffset = outputOffset;
  ctx.outputElements = outputElements;
#ifdef SODAP_OP_TRACE
  std::printf("OP_CONFIG call=%u",ctx.invocations);
  for(unsigned i=0;i<24;++i) std::printf(" %x",ctx.arena[2*i]);
  std::printf("\n");
#endif
  std::printf("%s launch=%u kind=%s instance=0 socket=%u\n",
              executionPrefix, ctx.invocations, kind, ctx.instance);
#ifndef __riscv
  std::fflush(stdout);
#endif
#ifdef SODAP_OP_ESP_HW
  if (ctx.flush) esp_flush(ACC_COH_NONE);
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  iowrite32(ctx.device, CMD_REG, CMD_MASK_START);
  while (!(ioread32(ctx.device, STATUS_REG) & STATUS_MASK_DONE)) {}
  iowrite32(ctx.device, CMD_REG, 0);
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  return true;
#elif defined(SODAP_OP_EXTERNAL_EXECUTOR) && !defined(SODAP_OP_VERIFY_RTL) && !defined(SODAP_OP_CPU_ONLY)
  esp_op_execute_rtl(ctx.arena, ctx.bytes);
  return true;
#else
  return false;
#endif
}
void complete(Context &ctx) {
#if defined(SODAP_OP_VERIFY_RTL) && !defined(__riscv)
  // In verification mode launch() requests the native oracle first. Preserve
  // its output, poison C, then run RTL. Every subsequent graph node consumes
  // the RTL result; the oracle is used only for the per-invocation assertion.
  auto *expected=static_cast<float *>(allocate(ctx.outputElements*sizeof(float)));
  for(uint64_t i=0;i<ctx.outputElements;++i) {
    expected[i]=read(ctx,ctx.outputOffset,i);
    ctx.arena[address(ctx.outputOffset)+2*i]=0x7fc00000;
  }
  esp_op_execute_rtl(ctx.arena,ctx.bytes);
  float maxError=0;
  for(uint64_t i=0;i<ctx.outputElements;++i) {
    float actual=read(ctx,ctx.outputOffset,i), error=std::fabs(actual-expected[i]);
    if(!std::isfinite(actual) || error>1e-4f+1e-4f*std::fabs(expected[i])) {
      std::printf("RTL mismatch invocation=%u element=%lu actual=%g expected=%g\n",
                  ctx.invocations,(unsigned long)i,double(actual),double(expected[i]));
      fail("RTL output differs from operator oracle");
    }
    maxError=std::max(maxError,error);
  }
  release(expected);
  std::printf("REUSABLE_OP RTL_CHECK elements=%lu max_error=%g\n",
              (unsigned long)ctx.outputElements,double(maxError));
#endif
#ifdef SODAP_OP_TRACE
  std::printf("OP_OUTPUT call=%u",ctx.invocations);
  for(uint64_t i=0;i<std::min(ctx.outputElements,uint64_t(16));++i)
    std::printf(" %x",ctx.arena[address(ctx.outputOffset)+2*i]);
  std::printf("\n");
#endif
  ctx.running = false;
  std::printf("%s complete=%u socket=%u\n", executionPrefix, ++ctx.invocations, ctx.instance);
#ifndef __riscv
  std::fflush(stdout);
#endif
}
void copy(int64_t handle, int64_t offset, int64_t rank, void *desc, bool in) {
  Context &ctx = lookup(handle);
  check(ctx, offset, 0);
  if (!sodap::copyMemRefF32(rank, desc, ctx.arena, ctx.bytes / 4,
                           address(offset), in, 2)) fail("invalid tensor copy");
}
} // namespace

extern "C" int64_t esp_op_open(int64_t bytes, int64_t instance) {
  if (bytes <= 0 || bytes > (1 << 30) - OP_HEADER_BYTES || bytes % 8 ||
      instance < 0 || instance > UINT32_MAX) fail("invalid arena or instance");
  for (auto *p = live; p; p = p->next)
    if (p->instance == instance) fail("socket already owned");
  auto *ctx = static_cast<Context *>(allocate(sizeof(Context)));
  *ctx = {};
  ctx->bytes = bytes + OP_HEADER_BYTES;
  ctx->instance = instance;
#ifdef SODAP_OP_ESP_HW
  int count = probe(&ctx->devices, VENDOR_SLD, SODAP_ESP_DEVICE_ID, SODAP_ESP_DEVICE_NAME);
  if (instance >= count) fail("operator socket not found");
  ctx->device = &ctx->devices[instance];
  unsigned shift = 20;
  unsigned capacity = ioread32(ctx->device, PT_NCHUNK_MAX_REG);
  if (!capacity) fail("missing DMA page table capacity");
  while ((uint64_t(ctx->bytes) + (1u << shift) - 1) >> shift > capacity) ++shift;
  unsigned chunk = 1u << shift, chunks = (uint64_t(ctx->bytes) + chunk - 1) >> shift;
  ctx->arena = static_cast<uint32_t *>(allocate(chunks * chunk));
  ctx->pages = static_cast<unsigned **>(allocate(chunks * sizeof(unsigned *)));
  for (unsigned i = 0; i < chunks; ++i) ctx->pages[i] = ctx->arena + i * (chunk / 4);
  esp_device *cache = nullptr;
  int llc = probe(&cache, VENDOR_CACHE, DEVID_LLC_CACHE, DEVNAME_LLC_CACHE);
  release(cache);
  int l2 = probe(&cache, VENDOR_CACHE, DEVID_L2_CACHE, DEVNAME_L2_CACHE);
  release(cache);
  ctx->flush = llc > 0 || l2 > 0;
  iowrite32(ctx->device, COHERENCE_REG, ACC_COH_NONE);
  iowrite32(ctx->device, PT_ADDRESS_REG, reinterpret_cast<uintptr_t>(ctx->pages));
  iowrite32(ctx->device, PT_NCHUNK_REG, chunks);
  iowrite32(ctx->device, PT_SHIFT_REG, shift);
  iowrite32(ctx->device, SRC_OFFSET_REG, 0);
  iowrite32(ctx->device, DST_OFFSET_REG, 0);
#else
  ctx->arena = static_cast<uint32_t *>(allocate(ctx->bytes));
#endif
  std::memset(ctx->arena, 0, ctx->bytes);
  ctx->next = live;
  live = ctx;
  std::printf("%s open socket=%u arena_bytes=%u engines=conv:1,add:1,gemm:1\n", executionPrefix, ctx->instance, ctx->bytes);
  return reinterpret_cast<intptr_t>(ctx);
}
extern "C" void esp_op_close(int64_t handle) {
  Context &ctx = lookup(handle);
  if (ctx.running) fail("cannot close a running socket");
  std::printf("%s close socket=%u invocations=%u\n", executionPrefix, ctx.instance, ctx.invocations);
  Context **p = &live;
  while (*p != &ctx) p = &(*p)->next;
  *p = ctx.next;
#ifdef SODAP_OP_ESP_HW
  iowrite32(ctx.device, PT_NCHUNK_REG, 0);
  release(ctx.pages);
  release(ctx.devices);
#endif
  release(ctx.arena);
  release(&ctx);
#ifdef SODAP_OP_CPU_ARENA_BYTES
  if (!live) cpuArenaUsed = 0;
#endif
}
extern "C" void esp_op_copy_in(int64_t h, int64_t o, int64_t r, void *d) { copy(h,o,r,d,true); }
extern "C" void esp_op_copy_out(int64_t h, int64_t o, int64_t r, void *d) { copy(h,o,r,d,false); }

extern "C" void esp_op_conv(int64_t handle, int64_t a, int64_t w, int64_t bias, int64_t c,
    int64_t n, int64_t ih, int64_t iw, int64_t ic, int64_t oc, int64_t kh, int64_t kw,
    int64_t oh, int64_t ow, int64_t sh, int64_t sw, int64_t pt, int64_t pl,
    int64_t dh, int64_t dw, int64_t relu) {
  flag(relu);
  esp_op_conv_activation(handle,a,w,bias,c,n,ih,iw,ic,oc,kh,kw,oh,ow,sh,sw,pt,pl,dh,dw,relu,0);
}
extern "C" void esp_op_conv_activation(int64_t handle, int64_t a, int64_t w, int64_t bias, int64_t c,
    int64_t n, int64_t ih, int64_t iw, int64_t ic, int64_t oc, int64_t kh, int64_t kw,
    int64_t oh, int64_t ow, int64_t sh, int64_t sw, int64_t pt, int64_t pl,
    int64_t dh, int64_t dw, int64_t mode, int64_t alpha_bits) {
  if (mode < OP_ACT_NONE || mode > OP_ACT_LEAKY_RELU || alpha_bits < 0 || alpha_bits > UINT32_MAX)
    fail("invalid convolution activation");
  uint32_t bits=uint32_t(alpha_bits);
  float alpha;
  std::memcpy(&alpha,&bits,4);
  if (!std::isfinite(alpha) || alpha < 0 || alpha > 1) fail("invalid LeakyReLU slope");
  Context &ctx = lookup(handle);
  auto ae = check(ctx,a,product({n,ih,iw,ic}));
  auto we = check(ctx,w,product({oc,kh,kw,ic}));
  auto be = check(ctx,bias,product({oc}));
  auto ce = check(ctx,c,product({n,oh,ow,oc}));
  disjoint(a,ae,c,ce); disjoint(w,we,c,ce); disjoint(bias,be,c,ce);
  spatial({sh,sw,dh,dw}); spatial({pt,pl},true);
  header(ctx,OP_CONV_MAGIC);
  if (mode == OP_ACT_LEAKY_RELU) field(ctx,1,CONV_VERSION);
  const int64_t dims[] = {n,ih,iw,ic,oc,kh,kw,oh,ow,sh,sw,pt,pl,dh,dw,mode};
  for (unsigned i=0;i<16;++i) field(ctx,CONV_N+i,dims[i]);
  field(ctx,CONV_A,address(a)); field(ctx,CONV_W,address(w));
  field(ctx,CONV_BIAS,address(bias)); field(ctx,CONV_C,address(c));
  field(ctx,CONV_ALPHA,bits);
  if (!launch(ctx,"conv",c,(ce-c)/8)) {
    for (int64_t batch=0;batch<n;++batch)
      for (int64_t y=0;y<oh;++y)
        for (int64_t x=0;x<ow;++x)
          for (int64_t channel=0;channel<oc;++channel) {
            float sum=0;
            for (int64_t ky=0;ky<kh;++ky) {
              int64_t sy=y*sh+ky*dh-pt;
              for (int64_t kx=0;kx<kw;++kx) {
                int64_t sx=x*sw+kx*dw-pl;
                if (sy<0 || sy>=ih || sx<0 || sx>=iw) continue;
                uint64_t ai=((batch*ih+sy)*iw+sx)*ic;
                uint64_t wi=((channel*kh+ky)*kw+kx)*ic;
                for (int64_t q=0;q<ic;++q) sum=sum+read(ctx,a,ai+q)*read(ctx,w,wi+q);
              }
            }
            float value=sum+read(ctx,bias,channel);
            if (mode == OP_ACT_LEAKY_RELU)
              value=(value < 0 ? 0.0f : value) + (value < 0 ? value : 0.0f)*alpha;
            else value=activation(value,mode);
            write(ctx,c,((batch*oh+y)*ow+x)*oc+channel,value);
          }
  }
  complete(ctx);
}
extern "C" void esp_op_add(int64_t handle, int64_t a, int64_t b, int64_t c,
    int64_t count, int64_t relu) {
  Context &ctx=lookup(handle);
  auto size=product({count}), ae=check(ctx,a,size), be=check(ctx,b,size), ce=check(ctx,c,size);
  disjoint(a,ae,c,ce); disjoint(b,be,c,ce); flag(relu);
  header(ctx,OP_ADD_MAGIC);
  field(ctx,ADD_COUNT,count); field(ctx,ADD_A,address(a)); field(ctx,ADD_B,address(b));
  field(ctx,ADD_C,address(c)); field(ctx,ADD_RELU,relu);
  if (!launch(ctx,"add",c,count))
    for (int64_t i=0;i<count;++i) write(ctx,c,i,activation(read(ctx,a,i)+read(ctx,b,i),relu));
  complete(ctx);
}
extern "C" void esp_op_gemm(int64_t handle, int64_t a, int64_t b, int64_t c,
    int64_t m, int64_t n, int64_t k) {
  Context &ctx=lookup(handle);
  auto ae=check(ctx,a,product({m,k})), be=check(ctx,b,product({k,n})), ce=check(ctx,c,product({m,n}));
  disjoint(a,ae,c,ce); disjoint(b,be,c,ce);
  header(ctx,GEMM_MAGIC);
  field(ctx,GEMM_FIELD_M,m); field(ctx,GEMM_FIELD_N,n); field(ctx,GEMM_FIELD_K,k);
  field(ctx,GEMM_FIELD_A,address(a)); field(ctx,GEMM_FIELD_B,address(b)); field(ctx,GEMM_FIELD_C,address(c));
  field(ctx,GEMM_FIELD_LDA,k*2); field(ctx,GEMM_FIELD_LDB,n*2); field(ctx,GEMM_FIELD_LDC,n*2);
  field(ctx,GEMM_FIELD_ALPHA,0x3f800000); field(ctx,GEMM_FIELD_BETA,0);
  if (!launch(ctx,"gemm",c,(ce-c)/8))
    for (int64_t i=0;i<m;++i)
      for (int64_t j=0;j<n;++j) {
        float sum=0;
        for (int64_t q=0;q<k;++q) sum=sum+read(ctx,a,i*k+q)*read(ctx,b,q*n+j);
        write(ctx,c,i*n+j,sum);
      }
  complete(ctx);
}

extern "C" void esp_op_transpose(int64_t handle, int64_t a, int64_t c, int64_t rank,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3,
    int64_t p0, int64_t p1, int64_t p2, int64_t p3) {
  Context &ctx=lookup(handle);
  int64_t dims[]={d0,d1,d2,d3}, perm[]={p0,p1,p2,p3}, sourceStrides[4], outputDims[4];
  if (rank<1 || rank>4) fail("unsupported transpose rank");
  uint64_t count=product({d0,d1,d2,d3});
  auto ae=check(ctx,a,count), ce=check(ctx,c,count); disjoint(a,ae,c,ce);
  unsigned used=0;
  for (unsigned i=0;i<4;++i) {
    if (perm[i]<0 || perm[i]>=4 || (used & (1u<<perm[i]))) fail("invalid permutation");
    used |= 1u<<perm[i]; outputDims[i]=dims[perm[i]];
  }
  sourceStrides[3]=1;
  for (int i=2;i>=0;--i) sourceStrides[i]=sourceStrides[i+1]*dims[i+1];
  for (uint64_t i=0;i<count;++i) {
    uint64_t remainder=i, source=0;
    for (int axis=3;axis>=0;--axis) {
      source+=(remainder%outputDims[axis])*sourceStrides[perm[axis]];
      remainder/=outputDims[axis];
    }
    write(ctx,c,i,read(ctx,a,source));
  }
  std::printf("REUSABLE_OP cpu=transpose\n");
}
extern "C" void esp_op_pool(int64_t handle, int64_t a, int64_t c,
    int64_t n, int64_t ih, int64_t iw, int64_t ch, int64_t oh, int64_t ow,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t pt, int64_t pl, int64_t average) {
  Context &ctx=lookup(handle);
  auto ae=check(ctx,a,product({n,ih,iw,ch})), ce=check(ctx,c,product({n,oh,ow,ch}));
  disjoint(a,ae,c,ce); spatial({kh,kw,sh,sw}); spatial({pt,pl},true); flag(average);
  if (average && (pt || pl)) fail("padded average pooling is unsupported");
  for (int64_t batch=0;batch<n;++batch)
    for (int64_t y=0;y<oh;++y)
      for (int64_t x=0;x<ow;++x)
        for (int64_t channel=0;channel<ch;++channel) {
          float value=average ? 0 : -FLT_MAX;
          for (int64_t ky=0;ky<kh;++ky)
            for (int64_t kx=0;kx<kw;++kx) {
              int64_t sy=y*sh+ky-pt, sx=x*sw+kx-pl;
              if (sy<0 || sy>=ih || sx<0 || sx>=iw) continue;
              float item=read(ctx,a,((batch*ih+sy)*iw+sx)*ch+channel);
              value=average ? value+item : std::max(value,item);
            }
          if (average) value=value/float(kh*kw);
          write(ctx,c,((batch*oh+y)*ow+x)*ch+channel,value);
        }
  std::printf("REUSABLE_OP cpu=%s_pool\n",average?"average":"max");
}
