//===- ESPPasses.cpp - Replace linalg.batch_matmul with ESP calls -*- C++-*-==//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

#include "sodap/SODAPPasses.h"

/// ESP runtime wrapper function names — mirror the C invocation steps.
static constexpr const char *kEspAllocShared = "esp_alloc_shared";
static constexpr const char *kEspFreeShared = "esp_free_shared";
static constexpr const char *kEspFloat2FixedF32 = "esp_float2fixed_f32";
static constexpr const char *kEspFixed2FloatF32 = "esp_fixed2float_f32";
static constexpr const char *kEspCopyToSharedF32 = "esp_copy_to_shared_f32";
static constexpr const char *kEspCopyFromSharedF32 = "esp_copy_from_shared_f32";
static constexpr const char *kEspAccelWriteReg = "esp_accel_write_reg";
static constexpr const char *kEspAccelStart = "esp_accel_start";
static constexpr const char *kEspAccelWait = "esp_accel_wait";
static constexpr const char *kEspProfBegin = "esp_prof_begin";
static constexpr const char *kEspProfEnd = "esp_prof_end";

/// Register map of the accelerator this pass targets, sld,ffn_sysc_catapult.
/// These describe the hardware, not the runtime: the runtime writes whatever
/// (offset, value) pairs it is handed. When the accelerator is generated, these
/// become an output of that generation rather than constants here.
///
/// socketgen assigns registers from the <param> order in the accelerator's
/// XML, which is the reverse of its conf_info_t field order; the offsets below
/// were confirmed against socketgen's own output.
static constexpr uint32_t kRegAddrO = 0x40;
static constexpr uint32_t kRegAddrB = 0x44;
static constexpr uint32_t kRegAddrW = 0x48;
static constexpr uint32_t kRegAddrI = 0x4c;
static constexpr uint32_t kRegOutDim = 0x50;
static constexpr uint32_t kRegInDim = 0x54;
static constexpr uint32_t kRegSeqLen = 0x58;

/// Region ids for esp_prof_begin/end. Must match esp_prof.h.
static constexpr uint32_t kProfPack = 1;
static constexpr uint32_t kProfAccel = 2;
static constexpr uint32_t kProfUnpack = 3;
static constexpr uint32_t kProfEpilogue = 4;
static constexpr uint32_t kProfAlloc = 5;
static constexpr uint32_t kProfKernel = 7;

using namespace mlir;

namespace mlir::sodap {
#define GEN_PASS_DEF_KERNELTOESPACCELERATOR
#define GEN_PASS_DEF_PROFILEKERNEL
#define GEN_PASS_DEF_LINALGBATCHMATMULTOESP
#define GEN_PASS_DEF_ORDERKERNELARGUMENTS
#include "sodap/SODAPPasses.h.inc"

namespace {

/// The accelerator's datatype, as the host sees it: what a token is and how a
/// float becomes one and back. With marshal=ir this is the whole of the
/// conversion -- the two bodies below are what the generated loops execute --
/// so an accelerator with another token type (f16, bf16, an int8 with a scale)
/// is a different instance of this struct and nothing else.
///
/// Q16.16 in an i32, truncating toward zero, matching the runtime's
/// static_cast<token_t>(x * 65536).
struct FixedPointToken {
  unsigned fracBits = 16;

  Type tokenType(MLIRContext *ctx) const { return IntegerType::get(ctx, 32); }

  Value pack(OpBuilder &b, Location loc, Value f) const {
    Type f32 = b.getF32Type();
    Value scale = b.create<arith::ConstantOp>(
        loc, FloatAttr::get(f32, double(1ull << fracBits)));
    Value scaled = b.create<arith::MulFOp>(loc, f, scale);
    return b.create<arith::FPToSIOp>(loc, tokenType(b.getContext()), scaled);
  }

  Value unpack(OpBuilder &b, Location loc, Value t) const {
    Type f32 = b.getF32Type();
    Value f = b.create<arith::SIToFPOp>(loc, f32, t);
    Value inv = b.create<arith::ConstantOp>(
        loc, FloatAttr::get(f32, 1.0 / double(1ull << fracBits)));
    return b.create<arith::MulFOp>(loc, f, inv);
  }
};

/// Look up or create a private function declaration in the module.
static FlatSymbolRefAttr getOrInsertFunc(ModuleOp module, OpBuilder &builder,
                                         StringRef name, TypeRange resultTypes,
                                         TypeRange argTypes) {
  MLIRContext *ctx = module.getContext();
  auto symRef = SymbolRefAttr::get(ctx, name);
  if (module.lookupSymbol<func::FuncOp>(symRef.getAttr()))
    return symRef;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto funcOp = builder.create<func::FuncOp>(
      module.getLoc(), name, FunctionType::get(ctx, argTypes, resultTypes));
  funcOp.setPrivate();
  return symRef;
}

/// Forward-declare all ESP runtime functions in the module.
///
/// Memref arguments use the MLIR unranked memref convention: memref<*xf32>
/// becomes (i64 rank, void *descriptor) at the C level, but at the MLIR func
/// level it is a single memref<*xf32> arg.
///
/// Function signatures, marshal=runtime:
///   esp_alloc_shared(i64 total_bytes) -> i64
///   esp_free_shared(i64 handle) -> ()
///   esp_float2fixed_f32(memref<*xf32>, i64 handle, i64 offset, i64 ld) -> ()
///   esp_fixed2float_f32(i64 handle, i64 offset, i64 ld, memref<*xf32>) -> ()
///
/// marshal=ir has no copy functions -- the conversion is generated -- and the
/// buffer is a pointer rather than an opaque handle, because the generated
/// loops index it:
///   esp_alloc_shared(i64 total_bytes) -> memref<i32>
///   esp_free_shared(memref<i32>) -> ()
/// Under the bare-pointer convention a rank-0 memref is a single pointer, so
/// this is the same C function as above; the runtime need not know which mode
/// produced its caller.
///
/// Common to both:
///   esp_accel_write_reg(i32 offset, i32 value) -> ()
///   esp_accel_start() -> ()
///   esp_accel_wait() -> ()
///   esp_prof_begin(i32 id) -> ()      (only with profile=true)
///   esp_prof_end(i32 id) -> ()
///
/// `ld` is the leading dimension of the operand's row in shared memory -- the
/// stride between rows -- so the runtime can lay a K x N operand into a
/// K x Npad region without knowing why the padding exists.
static void declareEspFunctions(ModuleOp module, OpBuilder &builder,
                                bool profile, bool marshalInIR) {
  MLIRContext *ctx = module.getContext();
  Type i32Ty = IntegerType::get(ctx, 32);
  Type i64Ty = IntegerType::get(ctx, 64);
  Type urMemRefF32 = UnrankedMemRefType::get(Float32Type::get(ctx), 0);

  if (marshalInIR) {
    Type bufTy = MemRefType::get({}, FixedPointToken().tokenType(ctx));
    // esp_alloc_shared(i64) -> memref<i32>
    getOrInsertFunc(module, builder, kEspAllocShared,
                    /*resultTypes=*/{bufTy}, /*argTypes=*/{i64Ty});
    // esp_free_shared(memref<i32>) -> ()
    getOrInsertFunc(module, builder, kEspFreeShared,
                    /*resultTypes=*/{}, /*argTypes=*/{bufTy});
  } else {
    // esp_alloc_shared(i64) -> i64
    getOrInsertFunc(module, builder, kEspAllocShared,
                    /*resultTypes=*/{i64Ty}, /*argTypes=*/{i64Ty});

    // esp_free_shared(i64) -> ()
    getOrInsertFunc(module, builder, kEspFreeShared,
                    /*resultTypes=*/{}, /*argTypes=*/{i64Ty});

    // esp_float2fixed_f32(memref<*xf32>, i64, i64, i64) -> ()
    getOrInsertFunc(module, builder, kEspFloat2FixedF32,
                    /*resultTypes=*/{},
                    /*argTypes=*/{urMemRefF32, i64Ty, i64Ty, i64Ty});

    // esp_fixed2float_f32(i64, i64, i64, memref<*xf32>) -> ()
    getOrInsertFunc(module, builder, kEspFixed2FloatF32,
                    /*resultTypes=*/{},
                    /*argTypes=*/{i64Ty, i64Ty, i64Ty, urMemRefF32});
  }

  // esp_accel_write_reg(i32, i32) -> ()
  getOrInsertFunc(module, builder, kEspAccelWriteReg,
                  /*resultTypes=*/{}, /*argTypes=*/{i32Ty, i32Ty});

  // esp_accel_start() -> ()
  getOrInsertFunc(module, builder, kEspAccelStart,
                  /*resultTypes=*/{}, /*argTypes=*/{});

  // esp_accel_wait() -> ()
  getOrInsertFunc(module, builder, kEspAccelWait,
                  /*resultTypes=*/{}, /*argTypes=*/{});

  if (profile) {
    // esp_prof_begin(i32) -> (), esp_prof_end(i32) -> ()
    getOrInsertFunc(module, builder, kEspProfBegin,
                    /*resultTypes=*/{}, /*argTypes=*/{i32Ty});
    getOrInsertFunc(module, builder, kEspProfEnd,
                    /*resultTypes=*/{}, /*argTypes=*/{i32Ty});
  }
}

/// Replace a single linalg.batch_matmul with the ESP call sequence.
///
/// linalg.batch_matmul semantics:
///   A: <batch x M x K>, B: <batch x K x N>, C: <batch x M x N>
///
/// Shared-memory layout, in elements, with Npad = roundup(N, vecLen):
///
///   [ I: M x K | W: K x Npad | B: Npad | O: M x Npad ]
///
/// Two things about it are dictated by the hardware and are the reason the pass
/// owns the layout rather than the runtime:
///   - the accelerator computes whole vecLen-wide output tiles, so N is padded
///     and W and O are laid out with the padded stride;
///   - the bias is re-read once per output tile while the output is being
///     written, so it cannot share a region with O. It stays zero (the
///     allocation is zeroed), which is what a bias-less matmul needs.
///
/// Generated code:
///   1. Allocates shared memory (zeroed)
///   2. Copies A, B to shared memory (float -> fixed-point), with their strides
///   3. Writes the accelerator registers, one call per register
///   4. Starts the accelerator
///   5. Waits for completion
///   6. Copies C from shared memory (fixed-point -> float), with its stride
///   7. Frees shared memory
///
/// Steps 2 and 6 are runtime calls with marshal=runtime. With marshal=ir they
/// are linalg.generic ops over strided views of the shared buffer -- the view
/// is how the layout is expressed: a K x N operand written into a K x Npad
/// region is a memref with row stride Npad, and the padding columns are simply
/// not part of it. Being ordinary linalg, the conversion is then visible to
/// every later transformation: it can be fused with whatever consumes the
/// output, and its element type and body are the token format, not the
/// runtime's.
///
/// In that mode the free moves to the end of the block. Loop fusion may sink
/// the unpack loop into its consumer, past any call in between, and the
/// buffer has to outlive wherever it lands.
///
/// With profile=true, when the block holds exactly one matmul, the code from
/// the unpack to the end of the block is bracketed as the epilogue: what the
/// kernel does with the result after the offload.
static LogicalResult replaceBatchMatmul(linalg::BatchMatmulOp op,
                                        OpBuilder &builder, unsigned vecLen,
                                        bool profile, bool marshalInIR,
                                        bool epilogue) {
  Location loc = op.getLoc();
  MLIRContext *ctx = builder.getContext();
  Type i32Ty = IntegerType::get(ctx, 32);
  Type i64Ty = IntegerType::get(ctx, 64);
  Type urMemRefF32 = UnrankedMemRefType::get(Float32Type::get(ctx), 0);
  FixedPointToken token;

  Value A = op.getInputs()[0];  // batch x M x K
  Value B = op.getInputs()[1];  // batch x K x N
  Value C = op.getOutputs()[0]; // batch x M x N

  // The layout arithmetic is built with createOrFold so that, for static
  // shapes, every value below is an arith.constant and the views can be typed
  // statically; a dynamic N still works in marshal=runtime mode.
  auto c64 = [&](int64_t v) -> Value {
    return builder.create<arith::ConstantOp>(loc, IntegerAttr::get(i64Ty, v));
  };
  auto c32 = [&](int64_t v) -> Value {
    return builder.create<arith::ConstantOp>(loc, IntegerAttr::get(i32Ty, v));
  };
  auto add = [&](Value a, Value b) -> Value {
    return builder.createOrFold<arith::AddIOp>(loc, a, b);
  };
  auto mul = [&](Value a, Value b) -> Value {
    return builder.createOrFold<arith::MulIOp>(loc, a, b);
  };
  auto divu = [&](Value a, Value b) -> Value {
    return builder.createOrFold<arith::DivUIOp>(loc, a, b);
  };
  auto to32 = [&](Value v) -> Value {
    return builder.createOrFold<arith::TruncIOp>(loc, i32Ty, v);
  };
  auto constOf = [&](Value v) -> int64_t {
    auto cst = v.getDefiningOp<arith::ConstantOp>();
    assert(cst && "static shapes fold the layout to constants");
    return cast<IntegerAttr>(cst.getValue()).getInt();
  };

  // Extract dimensions: A is <batch x M x K>, B is <batch x K x N>
  auto getDim = [&](Value memref, unsigned idx) -> Value {
    auto mrType = cast<MemRefType>(memref.getType());
    if (!mrType.isDynamicDim(idx))
      return c64(mrType.getDimSize(idx));
    Value dimIdx =
        builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(idx));
    Value dimVal = builder.create<memref::DimOp>(loc, memref, dimIdx);
    return builder.create<arith::IndexCastOp>(loc, i64Ty, dimVal);
  };

  Value M = getDim(A, 1);
  Value K = getDim(A, 2);
  Value N = getDim(B, 2);

  // Npad = ((N + vecLen - 1) / vecLen) * vecLen.
  Value vl = c64(vecLen);
  Value Npad = mul(divu(add(N, c64(vecLen - 1)), vl), vl);

  // Region sizes. Every base offset must be a whole number of DMA beats; with
  // vecLen even, only the input region (M*K) can be odd, so round it.
  Value szIn = mul(divu(add(mul(M, K), c64(1)), c64(2)), c64(2));
  Value szW = mul(K, Npad);
  Value szB = Npad;
  Value szO = mul(M, Npad);

  // Offsets: contiguous, in the order I, W, B, O.
  Value offIn = c64(0);
  Value offW = szIn;
  Value offB = add(offW, szW);
  Value offO = add(offB, szB);

  // total_bytes = (off_o + sz_o) * 4 (sizeof token_t)
  Value totalBytes = mul(add(offO, szO), c64(4));

  auto profBegin = [&](uint32_t id) {
    if (profile)
      builder.create<func::CallOp>(loc, kEspProfBegin, TypeRange{},
                                   ValueRange{c32(id)});
  };
  auto profEnd = [&](uint32_t id) {
    if (profile)
      builder.create<func::CallOp>(loc, kEspProfEnd, TypeRange{},
                                   ValueRange{c32(id)});
  };
  auto writeReg = [&](uint32_t reg, Value v) {
    builder.create<func::CallOp>(loc, kEspAccelWriteReg, TypeRange{},
                                 ValueRange{c32(reg), v});
  };

  // --- marshal=ir helpers -------------------------------------------------

  /// A view of the shared buffer shaped like `like`, its rows `ld` tokens
  /// apart, starting `offset` tokens in. Rows of one batch follow each other,
  /// so the batch stride is rows * ld.
  auto view = [&](Value base, Value like, Value offset, Value ld) -> Value {
    auto likeTy = cast<MemRefType>(like.getType());
    ArrayRef<int64_t> shape = likeTy.getShape();
    unsigned rank = shape.size();
    SmallVector<int64_t> strides(rank, 1);
    strides[rank - 2] = constOf(ld);
    for (int i = rank - 3; i >= 0; --i)
      strides[i] = strides[i + 1] * shape[i + 1];
    auto layout = StridedLayoutAttr::get(ctx, constOf(offset), strides);
    auto viewTy = MemRefType::get(shape, token.tokenType(ctx), layout);
    return builder.create<memref::ReinterpretCastOp>(
        loc, viewTy, base, constOf(offset), shape, strides);
  };

  /// An elementwise linalg.generic from `in` to `out`, converting each element
  /// with the token format.
  auto convert = [&](Value in, Value out, bool toToken) {
    unsigned rank = cast<MemRefType>(in.getType()).getRank();
    SmallVector<AffineMap> maps(2, builder.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    builder.create<linalg::GenericOp>(
        loc, TypeRange{}, ValueRange{in}, ValueRange{out}, maps, iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value v =
              toToken ? token.pack(b, l, args[0]) : token.unpack(b, l, args[0]);
          b.create<linalg::YieldOp>(l, v);
        });
  };

  // The runtime's copy functions take unranked memrefs.
  Value aUR, bUR, cUR;
  if (!marshalInIR) {
    aUR = builder.create<memref::CastOp>(loc, urMemRefF32, A);
    bUR = builder.create<memref::CastOp>(loc, urMemRefF32, B);
    cUR = builder.create<memref::CastOp>(loc, urMemRefF32, C);
  }

  // Step 1: Allocate shared memory
  Type memTy = marshalInIR ? MemRefType::get({}, token.tokenType(ctx)) : i64Ty;
  Value mem = builder
                  .create<func::CallOp>(loc, kEspAllocShared, TypeRange{memTy},
                                        ValueRange{totalBytes})
                  .getResult(0);

  // Step 2: Convert float inputs to fixed-point in shared memory.
  // A's rows are K wide and unpadded; B's rows are N wide but stored Npad
  // apart.
  profBegin(kProfPack);
  if (marshalInIR) {
    convert(A, view(mem, A, offIn, K), /*toToken=*/true);
    convert(B, view(mem, B, offW, Npad), /*toToken=*/true);
  } else {
    builder.create<func::CallOp>(loc, kEspFloat2FixedF32, TypeRange{},
                                 ValueRange{aUR, mem, offIn, K});
    builder.create<func::CallOp>(loc, kEspFloat2FixedF32, TypeRange{},
                                 ValueRange{bUR, mem, offW, Npad});
  }
  profEnd(kProfPack);

  // Step 3: Accelerator registers. The register map lives here, not in the
  // runtime; the values are the padded layout computed above.
  writeReg(kRegSeqLen, to32(M));
  writeReg(kRegInDim, to32(K));
  writeReg(kRegOutDim, to32(Npad));
  writeReg(kRegAddrI, to32(offIn));
  writeReg(kRegAddrW, to32(offW));
  writeReg(kRegAddrB, to32(offB));
  writeReg(kRegAddrO, to32(offO));

  // Step 4/5: Start, wait
  profBegin(kProfAccel);
  builder.create<func::CallOp>(loc, kEspAccelStart, TypeRange{}, ValueRange{});
  builder.create<func::CallOp>(loc, kEspAccelWait, TypeRange{}, ValueRange{});
  profEnd(kProfAccel);

  // Step 6: Convert fixed-point output back to float, dropping the padding.
  profBegin(kProfUnpack);
  if (marshalInIR) {
    convert(view(mem, C, offO, Npad), C, /*toToken=*/false);
  } else {
    builder.create<func::CallOp>(loc, kEspFixed2FloatF32, TypeRange{},
                                 ValueRange{mem, offO, Npad, cUR});
  }
  profEnd(kProfUnpack);

  // Step 7: Free shared memory -- here, or at the end of the block (see above).
  auto freeShared = [&]() {
    builder.create<func::CallOp>(loc, kEspFreeShared, TypeRange{},
                                 ValueRange{mem});
  };
  if (!marshalInIR)
    freeShared();

  if (epilogue)
    profBegin(kProfEpilogue);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op->getBlock()->getTerminator());
    if (epilogue)
      profEnd(kProfEpilogue);
    if (marshalInIR)
      freeShared();
  }

  // Erase the original op
  op.erase();
  return success();
}

/// Does anything in the function write through `value` or a view of it?
static bool isWritten(Value value) {
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
      if (isWritten(view->getResult(0)))
        return true;
      continue;
    }
    // An op that does not say what it does might write.
    auto effects = dyn_cast<MemoryEffectOpInterface>(user);
    if (!effects || effects.getEffectOnValue<MemoryEffects::Write>(value))
      return true;
  }
  return false;
}

class ProfileKernel : public impl::ProfileKernelBase<ProfileKernel> {
public:
  using impl::ProfileKernelBase<ProfileKernel>::ProfileKernelBase;
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto function = module.lookupSymbol<func::FuncOp>(kernel);
    if (!function || function.isExternal()) {
      module.emitError("no function with a body named '") << kernel << "'";
      return signalPassFailure();
    }
    // Reapplying a pipeline must not nest the same profiler region.
    if (function->hasAttr("sodap.kernel_profiled"))
      return;
    OpBuilder builder(module.getContext());
    Type i32 = builder.getI32Type();
    getOrInsertFunc(module, builder, kEspProfBegin, {}, {i32});
    getOrInsertFunc(module, builder, kEspProfEnd, {}, {i32});
    builder.setInsertionPointToStart(&function.getBody().front());
    Value id = builder.create<arith::ConstantIntOp>(function.getLoc(),
                                                    kProfKernel, 32);
    builder.create<func::CallOp>(function.getLoc(), kEspProfBegin, TypeRange{},
                                 ValueRange{id});
    function.walk([&](func::ReturnOp ret) {
      builder.setInsertionPoint(ret);
      builder.create<func::CallOp>(ret.getLoc(), kEspProfEnd, TypeRange{},
                                   ValueRange{id});
    });
    function->setAttr("sodap.kernel_profiled", builder.getUnitAttr());
  }
};

class KernelToESPAccelerator
    : public impl::KernelToESPAcceleratorBase<KernelToESPAccelerator> {
public:
  using impl::KernelToESPAcceleratorBase<
      KernelToESPAccelerator>::KernelToESPAcceleratorBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (dmaWordStride != 1 && dmaWordStride != 2) {
      module.emitError("dma-word-stride must be 1 or 2");
      return signalPassFailure();
    }
    const char *copyIn = dmaWordStride == 1 ? kEspCopyToSharedF32
                                            : "esp_copy_to_shared_f32_strided";
    const char *copyOut = dmaWordStride == 1
                              ? kEspCopyFromSharedF32
                              : "esp_copy_from_shared_f32_strided";
    auto function = module.lookupSymbol<func::FuncOp>(kernel);
    if (!function || function.isExternal()) {
      module.emitError("no function with a body named '") << kernel << "'";
      return signalPassFailure();
    }
    if (function.getNumResults() != 0) {
      function.emitError("expected results as arguments; run "
                         "buffer-results-to-out-params first");
      return signalPassFailure();
    }

    struct Argument {
      Value value;
      int64_t offset;
      bool written;
    };
    SmallVector<Argument> arguments;
    int64_t words = 0;
    for (BlockArgument arg : function.getArguments()) {
      auto type = dyn_cast<MemRefType>(arg.getType());
      if (!type || !type.getElementType().isF32() || !type.hasStaticShape() ||
          !type.getLayout().isIdentity()) {
        function.emitError("expected f32 memref arguments with static shapes "
                           "and the identity layout, got ")
            << arg.getType();
        return signalPassFailure();
      }
      arguments.push_back({arg, words, isWritten(arg)});
      words += type.getNumElements() * dmaWordStride;
    }

    MLIRContext *ctx = module.getContext();
    Location loc = function.getLoc();
    OpBuilder builder(ctx);
    Type i64Ty = builder.getI64Type();
    Type urMemRefF32 = UnrankedMemRefType::get(builder.getF32Type(), 0);
    getOrInsertFunc(module, builder, kEspAllocShared, {i64Ty}, {i64Ty});
    getOrInsertFunc(module, builder, kEspFreeShared, {}, {i64Ty});
    SmallVector<Type> inTypes{urMemRefF32, i64Ty, i64Ty};
    SmallVector<Type> outTypes{i64Ty, i64Ty, urMemRefF32};
    if (dmaWordStride != 1) {
      inTypes.push_back(i64Ty);
      outTypes.push_back(i64Ty);
    }
    getOrInsertFunc(module, builder, copyIn, {}, inTypes);
    getOrInsertFunc(module, builder, copyOut, {}, outTypes);
    getOrInsertFunc(module, builder, kEspAccelStart, {}, {});
    getOrInsertFunc(module, builder, kEspAccelWait, {}, {});
    if (profile) {
      getOrInsertFunc(module, builder, kEspProfBegin, {},
                      {builder.getI32Type()});
      getOrInsertFunc(module, builder, kEspProfEnd, {}, {builder.getI32Type()});
    }

    // The old body goes, and the entry block keeps its arguments. soda-opt's
    // outliner leaves a branch to a second block.
    Region &body = function.getBody();
    Block &entry = body.front();
    body.dropAllReferences();
    while (&body.back() != &entry)
      body.back().erase();
    while (!entry.empty())
      entry.back().erase();
    builder.setInsertionPointToEnd(&entry);

    auto constant = [&](int64_t value) -> Value {
      return builder.create<arith::ConstantOp>(
          loc, i64Ty, builder.getI64IntegerAttr(value));
    };
    auto mark = [&](const char *callee, uint32_t region) {
      if (!profile)
        return;
      Value id = builder.create<arith::ConstantIntOp>(loc, region, 32);
      builder.create<func::CallOp>(loc, callee, TypeRange{}, ValueRange{id});
    };
    mark(kEspProfBegin, kProfAlloc);
    Value handle =
        builder
            .create<func::CallOp>(loc, kEspAllocShared, TypeRange{i64Ty},
                                  ValueRange{constant(words * 4)})
            .getResult(0);
    mark(kEspProfEnd, kProfAlloc);
    mark(kEspProfBegin, kProfPack);
    for (const Argument &arg : arguments) {
      Value cast = builder.create<memref::CastOp>(loc, urMemRefF32, arg.value);
      SmallVector<Value> args{cast, handle, constant(arg.offset)};
      if (dmaWordStride != 1)
        args.push_back(constant(dmaWordStride));
      builder.create<func::CallOp>(loc, copyIn, TypeRange{}, args);
    }
    mark(kEspProfEnd, kProfPack);
    mark(kEspProfBegin, kProfAccel);
    builder.create<func::CallOp>(loc, kEspAccelStart, TypeRange{},
                                 ValueRange{});
    builder.create<func::CallOp>(loc, kEspAccelWait, TypeRange{}, ValueRange{});
    mark(kEspProfEnd, kProfAccel);
    mark(kEspProfBegin, kProfUnpack);
    for (const Argument &arg : arguments) {
      if (!arg.written)
        continue;
      Value cast = builder.create<memref::CastOp>(loc, urMemRefF32, arg.value);
      SmallVector<Value> args{handle, constant(arg.offset), cast};
      if (dmaWordStride != 1)
        args.push_back(constant(dmaWordStride));
      builder.create<func::CallOp>(loc, copyOut, TypeRange{}, args);
    }
    mark(kEspProfEnd, kProfUnpack);
    builder.create<func::CallOp>(loc, kEspFreeShared, TypeRange{},
                                 ValueRange{handle});
    builder.create<func::ReturnOp>(loc);
  }
};

class SODAPLinalgBatchMatmulToESP
    : public impl::LinalgBatchMatmulToESPBase<SODAPLinalgBatchMatmulToESP> {
public:
  using impl::LinalgBatchMatmulToESPBase<
      SODAPLinalgBatchMatmulToESP>::LinalgBatchMatmulToESPBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    bool marshalInIR;
    if (marshal == "runtime")
      marshalInIR = false;
    else if (marshal == "ir")
      marshalInIR = true;
    else {
      module.emitError("marshal must be 'runtime' or 'ir', got '")
          << marshal << "'";
      return signalPassFailure();
    }

    // Collect all batch_matmul ops first to avoid modifying while walking.
    SmallVector<linalg::BatchMatmulOp> opsToReplace;
    module.walk([&](linalg::BatchMatmulOp op) { opsToReplace.push_back(op); });

    if (opsToReplace.empty())
      return;

    // The epilogue region is only meaningful when one matmul owns the block.
    DenseMap<Block *, unsigned> perBlock;
    for (auto op : opsToReplace)
      ++perBlock[op->getBlock()];

    // Validate the whole set before introducing declarations or rewriting any
    // operations. The hardware layout has no batch dimension; an unknown batch
    // is not evidence that it is one.
    DenseMap<Operation *, unsigned> perFunction;
    for (auto op : opsToReplace) {
      for (Value operand : op->getOperands()) {
        auto type = dyn_cast<MemRefType>(operand.getType());
        if (!type || type.getRank() != 3 || !type.getElementType().isF32()) {
          op.emitError("ESP lowering requires rank-3 f32 memref operands");
          return signalPassFailure();
        }
        if (type.getDimSize(0) != 1) {
          op.emitError(
              "ESP lowering requires a statically known batch size of 1 "
              "for every operand");
          return signalPassFailure();
        }
        if (marshalInIR && !type.hasStaticShape()) {
          op.emitError("marshal=ir needs static shapes");
          return signalPassFailure();
        }
      }

      if (marshalInIR) {
        // IR marshalling keeps the buffer until the block terminator so fusion
        // can move its consumers. The runtime owns only one live buffer. Until
        // lifetimes are tracked, conservatively reject multiple offloads in a
        // function, including those in nested or different blocks.
        auto function = op->getParentOfType<func::FuncOp>();
        if (!function || ++perFunction[function.getOperation()] > 1) {
          op.emitError(
              "marshal=ir supports at most one ESP offload per function; "
              "use marshal=runtime for sequential offloads");
          return signalPassFailure();
        }
      }
    }

    // Forward-declare all ESP runtime functions.
    declareEspFunctions(module, builder, profile, marshalInIR);

    // Replace each batch_matmul with the ESP call sequence.
    for (auto op : opsToReplace) {
      bool epilogue = profile && perBlock[op->getBlock()] == 1;
      builder.setInsertionPoint(op);
      if (failed(replaceBatchMatmul(op, builder, vecLen, profile, marshalInIR,
                                    epilogue)))
        return signalPassFailure();
    }
  }
};

/// The soda ops by name: the plugin doesn't link the soda dialect.
class OrderKernelArguments
    : public impl::OrderKernelArgumentsBase<OrderKernelArguments> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult result = module.walk([&](Operation *launch) {
      if (launch->getName().getStringRef() != "soda.launch_func")
        return WalkResult::advance();
      return succeeded(order(module, launch)) ? WalkResult::advance()
                                              : WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }

private:
  /// The op named `name` with symbol `symbol` under `scope`, if any.
  static Operation *find(Operation *scope, StringRef name, StringAttr symbol) {
    Operation *found = nullptr;
    scope->walk([&](Operation *op) {
      if (op->getName().getStringRef() == name &&
          op->getAttrOfType<StringAttr>("sym_name") == symbol) {
        found = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return found;
  }

  LogicalResult order(ModuleOp module, Operation *launch) {
    auto ref = launch->getAttrOfType<SymbolRefAttr>("kernel");
    Operation *container =
        ref ? find(module, "soda.module", ref.getRootReference()) : nullptr;
    Operation *kernel =
        container ? find(container, "soda.func", ref.getLeafReference())
                  : nullptr;
    if (!kernel || kernel->getNumRegions() != 1 || kernel->getRegion(0).empty())
      return launch->emitOpError("expected the kernel's soda.func");
    if (auto sizes =
            launch->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
        sizes && sizes.asArrayRef().front() != 0)
      return launch->emitOpError("expected no async dependencies");

    // Each operand a distinct argument of the caller: sort by its number.
    Block &caller = launch->getParentOfType<func::FuncOp>().front();
    SmallVector<unsigned> numbers;
    for (Value operand : launch->getOperands()) {
      auto arg = dyn_cast<BlockArgument>(operand);
      if (!arg || arg.getOwner() != &caller ||
          llvm::is_contained(numbers, arg.getArgNumber()))
        return launch->emitOpError(
            "expected distinct arguments of the caller as operands");
      numbers.push_back(arg.getArgNumber());
    }
    auto order = llvm::to_vector(llvm::seq<unsigned>(0, numbers.size()));
    llvm::sort(order,
               [&](unsigned a, unsigned b) { return numbers[a] < numbers[b]; });

    Block &entry = kernel->getRegion(0).front();
    auto type = kernel->getAttrOfType<TypeAttr>("function_type");
    auto function = type ? dyn_cast<FunctionType>(type.getValue()) : nullptr;
    if (!function || entry.getNumArguments() != numbers.size() ||
        function.getNumInputs() != numbers.size())
      return kernel->emitOpError(
          "expected as many arguments as the launch has operands");

    // Permute the launch's operands, the kernel's arguments, their types and
    // attributes alike.
    unsigned n = numbers.size();
    SmallVector<Value> operands;
    SmallVector<Type> inputs;
    for (unsigned i : order) {
      operands.push_back(launch->getOperand(i));
      inputs.push_back(function.getInput(i));
      BlockArgument old = entry.getArgument(i);
      old.replaceAllUsesWith(entry.addArgument(old.getType(), old.getLoc()));
    }
    entry.eraseArguments(0, n);
    launch->setOperands(operands);
    kernel->setAttr("function_type",
                    TypeAttr::get(FunctionType::get(&getContext(), inputs,
                                                    function.getResults())));
    if (auto attrs = kernel->getAttrOfType<ArrayAttr>("arg_attrs")) {
      SmallVector<Attribute> permuted;
      for (unsigned i : order)
        permuted.push_back(attrs[i]);
      kernel->setAttr("arg_attrs", ArrayAttr::get(&getContext(), permuted));
    }
    return success();
  }
};

} // namespace
} // namespace mlir::sodap
