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

using namespace mlir;

namespace mlir::sodap {
#define GEN_PASS_DEF_LINALGBATCHMATMULTOESP
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

  if (marshalInIR) {
    for (Value v : {A, B, C})
      if (!cast<MemRefType>(v.getType()).hasStaticShape())
        return op.emitError("marshal=ir needs static shapes");
  }

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
    SmallVector<utils::IteratorType> iters(rank,
                                           utils::IteratorType::parallel);
    builder.create<linalg::GenericOp>(
        loc, TypeRange{}, ValueRange{in}, ValueRange{out}, maps, iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value v = toToken ? token.pack(b, l, args[0])
                            : token.unpack(b, l, args[0]);
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
  // A's rows are K wide and unpadded; B's rows are N wide but stored Npad apart.
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

} // namespace
} // namespace mlir::sodap
