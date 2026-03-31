//===- ESPPasses.cpp - Replace linalg.batch_matmul with ESP calls -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

#include "sodap/SODAPPasses.h"

/// ESP runtime wrapper function names — mirror the C invocation steps.
static constexpr const char *kEspAllocShared = "esp_alloc_shared";
static constexpr const char *kEspFreeShared = "esp_free_shared";
static constexpr const char *kEspFloat2FixedF32 = "esp_float2fixed_f32";
static constexpr const char *kEspFixed2FloatF32 = "esp_fixed2float_f32";
static constexpr const char *kEspAccelCfgRegs = "esp_accel_cfg_regs";
static constexpr const char *kEspAccelStart = "esp_accel_start";
static constexpr const char *kEspAccelWait = "esp_accel_wait";

using namespace mlir;

namespace mlir::sodap {
#define GEN_PASS_DEF_LINALGBATCHMATMULTOESP
#include "sodap/SODAPPasses.h.inc"

namespace {

/// Look up or create a private function declaration in the module.
static FlatSymbolRefAttr getOrInsertFunc(ModuleOp module, OpBuilder &builder,
                                         StringRef name,
                                         TypeRange resultTypes,
                                         TypeRange argTypes) {
  MLIRContext *ctx = module.getContext();
  auto symRef = SymbolRefAttr::get(ctx, name);
  if (module.lookupSymbol<func::FuncOp>(symRef.getAttr()))
    return symRef;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto funcOp = builder.create<func::FuncOp>(
      module.getLoc(), name,
      FunctionType::get(ctx, argTypes, resultTypes));
  funcOp.setPrivate();
  return symRef;
}

/// Forward-declare all ESP runtime functions in the module.
///
/// Function signatures:
///   esp_alloc_shared(i64 total_bytes) -> memref<*xi8>
///   esp_free_shared(memref<*xi8>) -> ()
///   esp_float2fixed_f32(memref<*xf32>, memref<*xi8>, i64 offset) -> ()
///   esp_fixed2float_f32(memref<*xi8>, i64 offset, memref<*xf32>) -> ()
///   esp_accel_cfg_regs(i64 seq_len, i64 indim, i64 outdim,
///                      i64 off_in, i64 off_w, i64 off_b, i64 off_o) -> ()
///   esp_accel_start() -> ()
///   esp_accel_wait() -> ()
static void declareEspFunctions(ModuleOp module, OpBuilder &builder) {
  MLIRContext *ctx = module.getContext();
  Type i64Ty = IntegerType::get(ctx, 64);
  Type f32Ty = Float32Type::get(ctx);
  Type urMemRefI8 = UnrankedMemRefType::get(IntegerType::get(ctx, 8), 0);
  Type urMemRefF32 = UnrankedMemRefType::get(f32Ty, 0);

  // esp_alloc_shared(i64) -> memref<*xi8>
  getOrInsertFunc(module, builder, kEspAllocShared,
                  /*resultTypes=*/{urMemRefI8}, /*argTypes=*/{i64Ty});

  // esp_free_shared(memref<*xi8>) -> ()
  getOrInsertFunc(module, builder, kEspFreeShared,
                  /*resultTypes=*/{}, /*argTypes=*/{urMemRefI8});

  // esp_float2fixed_f32(memref<*xf32>, memref<*xi8>, i64 offset) -> ()
  getOrInsertFunc(module, builder, kEspFloat2FixedF32,
                  /*resultTypes=*/{},
                  /*argTypes=*/{urMemRefF32, urMemRefI8, i64Ty});

  // esp_fixed2float_f32(memref<*xi8>, i64 offset, memref<*xf32>) -> ()
  getOrInsertFunc(module, builder, kEspFixed2FloatF32,
                  /*resultTypes=*/{},
                  /*argTypes=*/{urMemRefI8, i64Ty, urMemRefF32});

  // esp_accel_cfg_regs(seq_len, indim, outdim, off_in, off_w, off_b, off_o)
  SmallVector<Type, 7> cfgArgTypes(7, i64Ty);
  getOrInsertFunc(module, builder, kEspAccelCfgRegs,
                  /*resultTypes=*/{}, /*argTypes=*/cfgArgTypes);

  // esp_accel_start() -> ()
  getOrInsertFunc(module, builder, kEspAccelStart,
                  /*resultTypes=*/{}, /*argTypes=*/{});

  // esp_accel_wait() -> ()
  getOrInsertFunc(module, builder, kEspAccelWait,
                  /*resultTypes=*/{}, /*argTypes=*/{});
}

/// Replace a single linalg.batch_matmul with the ESP call sequence.
///
/// linalg.batch_matmul semantics:
///   A: <batch x M x K>, B: <batch x K x N>, C: <batch x M x N>
///
/// For each batch element, the generated code:
///   1. Computes sizes and offsets for the shared memory layout
///   2. Allocates shared memory
///   3. Copies A, B to shared memory (float -> fixed-point)
///   4. Configures accelerator registers
///   5. Starts the accelerator
///   6. Waits for completion
///   7. Copies C from shared memory (fixed-point -> float)
///   8. Frees shared memory
static void replaceBatchMatmul(linalg::BatchMatmulOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  MLIRContext *ctx = builder.getContext();
  Type i64Ty = IntegerType::get(ctx, 64);
  Type f32Ty = Float32Type::get(ctx);
  Type urMemRefI8 = UnrankedMemRefType::get(IntegerType::get(ctx, 8), 0);
  Type urMemRefF32 = UnrankedMemRefType::get(f32Ty, 0);

  Value A = op.getInputs()[0]; // batch x M x K
  Value B = op.getInputs()[1]; // batch x K x N
  Value C = op.getOutputs()[0]; // batch x M x N

  // Extract dimensions: A is <batch x M x K>, B is <batch x K x N>
  // For static shapes, use the type directly. For dynamic, use memref.dim.
  auto getDim = [&](Value memref, unsigned idx) -> Value {
    auto mrType = cast<MemRefType>(memref.getType());
    if (!mrType.isDynamicDim(idx)) {
      int64_t size = mrType.getDimSize(idx);
      return builder.create<arith::ConstantOp>(loc,
          IntegerAttr::get(i64Ty, size));
    }
    Value dimIdx = builder.create<arith::ConstantOp>(loc,
        builder.getIndexAttr(idx));
    Value dimVal = builder.create<memref::DimOp>(loc, memref, dimIdx);
    return builder.create<arith::IndexCastOp>(loc, i64Ty, dimVal);
  };

  Value M = getDim(A, 1);
  Value K = getDim(A, 2);
  Value N = getDim(B, 2);

  // Compute sizes: sz_in = M*K, sz_w = K*N, sz_o = M*N
  Value szIn = builder.create<arith::MulIOp>(loc, M, K);
  Value szW = builder.create<arith::MulIOp>(loc, K, N);
  Value szO = builder.create<arith::MulIOp>(loc, M, N);

  // Compute offsets: off_in=0, off_w=sz_in, off_o=off_w+sz_w
  Value offIn = builder.create<arith::ConstantOp>(loc,
      IntegerAttr::get(i64Ty, 0));
  Value offW = szIn;
  Value offO = builder.create<arith::AddIOp>(loc, offW, szW);

  // Total elements = off_o + sz_o; total_bytes = total * 4 (sizeof token_t)
  Value totalElems = builder.create<arith::AddIOp>(loc, offO, szO);
  Value elemSize = builder.create<arith::ConstantOp>(loc,
      IntegerAttr::get(i64Ty, 4));
  Value totalBytes = builder.create<arith::MulIOp>(loc, totalElems, elemSize);

  // Cast operands to unranked memrefs
  Value aUR = builder.create<memref::CastOp>(loc, urMemRefF32, A);
  Value bUR = builder.create<memref::CastOp>(loc, urMemRefF32, B);
  Value cUR = builder.create<memref::CastOp>(loc, urMemRefF32, C);

  // Step 1: Allocate shared memory
  auto allocCall = builder.create<func::CallOp>(
      loc, kEspAllocShared, TypeRange{urMemRefI8}, ValueRange{totalBytes});
  Value mem = allocCall.getResult(0);

  // Step 2: Convert float inputs to fixed-point in shared memory
  builder.create<func::CallOp>(loc, kEspFloat2FixedF32, TypeRange{},
                               ValueRange{aUR, mem, offIn});
  builder.create<func::CallOp>(loc, kEspFloat2FixedF32, TypeRange{},
                               ValueRange{bUR, mem, offW});

  // Step 3: Configure accelerator registers
  // esp_accel_cfg_regs(seq_len=M, indim=K, outdim=N,
  //                    off_in, off_w, off_b=off_o, off_o)
  // Note: no bias for batch_matmul, so off_b == off_o (zero-size bias region)
  builder.create<func::CallOp>(
      loc, kEspAccelCfgRegs, TypeRange{},
      ValueRange{M, K, N, offIn, offW, offO, offO});

  // Step 4: Start accelerator
  builder.create<func::CallOp>(loc, kEspAccelStart, TypeRange{}, ValueRange{});

  // Step 5: Wait for completion
  builder.create<func::CallOp>(loc, kEspAccelWait, TypeRange{}, ValueRange{});

  // Step 6: Convert fixed-point output back to float
  builder.create<func::CallOp>(loc, kEspFixed2FloatF32, TypeRange{},
                               ValueRange{mem, offO, cUR});

  // Step 7: Free shared memory
  builder.create<func::CallOp>(loc, kEspFreeShared, TypeRange{},
                               ValueRange{mem});

  // Erase the original op
  op.erase();
}

class SODAPLinalgBatchMatmulToESP
    : public impl::LinalgBatchMatmulToESPBase<SODAPLinalgBatchMatmulToESP> {
public:
  using impl::LinalgBatchMatmulToESPBase<
      SODAPLinalgBatchMatmulToESP>::LinalgBatchMatmulToESPBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Collect all batch_matmul ops first to avoid modifying while walking.
    SmallVector<linalg::BatchMatmulOp> opsToReplace;
    module.walk([&](linalg::BatchMatmulOp op) {
      opsToReplace.push_back(op);
    });

    if (opsToReplace.empty())
      return;

    // Forward-declare all ESP runtime functions.
    declareEspFunctions(module, builder);

    // Replace each batch_matmul with the ESP call sequence.
    for (auto op : opsToReplace) {
      builder.setInsertionPoint(op);
      replaceBatchMatmul(op, builder);
    }
  }
};

} // namespace
} // namespace mlir::sodap
