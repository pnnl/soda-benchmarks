//===- JoinMultipleTiles.cpp - One tile per operator ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// sodap-join-multiple-tiles: the tiles of one operator become one tile whose
// shapes are arguments, launched once per invocation.
//
//===----------------------------------------------------------------------===//

#include "sodap/ExecutionEngine/ReusableGemmABI.h"
#include "sodap/Tosa/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::sodap {
#define GEN_PASS_DEF_JOINMULTIPLETILES
#include "sodap/Tosa/Passes.h.inc"

namespace {
constexpr llvm::StringLiteral kName = "forward_kernel";

/// matmul's tile: (A: ?x?, B: ?x?, M, K, N) -> C: ?x?, through tosa.matmul's
/// batch of 1. The shapes reach the body only through M, K and N.
func::FuncOp buildMatmul(OpBuilder &b, Location loc) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  int64_t dyn = ShapedType::kDynamic;
  auto matrix = RankedTensorType::get({dyn, dyn}, f32);
  auto batch = RankedTensorType::get({1, dyn, dyn}, f32);
  Type index = b.getIndexType();
  auto fn = b.create<func::FuncOp>(
      loc, kName,
      FunctionType::get(ctx, {matrix, matrix, index, index, index}, matrix));
  // The descriptor's fields (ReusableGemmABI.h), argument by argument.
  int64_t fields[] = {GEMM_FIELD_A, GEMM_FIELD_B, GEMM_FIELD_M, GEMM_FIELD_K,
                      GEMM_FIELD_N};
  for (auto [i, field] : llvm::enumerate(fields))
    fn.setArgAttr(i, "sodap.descriptor_field", b.getI64IntegerAttr(field));
  fn.setResultAttr(0, "sodap.descriptor_field",
                   b.getI64IntegerAttr(GEMM_FIELD_C));
  // Each matrix's row stride, in words (sodap-dataflow-copy-to-dma).
  fn.setArgAttr(0, "sodap.descriptor_ld", b.getI64IntegerAttr(GEMM_FIELD_LDA));
  fn.setArgAttr(1, "sodap.descriptor_ld", b.getI64IntegerAttr(GEMM_FIELD_LDB));
  fn.setResultAttr(0, "sodap.descriptor_ld",
                   b.getI64IntegerAttr(GEMM_FIELD_LDC));
  // The arguments holding each matrix's dimensions (sodap-dataflow-bind-dims).
  fn.setArgAttr(0, "sodap.dims", b.getDenseI64ArrayAttr({2, 3})); // M, K
  fn.setArgAttr(1, "sodap.dims", b.getDenseI64ArrayAttr({3, 4})); // K, N
  fn.setResultAttr(0, "sodap.dims", b.getDenseI64ArrayAttr({2, 4})); // M, N

  Block *body = fn.addEntryBlock();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(body);
  Value a = body->getArgument(0), rhs = body->getArgument(1);
  Value m = body->getArgument(2), k = body->getArgument(3),
        n = body->getArgument(4);
  SmallVector<ReassociationIndices> batchOf = {{0, 1}, {2}};
  auto expand = [&](Value v, Value rows, Value cols) {
    return b.create<tensor::ExpandShapeOp>(
        loc, batch, v, batchOf,
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), rows, cols});
  };
  Value product = b.create<tosa::MatMulOp>(loc, batch, expand(a, m, k),
                                           expand(rhs, k, n));
  b.create<func::ReturnOp>(
      loc, b.create<tensor::CollapseShapeOp>(loc, matrix, product, batchOf)
               .getResult());
  return fn;
}

/// A matmul launch's dimensions, M, K and N, from its static operand types.
FailureOr<SmallVector<int64_t>> matmulDims(dataflow::LaunchOp launch) {
  if (launch.getInputs().size() != 2 || launch.getNumResults() != 1)
    return launch.emitOpError("expected a matmul launch: two inputs, one "
                              "result");
  auto a = dyn_cast<RankedTensorType>(launch.getInputs()[0].getType());
  auto b = dyn_cast<RankedTensorType>(launch.getInputs()[1].getType());
  if (!a || !b || a.getRank() != 2 || b.getRank() != 2 ||
      !a.hasStaticShape() || !b.hasStaticShape() ||
      a.getDimSize(1) != b.getDimSize(0))
    return launch.emitOpError("expected static matrices MxK and KxN");
  return SmallVector<int64_t>{a.getDimSize(0), a.getDimSize(1),
                              b.getDimSize(1)};
}

struct JoinMultipleTiles
    : impl::JoinMultipleTilesBase<JoinMultipleTiles> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    llvm::MapVector<StringRef, SmallVector<ModuleOp>> groups;
    for (auto tile : module.getOps<ModuleOp>())
      if (auto op = tile->getAttrOfType<StringAttr>("sodap.operator"))
        groups[op.getValue()].push_back(tile);
    if (groups.empty())
      return;
    // ponytail: one operator, one tile named forward_kernel (Make, Bambu's
    // top, the ESP wrapper); name the tiles after their operator when a
    // second operator comes.
    if (groups.size() != 1 || groups.front().first != "matmul") {
      module.emitError("expected only matmul tiles");
      return signalPassFailure();
    }
    SmallVector<ModuleOp> &tiles = groups.front().second;
    DenseSet<Attribute> names;
    for (ModuleOp tile : tiles) {
      names.insert(tile.getSymNameAttr());
      int matmuls = 0;
      tile.walk([&](Operation *op) {
        if (isa<tosa::MatMulOp>(op))
          ++matmuls;
      });
      if (matmuls != 1) {
        tile.emitError("expected one tosa.matmul in a matmul tile");
        return signalPassFailure();
      }
    }

    OpBuilder b(ctx);
    b.setInsertionPoint(tiles.front());
    Location loc = tiles.front().getLoc();
    auto joined = b.create<ModuleOp>(loc, kName);
    joined->setAttr("sodap.tile", b.getI64IntegerAttr(0));
    joined->setAttr("sodap.tile_dir", b.getStringAttr(("tiles/" + kName).str()));
    joined->setAttr("sodap.operator", b.getStringAttr("matmul"));
    b.setInsertionPointToStart(joined.getBody());
    buildMatmul(b, loc);

    SmallVector<dataflow::LaunchOp> launches;
    module.walk([&](dataflow::LaunchOp l) {
      if (names.contains(l.getKernel().getRootReference()))
        launches.push_back(l);
    });
    SmallVector<Attribute> invocations;
    auto kernel =
        SymbolRefAttr::get(ctx, kName, {FlatSymbolRefAttr::get(ctx, kName)});
    for (dataflow::LaunchOp launch : launches) {
      FailureOr<SmallVector<int64_t>> dims = matmulDims(launch);
      if (failed(dims))
        return signalPassFailure();
      invocations.push_back(b.getDenseI64ArrayAttr(*dims));
      b.setInsertionPoint(launch);
      SmallVector<Value> operands(launch.getInputs());
      for (int64_t d : *dims)
        operands.push_back(b.create<arith::ConstantIndexOp>(launch.getLoc(), d));
      auto joinedLaunch = b.create<dataflow::LaunchOp>(
          launch.getLoc(), launch.getResultTypes(), kernel, operands,
          ValueRange());
      launch.replaceAllUsesWith(joinedLaunch.getResults());
      launch.erase();
    }
    joined->setAttr("sodap.invocations", b.getArrayAttr(invocations));
    for (ModuleOp tile : tiles)
      tile.erase();
  }
};
} // namespace

} // namespace mlir::sodap
