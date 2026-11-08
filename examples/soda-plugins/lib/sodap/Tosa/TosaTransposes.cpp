//===- TosaTransposes.cpp - Move and cancel TOSA transposes -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// sodap-tosa-sink-transposes and sodap-tosa-hoist-transposes: a model from
// PyTorch in TOSA's layouts (NHWC), with no transposes inside its blocks.
//
//===----------------------------------------------------------------------===//

#include "sodap/Tosa/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

namespace mlir::sodap {
#define GEN_PASS_DEF_TOSASINKTRANSPOSES
#define GEN_PASS_DEF_TOSAHOISTTRANSPOSES
#include "sodap/Tosa/Passes.h.inc"

namespace {
/// The constant permutation of `transpose`, if any.
std::optional<SmallVector<int64_t>> permsOf(tosa::TransposeOp transpose) {
  SmallVector<int64_t> perms;
  if (failed(transpose.getConstantPerms(perms)))
    return std::nullopt;
  return perms;
}

/// `type` with its dims permuted by `perms`, as tosa.transpose permutes them.
RankedTensorType permuted(RankedTensorType type, ArrayRef<int64_t> perms) {
  SmallVector<int64_t> shape;
  for (int64_t d : perms)
    shape.push_back(type.getDimSize(d));
  return RankedTensorType::get(shape, type.getElementType());
}

SmallVector<int64_t> inverse(ArrayRef<int64_t> perms) {
  SmallVector<int64_t> result(perms.size());
  for (auto [i, d] : llvm::enumerate(perms))
    result[d] = i;
  return result;
}

/// `value` transposed by `perms`, a new tosa.const holding them.
Value transpose(OpBuilder &builder, Location loc, Value value,
                ArrayRef<int64_t> perms) {
  auto permsType = RankedTensorType::get({static_cast<int64_t>(perms.size())},
                                         builder.getI32Type());
  SmallVector<int32_t> values(perms.begin(), perms.end());
  Value permsValue = builder.create<tosa::ConstOp>(
      loc, permsType, DenseIntElementsAttr::get(permsType, values));
  return builder.create<tosa::TransposeOp>(
      loc, permuted(cast<RankedTensorType>(value.getType()), perms), value,
      permsValue);
}

/// A transpose of a non-constant value: one the pass moves.
tosa::TransposeOp movable(Value value) {
  auto transpose = value.getDefiningOp<tosa::TransposeOp>();
  if (!transpose || matchPattern(transpose.getInput1(), m_Constant()))
    return nullptr;
  return transpose;
}

/// elementwise(transpose(x), ...) -> transpose(elementwise(x, ...)).
struct SinkTranspose : RewritePattern {
  SinkTranspose(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa_and_nonnull<tosa::TosaDialect>(op->getDialect()) ||
        !op->hasTrait<OpTrait::ResultsBroadcastableShape>() ||
        op->getNumResults() != 1)
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return failure();

    // The permutation, the same for every transposed operand.
    std::optional<SmallVector<int64_t>> perms;
    Value permsValue;
    for (Value operand : op->getOperands()) {
      tosa::TransposeOp transpose = movable(operand);
      if (!transpose)
        continue;
      std::optional<SmallVector<int64_t>> these = permsOf(transpose);
      if (!these || (perms && *perms != *these))
        return failure();
      perms = these;
      permsValue = transpose.getPerms();
    }
    if (!perms)
      return failure();
    // The others: constants of the same rank, transposed back.
    SmallVector<int64_t> back = inverse(*perms);
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      if (tosa::TransposeOp transpose = movable(operand)) {
        operands.push_back(transpose.getInput1());
        continue;
      }
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (!type || type.getRank() != resultType.getRank() ||
          !matchPattern(operand, m_Constant()))
        return failure();
      operands.push_back(transpose(rewriter, op->getLoc(), operand, back));
    }

    Operation *sunk = rewriter.clone(*op);
    sunk->setOperands(operands);
    sunk->getResult(0).setType(permuted(resultType, back));
    rewriter.replaceOpWithNewOp<tosa::TransposeOp>(
        op, resultType, sunk->getResult(0), permsValue);
    return success();
  }
};

struct TosaSinkTransposes : impl::TosaSinkTransposesBase<TosaSinkTransposes> {
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<SinkTranspose>(&getContext());
    tosa::TransposeOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct TosaHoistTransposes
    : impl::TosaHoistTransposesBase<TosaHoistTransposes> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    for (auto func : llvm::to_vector(module.getOps<func::FuncOp>())) {
      if (!func.isPrivate() || func.isExternal())
        continue;
      std::optional<SymbolTable::UseRange> uses =
          SymbolTable::getSymbolUses(func, module);
      if (!uses || llvm::any_of(*uses, [](const SymbolTable::SymbolUse &use) {
            return !isa<func::CallOp>(use.getUser());
          }))
        continue;

      // Arguments read by transposes with one permutation. Other readers get
      // the inverse transpose, in the function: a residual add then reads two
      // values transposed alike, which sodap-tosa-sink-transposes moves on.
      // ponytail: hoists on any transposed read; weigh the reads if a block
      // gets more transposes than it loses.
      SmallVector<std::optional<SmallVector<int64_t>>> argPerms;
      for (BlockArgument arg : func.getArguments()) {
        std::optional<SmallVector<int64_t>> perms;
        bool hoist = true;
        for (Operation *user : arg.getUsers()) {
          auto transpose = dyn_cast<tosa::TransposeOp>(user);
          if (!transpose || transpose.getInput1() != arg)
            continue;
          std::optional<SmallVector<int64_t>> these = permsOf(transpose);
          if (!these || (perms && *perms != *these))
            hoist = false;
          else
            perms = these;
        }
        argPerms.push_back(hoist ? perms : std::nullopt);
      }
      // Results returned straight from a transpose.
      auto ret = cast<func::ReturnOp>(func.getBody().back().getTerminator());
      SmallVector<std::optional<SmallVector<int64_t>>> resultPerms;
      for (Value result : ret.getOperands()) {
        auto transpose = result.getDefiningOp<tosa::TransposeOp>();
        resultPerms.push_back(transpose ? permsOf(transpose) : std::nullopt);
      }
      if (llvm::none_of(argPerms, [](auto &p) { return p.has_value(); }) &&
          llvm::none_of(resultPerms, [](auto &p) { return p.has_value(); }))
        continue;

      // The function.
      OpBuilder builder(func.getContext());
      for (auto [arg, perms] : llvm::zip_equal(func.getArguments(), argPerms)) {
        if (!perms)
          continue;
        Type original = arg.getType();
        arg.setType(permuted(cast<RankedTensorType>(original), *perms));
        builder.setInsertionPointToStart(&func.front());
        SmallVector<OpOperand *> argUses;
        for (OpOperand &use : arg.getUses())
          argUses.push_back(&use);
        Value back;
        for (OpOperand *use : argUses) {
          auto read = dyn_cast<tosa::TransposeOp>(use->getOwner());
          if (read && read.getInput1() == arg) {
            read.getResult().replaceAllUsesWith(arg);
            read.erase();
            continue;
          }
          if (!back)
            back = transpose(builder, func.getLoc(), arg, inverse(*perms));
          use->set(back);
        }
      }
      for (auto [operand, perms] :
           llvm::zip_equal(ret->getOpOperands(), resultPerms))
        if (perms)
          operand.set(
              operand.get().getDefiningOp<tosa::TransposeOp>().getInput1());
      func.setType(builder.getFunctionType(func.front().getArgumentTypes(),
                                           ret.getOperandTypes()));

      // The callers.
      for (const SymbolTable::SymbolUse &use : *uses) {
        auto call = cast<func::CallOp>(use.getUser());
        builder.setInsertionPoint(call);
        SmallVector<Value> operands;
        for (auto [operand, perms] :
             llvm::zip_equal(call.getOperands(), argPerms))
          operands.push_back(
              perms ? transpose(builder, call.getLoc(), operand, *perms)
                    : operand);
        auto hoisted =
            builder.create<func::CallOp>(call.getLoc(), func, operands);
        for (auto [old, result, perms] : llvm::zip_equal(
                 call.getResults(), hoisted.getResults(), resultPerms))
          old.replaceAllUsesWith(
              perms ? transpose(builder, call.getLoc(), result, *perms)
                    : result);
        call.erase();
      }
    }
  }
};
} // namespace

} // namespace mlir::sodap
