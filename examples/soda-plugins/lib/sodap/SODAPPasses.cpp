//===- SODAPPasses.cpp - SODAP passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>
#include <string>

#include "sodap/SODAPPasses.h"

namespace mlir::sodap {
#define GEN_PASS_DEF_SODAPSWITCHBARFOO
#define GEN_PASS_DEF_TAGOPS
#define GEN_PASS_DEF_LINALGRANKREDUCECONTRACTIONS
#define GEN_PASS_DEF_LINALGCONSTANTSTOINDICES
#include "sodap/SODAPPasses.h.inc"

namespace {
/// Element (i, j) of a rank-2 constant is `j - i <pred> k ? c1 : c0`.
struct DiagonalRule {
  arith::CmpIPredicate pred;
  int64_t k;
  TypedAttr c0, c1;
};

/// The rule of `attr`, if every diagonal holds one value and the values along
/// the diagonals form a step (tril, triu) or a single spike (eye).
static std::optional<DiagonalRule> classify(DenseElementsAttr attr) {
  auto type = attr.getType();
  if (attr.isSplat() || type.getRank() != 2)
    return std::nullopt;
  int64_t m = type.getDimSize(0), n = type.getDimSize(1);
  // diag[d + m - 1] is the value on diagonal d = j - i. Real data (weights)
  // fails on one of the first elements.
  SmallVector<Attribute> diag(m + n - 1);
  auto it = attr.value_begin<Attribute>();
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j, ++it) {
      Attribute &slot = diag[j - i + m - 1];
      if (!slot)
        slot = *it;
      else if (slot != *it)
        return std::nullopt;
    }
  SmallVector<int64_t> steps; // t where diag[t] != diag[t - 1]
  for (int64_t t = 1, e = diag.size(); t < e; ++t)
    if (diag[t] != diag[t - 1])
      steps.push_back(t);
  auto at = [&](int64_t t) { return cast<TypedAttr>(diag[t]); };
  if (steps.size() == 1) // d >= k ? c1 : c0
    return DiagonalRule{arith::CmpIPredicate::sge, steps[0] - (m - 1),
                        at(steps[0] - 1), at(steps[0])};
  if (steps.size() == 2 && steps[1] == steps[0] + 1 &&
      diag[steps[0] - 1] == diag[steps[1]]) // d == k ? c1 : c0
    return DiagonalRule{arith::CmpIPredicate::eq, steps[0] - (m - 1),
                        at(steps[1]), at(steps[0])};
  return std::nullopt;
}

/// Drops one diagonal-rule constant from the `ins` of a generic and computes
/// it in the body, from the loop indices.
struct ConstantToIndices : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    for (OpOperand *operand : op.getDpsInputOperands()) {
      DenseElementsAttr attr;
      if (!matchPattern(operand->get(), m_Constant(&attr)))
        continue;
      AffineMap map = op.getMatchingIndexingMap(operand);
      if (map.getNumResults() != 2 || !map.isProjectedPermutation())
        continue;
      std::optional<DiagonalRule> rule = classify(attr);
      if (!rule)
        continue;

      unsigned number = operand->getOperandNumber();
      SmallVector<Value> inputs;
      SmallVector<AffineMap> maps;
      for (OpOperand &other : op->getOpOperands())
        if (other.getOperandNumber() != number) {
          if (op.isDpsInput(&other))
            inputs.push_back(other.get());
          maps.push_back(op.getMatchingIndexingMap(&other));
        }
      auto generic = rewriter.create<linalg::GenericOp>(
          op.getLoc(), op->getResultTypes(), inputs, op.getOutputs(), maps,
          op.getIteratorTypesArray());
      rewriter.inlineRegionBefore(op.getRegion(), generic.getRegion(),
                                  generic.getRegion().begin());
      Block *body = generic.getBody();
      Location loc = operand->get().getLoc();
      rewriter.setInsertionPointToStart(body);
      Value i = rewriter.create<linalg::IndexOp>(loc, map.getDimPosition(0));
      Value j = rewriter.create<linalg::IndexOp>(loc, map.getDimPosition(1));
      Value d = rewriter.create<arith::SubIOp>(loc, j, i);
      Value k = rewriter.create<arith::ConstantIndexOp>(loc, rule->k);
      Value cond = rewriter.create<arith::CmpIOp>(loc, rule->pred, d, k);
      Value value = rewriter.create<arith::SelectOp>(
          loc, cond, rewriter.create<arith::ConstantOp>(loc, rule->c1),
          rewriter.create<arith::ConstantOp>(loc, rule->c0));
      rewriter.replaceAllUsesWith(body->getArgument(number), value);
      body->eraseArgument(number);
      rewriter.replaceOp(op, generic->getResults());
      return success();
    }
    return failure();
  }
};

struct LinalgConstantsToIndices
    : impl::LinalgConstantsToIndicesBase<LinalgConstantsToIndices> {
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantToIndices>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace {
struct LinalgRankReduceContractions
    : impl::LinalgRankReduceContractionsBase<LinalgRankReduceContractions> {
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    linalg::populateContractionOpRankReducingPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

namespace {
class SODAPSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class SODAPSwitchBarFoo
    : public impl::SODAPSwitchBarFooBase<SODAPSwitchBarFoo> {
public:
  using impl::SODAPSwitchBarFooBase<SODAPSwitchBarFoo>::SODAPSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<SODAPSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace

class SODAPTagOps : public impl::TagOpsBase<SODAPTagOps> {
public:
  using impl::TagOpsBase<SODAPTagOps>::TagOpsBase;
  void runOnOperation() final {
    unsigned counter = 0;
    getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() == anchorOp) {
        std::string uid = prefix + "_" + std::to_string(counter++);
        op->setAttr("uid", StringAttr::get(op->getContext(), uid));
      }
    });
  }
};
} // namespace mlir::sodap
