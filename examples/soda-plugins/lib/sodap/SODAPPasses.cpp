//===- SODAPPasses.cpp - SODAP passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <string>

#include "sodap/SODAPPasses.h"

namespace mlir::sodap {
#define GEN_PASS_DEF_SODAPSWITCHBARFOO
#define GEN_PASS_DEF_TAGOPS
#include "sodap/SODAPPasses.h.inc"

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

class SODAPTagOps
    : public impl::TagOpsBase<SODAPTagOps> {
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
