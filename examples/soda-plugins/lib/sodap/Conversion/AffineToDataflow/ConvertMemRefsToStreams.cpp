//===- ConvertMemRefsToStreams.cpp - Buffers to stream channels -----------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace intermediate buffers with dataflow stream channels, so that the
// stages that produce and consume them can run concurrently.
//
// A buffer qualifies when exactly one affine store fills it, exactly one
// affine load drains it, and both walk it in the same order. Requiring a
// single producer and consumer is what keeps this pass self-contained: a
// buffer with several readers would first have to be split, and rather than
// attempt that here we simply decline to convert it.
//
// The load and store are rewritten where they stand, so whatever affine.if
// guards them keeps guarding the channel access.
//
//===----------------------------------------------------------------------===//

#include "sodap/Conversion/AffineToDataflow/Passes.h"
#include "sodap/Conversion/AffineToDataflow/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace sodap {
#define GEN_PASS_DEF_CONVERTMEMREFSTOSTREAMS
#include "sodap/Conversion/AffineToDataflow/Passes.h.inc"
} // namespace sodap
} // namespace mlir

#define DEBUG_TYPE "sodap-convert-memrefs-to-streams"

using namespace mlir;
using namespace mlir::sodap;
using namespace mlir::sodap::dataflow;
using namespace mlir::affine;

namespace mlir {
namespace sodap {
namespace {

/// Number of elements a memref holds, or std::nullopt when it is not static.
std::optional<int64_t> getStaticNumElements(MemRefType type) {
  if (!type.hasStaticShape())
    return std::nullopt;
  int64_t size = 1;
  for (auto dim : type.getShape())
    size *= dim;
  return size;
}

struct BufferToStream : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    Value memref = allocOp.getResult();
    auto memrefType = llvm::cast<MemRefType>(memref.getType());

    auto numElements = getStaticNumElements(memrefType);
    if (!numElements)
      return rewriter.notifyMatchFailure(allocOp, "dynamic shape");

    // Classify every user. Anything we do not understand disqualifies the
    // buffer, since converting it would silently drop that access.
    SmallVector<AffineLoadOp, 2> loadOps;
    SmallVector<AffineStoreOp, 2> storeOps;
    SmallVector<memref::DeallocOp, 2> deallocOps;
    for (auto *user : memref.getUsers()) {
      if (auto loadOp = dyn_cast<AffineLoadOp>(user))
        loadOps.push_back(loadOp);
      else if (auto storeOp = dyn_cast<AffineStoreOp>(user))
        storeOps.push_back(storeOp);
      else if (auto deallocOp = dyn_cast<memref::DeallocOp>(user))
        deallocOps.push_back(deallocOp);
      else
        return rewriter.notifyMatchFailure(allocOp, "unsupported user");
    }

    if (loadOps.size() != 1 || storeOps.size() != 1)
      return rewriter.notifyMatchFailure(
          allocOp, "not a single-producer/single-consumer buffer");

    auto loadOp = loadOps.front();
    auto storeOp = storeOps.front();

    // Producer and consumer in the same loop band describe a value carried
    // within one stage, not a channel between two of them.
    auto loadParent = loadOp->getParentOfType<AffineForOp>();
    auto storeParent = storeOp->getParentOfType<AffineForOp>();
    if (loadParent && storeParent && loadParent == storeParent)
      return rewriter.notifyMatchFailure(allocOp,
                                         "producer and consumer share a band");

    // A FIFO can stand in for the buffer only if both sides traverse it in the
    // same order.
    auto storePattern = getMinimalAccessPattern(storeOp);
    auto loadPattern = getMinimalAccessPattern(loadOp);
    if (!storePattern || !loadPattern)
      return rewriter.notifyMatchFailure(allocOp,
                                         "access pattern not analysable");
    if (storePattern != loadPattern ||
        storeOp.getAffineMap() != loadOp.getAffineMap())
      return rewriter.notifyMatchFailure(allocOp, "mismatched access order");

    LLVM_DEBUG(llvm::dbgs()
               << "converting buffer to stream: " << allocOp << "\n");

    // Declare the channel where the buffer used to be.
    rewriter.setInsertionPoint(allocOp);
    auto streamType = StreamType::get(
        rewriter.getContext(), memrefType.getElementType(), *numElements);
    auto streamOp =
        rewriter.create<StreamOp>(allocOp.getLoc(), streamType, *numElements);

    // Rewrite producer and consumer in place, so that whatever affine.if
    // guards them keeps guarding the channel access.
    rewriter.setInsertionPoint(storeOp);
    rewriter.create<StreamWriteOp>(storeOp.getLoc(), streamOp.getChannel(),
                                   storeOp.getValueToStore());
    rewriter.eraseOp(storeOp);

    rewriter.setInsertionPoint(loadOp);
    auto readOp = rewriter.create<StreamReadOp>(
        loadOp.getLoc(), memrefType.getElementType(), streamOp.getChannel());
    rewriter.replaceOp(loadOp, readOp.getResult());

    for (auto deallocOp : deallocOps)
      rewriter.eraseOp(deallocOp);
    rewriter.eraseOp(allocOp);
    return success();
  }
};

struct ConvertMemRefsToStreams
    : public impl::ConvertMemRefsToStreamsBase<ConvertMemRefsToStreams> {
  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(func.getContext());
    patterns.add<BufferToStream>(func.getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
} // namespace sodap
} // namespace mlir
