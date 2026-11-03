//===- CreateDataflowTasks.cpp - Partition a function into tasks ----------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrap a function body in a dataflow dispatch region and group its top-level
// affine loop bands into dataflow tasks.
//
// The partition is a plain walk of the dispatch block in order, one task per
// top-level loop band, so the result is reproducible across runs.
//
//===----------------------------------------------------------------------===//

#include "sodap/Conversion/AffineToDataflow/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace sodap {
#define GEN_PASS_DEF_CREATEDATAFLOWTASKS
#include "sodap/Conversion/AffineToDataflow/Passes.h.inc"
} // namespace sodap
} // namespace mlir

#define DEBUG_TYPE "sodap-create-dataflow-tasks"

using namespace mlir;
using namespace mlir::sodap;
using namespace mlir::sodap::dataflow;

namespace mlir {
namespace sodap {
namespace {

/// Wrap every operation of `block` (except its terminator) in a dispatch op.
///
/// Returns a null op when the block already holds a dispatch, so that running
/// the pass twice is a no-op rather than a nesting.
DispatchOp dispatchBlock(Block *block) {
  if (!block->getOps<DispatchOp>().empty() ||
      !isa<func::FuncOp, affine::AffineForOp>(block->getParentOp()))
    return DispatchOp();

  OpBuilder builder(block, block->begin());
  ValueRange returnValues(block->getTerminator()->getOperands());
  auto loc = builder.getUnknownLoc();
  auto dispatch = builder.create<DispatchOp>(loc, returnValues);

  auto &dispatchBody = dispatch.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&dispatchBody);
  builder.create<YieldOp>(loc, returnValues);

  // Splice everything between the dispatch we just created and the block's
  // terminator into the dispatch body.
  auto &dispatchOps = dispatchBody.getOperations();
  auto &parentOps = block->getOperations();
  dispatchOps.splice(dispatchBody.begin(), parentOps,
                     std::next(parentOps.begin()), std::prev(parentOps.end()));
  block->getTerminator()->setOperands(dispatch.getResults());
  return dispatch;
}

/// Move all stream declarations to the top of `block`, preserving their
/// relative order, so that every task can refer to the channels it uses.
void hoistStreamDecls(Block *block) {
  SmallVector<Operation *> streams;
  for (auto &op : *block)
    if (isa<StreamOp>(op))
      streams.push_back(&op);

  // Walk backwards so that repeatedly moving to the front restores the
  // original order.
  for (auto *op : llvm::reverse(streams))
    op->moveBefore(block, block->begin());
}

/// Collect the buffers that `loop` accesses and that belong exclusively to it,
/// so they can be pulled into the loop's task.
///
/// A buffer shared with another loop must stay in the dispatch region: moving
/// it inside one task would put it out of scope for the other. Running the
/// memref-to-stream conversion first leaves few shared buffers behind, but
/// this pass is also useful on its own, so the check is made regardless.
SetVector<Operation *> getExclusiveBuffers(affine::AffineForOp loop) {
  SetVector<Operation *> buffers;

  auto collect = [&](Value memref) {
    if (isa<BlockArgument>(memref))
      return;
    Operation *defOp = memref.getDefiningOp();
    if (!defOp || !isa<memref::AllocOp, memref::AllocaOp>(defOp))
      return;
    // Only claim the buffer if every one of its users is inside this loop.
    if (llvm::all_of(memref.getUsers(),
                     [&](Operation *user) { return loop->isAncestor(user); }))
      buffers.insert(defOp);
  };

  loop.walk([&](Operation *op) {
    if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op))
      collect(storeOp.getMemRef());
    else if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op))
      collect(loadOp.getMemRef());
  });
  return buffers;
}

struct CreateDataflowTasks
    : public impl::CreateDataflowTasksBase<CreateDataflowTasks> {
  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal())
      return;

    auto dispatch = dispatchBlock(&func.front());
    if (!dispatch) {
      LLVM_DEBUG(llvm::dbgs()
                 << "function already contains a dispatch; skipping\n");
      return;
    }

    Block *body = &dispatch.getBody().front();
    hoistStreamDecls(body);

    OpBuilder builder(func.getContext());
    auto loc = builder.getUnknownLoc();

    // One task per top-level loop band, in block order.
    for (auto &op : llvm::make_early_inc_range(*body)) {
      auto loop = dyn_cast<affine::AffineForOp>(op);
      if (!loop)
        continue;

      auto buffers = getExclusiveBuffers(loop);

      builder.setInsertionPoint(&op);
      auto task = builder.create<TaskOp>(loc, ValueRange({}));
      auto *taskBlock = builder.createBlock(&task.getBody());
      builder.setInsertionPointToStart(taskBlock);
      auto yield = builder.create<YieldOp>(loc, ValueRange({}));

      for (auto *buffer : buffers)
        buffer->moveBefore(yield);
      op.moveBefore(yield);
    }
  }
};

} // namespace
} // namespace sodap
} // namespace mlir
