//===- TasksToNodes.cpp - Lower dataflow tasks to nodes -------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Give each task explicit ports, turning it into a node.
//
// This deliberately stops at the node level: outlining nodes into functions is
// a separate, opt-in pass, so that the structural dataflow operations survive
// for consumers that want to see them.
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_TASKSTONODES
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

#define DEBUG_TYPE "sodap-dataflow-tasks-to-nodes"

using namespace mlir;
using namespace mlir::sodap;
using namespace mlir::sodap::dataflow;

namespace mlir {
namespace sodap {
namespace dataflow {
namespace {

/// Return whether `use` writes the value it refers to.
///
/// The stream ops deliberately declare no memory effects (so that
/// canonicalization cannot reorder or delete them against the channel's
/// implied ordering), which means the effect interface alone would report a
/// stream_write as harmless and misclassify the channel as an input.
bool isWritten(OpOperand &use) {
  if (auto node = dyn_cast<NodeOp>(use.getOwner()))
    return node.getOperandKind(use) == OperandKind::OUTPUT;
  if (auto view = dyn_cast<ViewLikeOpInterface>(use.getOwner()))
    return llvm::any_of(view->getUses(),
                        [](OpOperand &viewUse) { return isWritten(viewUse); });
  return hasEffect<MemoryEffects::Write>(use.getOwner(), use.get()) ||
         isa<StreamWriteOp>(use.getOwner());
}

struct TaskToNode : public OpRewritePattern<TaskOp> {
  using OpRewritePattern<TaskOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TaskOp task,
                                PatternRewriter &rewriter) const override {
    if (task.getNumResults())
      return task.emitOpError("should not yield any results");

    auto isInTask = [&](OpOperand &use) {
      return task->isAncestor(use.getOwner());
    };

    SmallVector<Value, 8> inputs, outputs, params;
    SmallVector<Location, 8> inputLocs, outputLocs, paramLocs;

    // Classify each livein of the task body into a port.
    auto liveins = Liveness(task).getLiveIn(&task.getBody().front());
    for (auto livein : liveins) {
      if (task.getBody().isAncestor(livein.getParentRegion()))
        continue;

      if (isa<MemRefType, StreamType, RankedTensorType>(livein.getType())) {
        LLVM_DEBUG(llvm::dbgs() << "livein: " << livein << "\n");
        auto uses = llvm::make_filter_range(livein.getUses(), isInTask);
        if (llvm::any_of(uses, [](OpOperand &use) { return isWritten(use); })) {
          outputs.push_back(livein);
          outputLocs.push_back(livein.getLoc());
        } else {
          inputs.push_back(livein);
          inputLocs.push_back(livein.getLoc());
        }
      } else {
        params.push_back(livein);
        paramLocs.push_back(livein.getLoc());
      }
    }

    rewriter.setInsertionPoint(task);
    auto node = rewriter.create<NodeOp>(task.getLoc(), inputs, outputs, params);
    auto *nodeBlock = rewriter.createBlock(&node.getBody());

    // A node is IsolatedFromAbove, so every port must arrive as a block
    // argument and each in-task use be rewritten to it.
    auto inputArgs = nodeBlock->addArguments(ValueRange(inputs), inputLocs);
    for (auto t : llvm::zip(inputs, inputArgs))
      std::get<0>(t).replaceUsesWithIf(std::get<1>(t), isInTask);

    auto outputArgs = nodeBlock->addArguments(ValueRange(outputs), outputLocs);
    for (auto t : llvm::zip(outputs, outputArgs))
      std::get<0>(t).replaceUsesWithIf(std::get<1>(t), isInTask);

    auto paramArgs = nodeBlock->addArguments(ValueRange(params), paramLocs);
    for (auto t : llvm::zip(params, paramArgs))
      std::get<0>(t).replaceUsesWithIf(std::get<1>(t), isInTask);

    // Move the task body across, dropping its terminator: a node has none.
    auto &nodeOps = nodeBlock->getOperations();
    auto &taskOps = task.getBody().front().getOperations();
    nodeOps.splice(nodeOps.begin(), taskOps, taskOps.begin(),
                   std::prev(taskOps.end()));

    rewriter.eraseOp(task);
    return success();
  }
};

struct TasksToNodes : public impl::TasksToNodesBase<TasksToNodes> {
  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(func.getContext());
    patterns.add<TaskToNode>(func.getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
} // namespace dataflow
} // namespace sodap
} // namespace mlir
