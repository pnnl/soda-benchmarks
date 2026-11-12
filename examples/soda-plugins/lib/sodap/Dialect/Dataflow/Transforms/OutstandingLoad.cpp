//===- OutstandingLoad.cpp - The read requests in a task of their own -----===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Moves the read requests of the load task, the writes of the DMA's request
// stream, into a task of their own, in front of it: a copy of the load task
// with only its loops and those writes. The receiver supplies each descriptor
// when it reaches the corresponding receive loop; no speculative request may
// occupy a serialized DMA engine while the receiver is blocked on compute.
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_OUTSTANDINGLOAD
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

using namespace mlir;
using namespace mlir::sodap::dataflow;

namespace {
struct OutstandingLoad
    : public mlir::sodap::dataflow::impl::OutstandingLoadBase<OutstandingLoad> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<TaskOp> loads;
    func.walk([&](TaskOp task) {
      if (task->hasAttr("load"))
        loads.push_back(task);
    });
    if (loads.empty())
      return;
    if (loads.size() > 1) {
      func.emitOpError("expected one load task");
      return signalPassFailure();
    }
    TaskOp load = loads.front();

    // The read requests: the first argument, a stream only written to.
    Value ctrl = func.getNumArguments() ? func.getArgument(0) : Value();
    auto isRequest = [&](Operation *op) {
      auto write = dyn_cast<StreamWriteOp>(op);
      return write && write.getChannel() == ctrl;
    };
    if (!ctrl || !isa<StreamType>(ctrl.getType()) ||
        !llvm::all_of(ctrl.getUsers(), isRequest)) {
      func.emitOpError("expected the first argument to be the read-request "
                       "stream, only written to");
      return signalPassFailure();
    }

    OpBuilder builder(load);
    Value admitted = builder.create<StreamOp>(
        load.getLoc(), StreamType::get(func.getContext(), builder.getI64Type(), 1));
    auto issue = cast<TaskOp>(builder.clone(*load));
    issue->removeAttr("load");
    issue->setAttr("outstanding_load", builder.getUnitAttr());
    issue.walk([&](StreamWriteOp write) {
      if (!isRequest(write))
        return;
      builder.setInsertionPoint(write);
      Value request = builder.create<StreamReadOp>(
          write.getLoc(), builder.getI64Type(), admitted).getResult();
      write.getValueMutable().assign(request);
    });

    // Keep the loops, the requests and what they compute from; and every
    // block's terminator.
    auto isRoot = [&](Operation *op) {
      return isa<scf::ForOp>(op) || op->hasTrait<OpTrait::IsTerminator>() ||
             isRequest(op);
    };
    llvm::SetVector<Operation *> keep;
    BackwardSliceOptions options;
    options.inclusive = true;
    options.omitBlockArguments = true; // An IV's loop is kept on its own.
    options.filter = [&](Operation *op) { return issue->isProperAncestor(op); };
    issue.walk([&](Operation *op) {
      if (!isRoot(op))
        return;
      if (op->hasTrait<OpTrait::IsTerminator>())
        keep.insert(op);
      else if (isa<scf::ForOp>(op))
        (void)getBackwardSlice(op, &keep, options); // inclusive: keeps op too
      else if (isRequest(op)) {
        // Only loops are kept around it: under anything else it'd be erased.
        assert((isa<scf::ForOp, TaskOp>(op->getParentOp())) &&
               "a request outside a loop nest");
        (void)getBackwardSlice(op, &keep, options);
      } else
        llvm_unreachable("The operation should be a scf.for or a terminator or "
                         "a dataflow.write on the control fifo");
    });
    SmallVector<Operation *> dead;
    issue->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op == issue || keep.contains(op))
        return WalkResult::advance();
      dead.push_back(op);
      return WalkResult::skip();
    });
    for (Operation *op : llvm::reverse(dead))
      op->erase();

    load.walk([&](StreamWriteOp write) {
      if (isRequest(write)) {
        write.getChannelMutable().assign(admitted);
        write->setAttr("sodap.dma_ordered", builder.getUnitAttr());
      }
    });
  }
};
} // namespace
