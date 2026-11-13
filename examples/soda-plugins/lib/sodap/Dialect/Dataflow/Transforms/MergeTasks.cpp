//===- MergeTasks.cpp - The tasks of a dispatch as one, in order ----------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Moves the bodies of the tasks of a dispatch, in order, into its last task,
// so they run one after the other (see the pass description).
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_MERGETASKS
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

using namespace mlir;
using namespace mlir::sodap::dataflow;

namespace {
struct MergeTasks
    : public mlir::sodap::dataflow::impl::MergeTasksBase<MergeTasks> {
  void runOnOperation() override {
    SmallVector<DispatchOp> dispatches;
    getOperation().walk([&](DispatchOp op) { dispatches.push_back(op); });
    for (DispatchOp dispatch : dispatches) {
      auto tasks = llvm::to_vector(dispatch.getOps<TaskOp>());
      if (tasks.size() < 2)
        continue;
      for (TaskOp task : tasks)
        if (task.getNumResults()) {
          task.emitOpError("expected a task with no results");
          return signalPassFailure();
        }
      TaskOp last = tasks.back();
      Block &into = last.getBody().front();
      // Before the last task's own ops; each splice lands after the previous.
      Block::iterator pos = into.begin();
      for (TaskOp task : ArrayRef(tasks).drop_back()) {
        auto &ops = task.getBody().front().getOperations();
        into.getOperations().splice(pos, ops, ops.begin(),
                                    std::prev(ops.end()));
        task.erase();
      }
      // The buffers go in the task, local to its node: Bambu takes an alloca
      // of the dataflow top for a ping-pong channel between two modules.
      Operation *first = &into.front();
      for (auto alloc : llvm::make_early_inc_range(
               dispatch.getBody().front().getOps<memref::AllocOp>()))
        if (llvm::all_of(alloc->getUsers(), [&](Operation *user) {
              return last->isAncestor(user);
            }))
          alloc->moveBefore(first);
      for (StringRef role : {"load", "compute", "store"})
        last->removeAttr(role);
    }
  }
};
} // namespace
