//===- LoadStore.cpp - The load and store tasks of a dispatch -------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bring the memory accesses of a dispatch of compute tasks into it: one load
// task in front, copying every argument and constant the compute tasks read,
// and the store `buffer-results-to-out-params` leaves after it, moved in.
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "Prefix.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_LOADSTORE
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

using namespace mlir;
using namespace mlir::sodap::dataflow;

namespace {
/// The op writing `use`, or a view of it, if any.
Operation *writer(OpOperand &use) {
  if (auto view = dyn_cast<ViewLikeOpInterface>(use.getOwner())) {
    for (OpOperand &viewUse : view->getUses())
      if (Operation *op = writer(viewUse))
        return op;
    return nullptr;
  }
  return hasEffect<MemoryEffects::Write>(use.getOwner(), use.get())
             ? use.getOwner()
             : nullptr;
}

struct LoadStore
    : public mlir::sodap::dataflow::impl::LoadStoreBase<LoadStore> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Block &entry = func.front();
    auto dispatches = entry.getOps<DispatchOp>();
    if (dispatches.empty())
      return;
    if (!llvm::hasSingleElement(dispatches)) {
      func.emitOpError("expected one dataflow.dispatch");
      return signalPassFailure();
    }
    DispatchOp dispatch = *dispatches.begin();
    Block &body = dispatch.getBody().front();

    // The stores: copies of a dispatch result into an argument, after it.
    SmallVector<memref::CopyOp> stores;
    for (Operation *op = dispatch->getNextNode(); op; op = op->getNextNode())
      if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        auto out = dyn_cast<BlockArgument>(copy.getTarget());
        if (out && out.getOwner() == &entry &&
            copy.getSource().getDefiningOp() == dispatch)
          stores.push_back(copy);
      }
    if (stores.size() > 1) {
      func.emitOpError("expected one output: more aren't supported yet");
      return signalPassFailure();
    }

    // The loads: what each compute task reads from memory, into a buffer of
    // its own.
    auto isMemory = [&](Value value) {
      if (auto arg = dyn_cast<BlockArgument>(value))
        return arg.getOwner() == &entry;
      return isa_and_nonnull<memref::GetGlobalOp>(value.getDefiningOp());
    };
    SmallVector<std::pair<Value, Value>> loads;
    OpBuilder builder = OpBuilder::atBlockBegin(&body);
    for (TaskOp task : llvm::to_vector(body.getOps<TaskOp>())) {
      llvm::SetVector<Value> sources;
      task.walk([&](Operation *op) {
        for (Value operand : op->getOperands())
          if (isa<MemRefType>(operand.getType()) && isMemory(operand))
            sources.insert(operand);
      });
      for (Value source : sources) {
        auto inTask = [&](OpOperand &use) {
          return task->isAncestor(use.getOwner());
        };
        auto type = cast<MemRefType>(source.getType());
        if (!type.hasStaticShape() || !type.getLayout().isIdentity()) {
          task.emitOpError("expected memory with a static shape and the "
                           "identity layout")
                  .attachNote(source.getLoc())
              << "memory";
          return signalPassFailure();
        }
        for (OpOperand &use : source.getUses())
          if (Operation *op = inTask(use) ? writer(use) : nullptr) {
            task.emitOpError("expected only the store to write memory")
                    .attachNote(op->getLoc())
                << "write";
            return signalPassFailure();
          }
        Value buffer = builder.create<memref::AllocOp>(source.getLoc(), type);
        source.replaceUsesWithIf(buffer, inTask);
        loads.push_back({source, buffer});
      }
    }
    if (!loads.empty()) {
      auto load = builder.create<TaskOp>(dispatch.getLoc(), TypeRange());
      load->setAttr("load", builder.getUnitAttr());
      builder.createBlock(&load.getBody());
      for (auto [source, buffer] : loads)
        builder.create<memref::CopyOp>(source.getLoc(), source, buffer);
      builder.create<YieldOp>(dispatch.getLoc());
    }

    for (memref::CopyOp copy : stores) {
      unsigned result = cast<OpResult>(copy.getSource()).getResultNumber();
      Value yielded = dispatch.getYieldOp().getOperand(result);
      builder.setInsertionPoint(dispatch.getYieldOp());
      auto store = builder.create<TaskOp>(copy.getLoc(), TypeRange());
      store->setAttr("store", builder.getUnitAttr());
      builder.createBlock(&store.getBody());
      auto whole = builder.create<memref::CopyOp>(copy.getLoc(), yielded,
                                                  copy.getTarget());
      builder.create<YieldOp>(copy.getLoc());
      copy.erase();

      // A tile per iteration of the writer's prefix: each tile as soon as
      // it's final.
      auto accesses = accessesOf(yielded, dispatch);
      if (failed(accesses))
        return signalPassFailure();
      SmallVector<TaskOp> writers;
      for (auto &[task, list] : *accesses)
        if (llvm::any_of(list, [](const Access &a) { return a.writes; }))
          writers.push_back(task);
      if (writers.size() == 1)
        splitCopy(
            builder, whole,
            prefixOf((*accesses)[writers.front()], yielded, writers.front()));
    }
  }
};
} // namespace
