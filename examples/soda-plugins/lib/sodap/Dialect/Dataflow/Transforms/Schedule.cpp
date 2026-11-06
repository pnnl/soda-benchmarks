//===- Schedule.cpp - One loop nest for the tasks of a dispatch -----------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Give every task of a dispatch the same loop nest, built on demand: the nest
// of the store, where each read of a tile another task writes is preceded by
// that task's nest for the tile, and each read of a load buffer by its load.
// Each task keeps its own ops of the nest (see the pass description).
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "Prefix.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_SCHEDULE
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

using namespace mlir;
using namespace mlir::sodap::dataflow;

namespace {
/// The buffer `value` is a view of, and the views in between, innermost
/// first.
Value rootOf(Value value, SmallVectorImpl<Operation *> *chain = nullptr) {
  while (auto view = value.getDefiningOp<ViewLikeOpInterface>()) {
    if (chain)
      chain->push_back(view);
    value = view.getViewSource();
  }
  return value;
}

/// A tile of a buffer: per dim, its offset and size.
struct Tile {
  Value buffer;
  SmallVector<OpFoldResult> offsets;
  SmallVector<int64_t> sizes;
};

/// The tile `view` is of the buffer it views, through rank-preserving,
/// unit-stride subviews with static sizes. The offsets go through `map`; where
/// two subviews offset a dim, they are added with `builder`, or left null
/// without one.
std::optional<Tile> tileOf(Value view, IRMapping &map, OpBuilder *builder) {
  SmallVector<Operation *> chain;
  Value buffer = rootOf(view, &chain);
  auto type = cast<MemRefType>(buffer.getType());
  if (!type.hasStaticShape())
    return std::nullopt;
  unsigned rank = type.getRank();
  Builder attrs(view.getContext());
  Tile tile{buffer, SmallVector<OpFoldResult>(rank, attrs.getIndexAttr(0)),
            SmallVector<int64_t>(type.getShape())};
  for (Operation *op : llvm::reverse(chain)) {
    auto subview = dyn_cast<memref::SubViewOp>(op);
    if (!subview || subview.getType().getRank() != rank ||
        !llvm::all_of(subview.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; }))
      return std::nullopt;
    for (unsigned d = 0; d < rank; ++d) {
      std::optional<int64_t> size =
          getConstantIntValue(subview.getMixedSizes()[d]);
      if (!size)
        return std::nullopt;
      tile.sizes[d] = *size;
      OpFoldResult offset = subview.getMixedOffsets()[d];
      if (auto value = dyn_cast<Value>(offset))
        offset = map.lookupOrDefault(value);
      OpFoldResult &total = tile.offsets[d];
      if (isConstantIntValue(offset, 0) || !total)
        continue;
      if (isConstantIntValue(total, 0)) {
        total = offset;
      } else if (!builder) {
        total = OpFoldResult();
      } else {
        Location loc = subview.getLoc();
        auto valueOf = [&](OpFoldResult ofr) -> Value {
          if (auto value = dyn_cast<Value>(ofr))
            return value;
          return builder->create<arith::ConstantIndexOp>(
              loc, *getConstantIntValue(ofr));
        };
        total =
            builder->create<arith::AddIOp>(loc, valueOf(total), valueOf(offset))
                .getResult();
      }
    }
  }
  return tile;
}

/// Where a pinned loop of a producer runs: the offset and size of the tile
/// the consumer reads, along the dim the loop steps through.
using Pins = DenseMap<Value, std::pair<OpFoldResult, int64_t>>;

/// Builds the nest of one task, `target`, into a block of its own.
struct Scheduler {
  Scheduler(TaskOp target, TaskOp loadTask, Block *into, DominanceInfo &dom)
      : target(target), loadTask(loadTask), dom(dom),
        builder(target.getContext()) {
    builder.setInsertionPointToEnd(into);
  }

  TaskOp target;
  TaskOp loadTask;
  DominanceInfo &dom;
  /// Load buffer -> the argument or constant it is loaded from.
  DenseMap<Value, Value> sources;
  /// Buffer shared by tasks -> the task writing it, and its writes.
  DenseMap<Value, std::pair<TaskOp, SmallVector<Access>>> writers;
  /// Appends to the target's new body.
  OpBuilder builder;
  IRMapping map;

  bool available(Value value) {
    return map.contains(value) || dom.properlyDominates(value, target);
  }

  Value valueOf(OpFoldResult ofr, Location loc) {
    if (auto value = dyn_cast<Value>(ofr))
      return value;
    return builder.create<arith::ConstantIndexOp>(loc,
                                                  *getConstantIntValue(ofr));
  }

  /// Copies `view` of a load buffer in, at the place of `op`.
  void load(Operation *op, Value view) {
    SmallVector<Operation *> chain;
    Value buffer = rootOf(view, &chain);
    // The same views, of the source.
    IRMapping onSource;
    onSource.map(buffer, sources.lookup(buffer));
    for (Operation *viewOp : llvm::reverse(chain)) {
      for (Value operand : viewOp->getOperands())
        if (!onSource.contains(operand))
          onSource.map(operand, map.lookupOrDefault(operand));
      builder.clone(*viewOp, onSource);
    }
    builder.create<memref::CopyOp>(op->getLoc(), onSource.lookup(view),
                                   map.lookupOrDefault(view));
  }

  /// Emits the nest of the task writing the tile `view`, with the loops
  /// stepping through the written tiles pinned to the tile read.
  LogicalResult produce(Operation *op, Value view) {
    std::optional<Tile> tile = tileOf(view, map, &builder);
    if (!tile)
      return op->emitOpError("expected to read a tile of another task's "
                             "buffer through static, unit-stride subviews");
    auto &[producer, writes] = writers[tile->buffer];
    // The loops whose induction variable is the offset of a write, a tile at
    // a time.
    Pins pins;
    IRMapping identity;
    for (const Access &write : writes) {
      std::optional<Tile> written = tileOf(write.view, identity, nullptr);
      if (!written)
        continue;
      for (auto [d, offset] : llvm::enumerate(written->offsets)) {
        auto iv = dyn_cast_if_present<Value>(offset);
        scf::ForOp loop = iv ? scf::getForInductionVarOwner(iv) : nullptr;
        auto bounds = loop ? boundsOf(loop) : std::nullopt;
        if (bounds && (*bounds)[2] == written->sizes[d])
          pins[iv] = {tile->offsets[d], tile->sizes[d]};
      }
    }
    return walk(producer, &producer.getBody().front(), pins);
  }

  /// Emits `block` of `task`, cloning its loops, and its ops if it's the
  /// target.
  LogicalResult walk(TaskOp task, Block *block, const Pins &pins) {
    for (Operation &op : *block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      // Allocated once, at the start of the target (scheduleTask).
      if (isa<memref::AllocOp>(op) && block == &task.getBody().front())
        continue;
      if (auto loop = dyn_cast<scf::ForOp>(&op)) {
        if (loop.getNumResults())
          return loop.emitOpError("expected a loop with no results");
        Location loc = loop.getLoc();
        Value lb = map.lookupOrDefault(loop.getLowerBound());
        Value ub = map.lookupOrDefault(loop.getUpperBound());
        Value step = map.lookupOrDefault(loop.getStep());
        OpBuilder::InsertionGuard guard(builder);
        if (auto pin = pins.find(loop.getInductionVar()); pin != pins.end()) {
          auto [offset, size] = pin->second;
          int64_t stride = (*boundsOf(loop))[2];
          lb = valueOf(offset, loc);
          if (size == stride) {
            // One iteration: the tile read.
            map.map(loop.getInductionVar(), lb);
            if (failed(walk(task, loop.getBody(), pins)))
              return failure();
            continue;
          }
          if (size % stride)
            return loop.emitOpError("expected the tiles read of the buffer it "
                                    "writes to be made of whole tiles");
          ub = builder.create<arith::AddIOp>(
              loc, lb, builder.create<arith::ConstantIndexOp>(loc, size));
        }
        auto clone = builder.create<scf::ForOp>(loc, lb, ub, step);
        map.map(loop.getInductionVar(), clone.getInductionVar());
        builder.setInsertionPoint(clone.getBody()->getTerminator());
        if (failed(walk(task, loop.getBody(), pins)))
          return failure();
        continue;
      }
      if (op.getNumRegions() == 0 && isMemoryEffectFree(&op)) {
        if (llvm::all_of(op.getOperands(),
                         [&](Value value) { return available(value); }))
          builder.clone(op, map);
        continue;
      }
      for (Value operand : op.getOperands()) {
        if (!isa<MemRefType>(operand.getType()))
          continue;
        Value buffer = rootOf(operand);
        if (sources.count(buffer)) {
          if (target == loadTask)
            load(&op, operand);
        } else if (auto it = writers.find(buffer);
                   it != writers.end() && it->second.first != task) {
          if (failed(produce(&op, operand)))
            return failure();
        }
      }
      if (task != target)
        continue;
      // A clone maps the block arguments of its regions only if they aren't
      // already: drop those of the previous clone.
      op.walk([&](Block *nested) {
        for (BlockArgument arg : nested->getArguments())
          map.erase(arg);
      });
      builder.clone(op, map);
    }
    return success();
  }
};

/// The new body of `target`: its allocations, then the nest from the store.
FailureOr<std::unique_ptr<Block>> scheduleTask(
    TaskOp target, TaskOp loadTask, TaskOp store,
    const DenseMap<Value, Value> &sources,
    const DenseMap<Value, std::pair<TaskOp, SmallVector<Access>>> &writers,
    DominanceInfo &dom) {
  auto body = std::make_unique<Block>();
  Scheduler scheduler(target, loadTask, body.get(), dom);
  scheduler.sources = sources;
  scheduler.writers = writers;
  for (auto alloc : target.getBody().front().getOps<memref::AllocOp>())
    scheduler.builder.clone(*alloc, scheduler.map);
  if (failed(scheduler.walk(store, &store.getBody().front(), Pins())))
    return failure();
  return body;
}

LogicalResult scheduleDispatch(DispatchOp dispatch, DominanceInfo &dom) {
  Block &body = dispatch.getBody().front();
  auto tasks = llvm::to_vector(body.getOps<TaskOp>());
  if (tasks.size() < 2)
    return success();
  TaskOp loadTask = tasks.front();
  // The store, last (sodap-dataflow-load-store).
  TaskOp store = tasks.back();
  Block &loadBody = loadTask.getBody().front();
  DenseMap<Value, Value> sources;
  DenseMap<Value, std::pair<TaskOp, SmallVector<Access>>> writers;
  for (Operation &op : loadBody.without_terminator()) {
    auto copy = dyn_cast<memref::CopyOp>(op);
    if (!copy || !copy.getTarget().getDefiningOp<memref::AllocOp>())
      return op.emitOpError("expected the first task of a dispatch to copy "
                            "whole buffers in (sodap-dataflow-load-store)");
    sources[copy.getTarget()] = copy.getSource();
  }

  // The writer of each buffer shared by tasks, the tasks each task reads
  // from, and the tasks reading each task.
  DenseMap<TaskOp, SmallVector<TaskOp>> producers;
  DenseMap<TaskOp, SetVector<TaskOp>> consumers;
  for (auto alloc : body.getOps<memref::AllocOp>()) {
    Value buffer = alloc.getResult();
    if (sources.count(buffer))
      continue;
    auto accesses = accessesOf(buffer, dispatch);
    if (failed(accesses))
      return failure();
    if (accesses->size() < 2)
      continue;
    SmallVector<TaskOp> writing;
    for (auto &[task, list] : *accesses)
      if (llvm::any_of(list, [](const Access &a) { return a.writes; }))
        writing.push_back(task);
    if (writing.size() != 1)
      return alloc.emitOpError(
          "expected one task writing a buffer shared by tasks");
    SmallVector<Access> writes;
    for (const Access &access : (*accesses)[writing.front()])
      if (access.writes)
        writes.push_back(access);
    writers[buffer] = {writing.front(), writes};
    for (auto &[task, list] : *accesses)
      if (task != writing.front()) {
        producers[task].push_back(writing.front());
        consumers[writing.front()].insert(task);
      }
  }

  // A compute task feeds one task: its tiles, made for each read, go to one
  // reader.
  for (TaskOp task : ArrayRef(tasks).drop_front().drop_back())
    if (consumers.lookup(task).size() > 1)
      return task.emitOpError("expected a compute task read by one task, "
                              "got ")
             << consumers.lookup(task).size();

  // Every compute task feeds the store: the nest only reaches those.
  DenseSet<TaskOp> feeding{store};
  SmallVector<TaskOp> worklist{store};
  while (!worklist.empty())
    for (TaskOp producer : producers.lookup(worklist.pop_back_val()))
      if (feeding.insert(producer).second)
        worklist.push_back(producer);
  for (TaskOp task : ArrayRef(tasks).drop_front().drop_back())
    if (!feeding.contains(task))
      return task.emitOpError("expected a compute task the store depends on");

  // Every new body first, from the old ones; then replace them.
  SmallVector<std::unique_ptr<Block>> bodies;
  for (TaskOp task : tasks) {
    auto body = scheduleTask(task, loadTask, store, sources, writers, dom);
    if (failed(body))
      return failure();
    bodies.push_back(std::move(*body));
  }
  IRRewriter rewriter(dispatch.getContext());
  for (auto [task, body] : llvm::zip(tasks, bodies)) {
    Block &old = task.getBody().front();
    auto ops = llvm::map_to_vector(old.without_terminator(),
                                   [](Operation &op) { return &op; });
    for (Operation *op : llvm::reverse(ops))
      op->erase();
    old.getOperations().splice(old.getTerminator()->getIterator(),
                               body->getOperations());
    // Drop the views and loops left with none of the task's ops.
    (void)runRegionDCE(rewriter, task->getRegions());
  }
  return success();
}

struct Schedule : public mlir::sodap::dataflow::impl::ScheduleBase<Schedule> {
  void runOnOperation() override {
    DominanceInfo &dom = getAnalysis<DominanceInfo>();
    for (DispatchOp dispatch :
         llvm::to_vector(getOperation().front().getOps<DispatchOp>()))
      if (failed(scheduleDispatch(dispatch, dom)))
        return signalPassFailure();
  }
};
} // namespace
