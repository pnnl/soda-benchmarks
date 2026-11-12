#include <limits>
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

#include "Reuse.h"
#include "llvm/ADT/SmallPtrSet.h"

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
/// Where a pinned loop of a producer runs: the offset and size of the tile
/// the consumer reads, along the dim the loop steps through.
using Pins = DenseMap<Value, std::pair<OpFoldResult, int64_t>>;

/// Builds the nest of one task, `target`, into a block of its own.
struct Scheduler {
  Scheduler(TaskOp target, TaskOp loadTask, Block *into, DominanceInfo &dom,
            bool serial = false)
      : target(target), loadTask(loadTask), dom(dom),
        builder(target.getContext()), serial(serial) {
    rootBlock = into;
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
  Block *rootBlock;
  DenseMap<Value, llvm::SmallPtrSet<Block *, 4>> produced;
  bool serial;

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
    // A selected reuse edge is produced once per shared prefix, before the
    // consumer's reuse loop. The consumer retains that aggregate transaction.
    OpBuilder::InsertionGuard guard(builder);
    if (auto dims =
            tile->buffer.getDefiningOp()->getAttrOfType<DenseI64ArrayAttr>(
                "sodap.reuse_dims")) {
      SmallVector<scf::ForOp> retained;
      for (int64_t dim : dims.asArrayRef())
        if (auto value = dyn_cast<Value>(tile->offsets[dim])) {
          auto loop = scf::getForInductionVarOwner(value);
          if (!loop)
            return op->emitOpError("reuse prefix offset is not a loop index");
          retained.push_back(loop);
        }
      Block *scope = rootBlock;
      for (Block *b = builder.getInsertionBlock(); b != rootBlock;
           b = b->getParentOp()->getBlock())
        if (llvm::is_contained(retained,
                               dyn_cast<scf::ForOp>(b->getParentOp()))) {
          scope = b;
          break;
        }
      if (!produced[tile->buffer].insert(scope).second)
        return success();
      if (scope != builder.getInsertionBlock()) {
        Operation *ancestor = scope->findAncestorOpInBlock(
            *builder.getInsertionBlock()->getParentOp());
        builder.setInsertionPoint(ancestor);
      }
      auto shape = cast<MemRefType>(tile->buffer.getType()).getShape();
      for (unsigned d = 0; d < shape.size(); ++d)
        if (!llvm::is_contained(dims.asArrayRef(), int64_t(d))) {
          tile->offsets[d] = builder.getIndexAttr(0);
          tile->sizes[d] = shape[d];
        }
    }
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
          if (serial || target == loadTask)
            load(&op, operand);
        } else if (auto it = writers.find(buffer);
                   it != writers.end() && it->second.first != task) {
          if (failed(produce(&op, operand)))
            return failure();
        }
      }
      if (!serial && task != target)
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

// A sequential LCS schedule has no simultaneously executing tasks. Compact
// each dispatch buffer to the same tile/slab selected for the parallel flow,
// with one physical copy. The reuse admission budget is deliberately unchanged
// so turning off overlap cannot silently select additional cached edges.
LogicalResult compactSerialBuffers(DispatchOp dispatch, TaskOp serialTask) {
  OpBuilder builder(dispatch.getContext());
  for (auto alloc : llvm::to_vector(
           dispatch.getBody().front().getOps<memref::AllocOp>())) {
    auto accesses = accessesOf(alloc, dispatch);
    if (failed(accesses))
      return failure();
    if (accesses->empty())
      continue;
    if (accesses->size() != 1 || !accesses->count(serialTask))
      return alloc.emitOpError("expected one sequential LCS task");
    auto dims = alloc->getAttrOfType<DenseI64ArrayAttr>("sodap.reuse_dims");
    auto shape = alloc->getAttrOfType<DenseI64ArrayAttr>("sodap.reuse_shape");
    auto &list = (*accesses)[serialTask];
    SmallVector<int64_t> footprint;
    if (shape)
      llvm::append_range(footprint, shape.asArrayRef());
    else
      llvm::append_range(footprint,
                         cast<MemRefType>(list.front().view.getType()).getShape());
    // These buffers belong to the one sequential process, not to Bambu's
    // producer/consumer shared-memory protocol at the enclosing dispatch.
    builder.setInsertionPointToStart(&serialTask.getBody().front());
    Value local = builder.create<memref::AllocOp>(
        alloc.getLoc(), MemRefType::get(footprint,
                                       alloc.getType().getElementType()));
    if (shape) {
      local.getDefiningOp()->setAttr("sodap.reuse_dims", dims);
      local.getDefiningOp()->setAttr("sodap.reuse_shape", shape);
    }
    for (const Access &access : list) {
      if (!shape) {
        if (cast<MemRefType>(access.view.getType()).getShape() !=
            ArrayRef<int64_t>(footprint))
          return access.op->emitOpError("inconsistent sequential tile shapes");
        access.op->replaceUsesOfWith(access.view, local);
        continue;
      }
      builder.setInsertionPoint(access.op);
      IRMapping identity;
      auto tile = tileOf(access.view, identity, &builder);
      if (!tile)
        return access.op->emitOpError("cannot rebase sequential reuse access");
      for (int64_t dim : dims.asArrayRef())
        tile->offsets[dim] = builder.getIndexAttr(0);
      SmallVector<OpFoldResult> sizes,
          strides(tile->sizes.size(), builder.getIndexAttr(1));
      for (int64_t size : tile->sizes)
        sizes.push_back(builder.getIndexAttr(size));
      Value view = builder.create<memref::SubViewOp>(
          access.op->getLoc(), local, tile->offsets, sizes, strides);
      access.op->replaceUsesOfWith(access.view, view);
    }
  }
  IRRewriter rewriter(dispatch.getContext());
  (void)runRegionDCE(rewriter, serialTask->getRegions());
  for (auto alloc : llvm::make_early_inc_range(
           dispatch.getBody().front().getOps<memref::AllocOp>()))
    if (alloc.use_empty())
      alloc.erase();
  return success();
}

LogicalResult scheduleDispatch(DispatchOp dispatch, DominanceInfo &dom,
                               uint64_t reuseBudget, bool serial) {
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

  // Infer reuse from the original accesses, before constructing a common
  // schedule. Only regular, statically bounded, single-reader edges qualify.
  // Unproven accesses and oversized footprints keep the existing schedule.
  uint64_t remaining = reuseBudget;
  for (auto alloc : body.getOps<memref::AllocOp>()) {
    Value buffer = alloc.getResult();
    if (!reuseBudget || !writers.count(buffer) || sources.count(buffer))
      continue;
    auto accesses = accessesOf(buffer, dispatch);
    if (failed(accesses) || accesses->size() != 2)
      continue;
    TaskOp producer = writers[buffer].first;
    TaskOp reader;
    for (auto &[task, list] : *accesses)
      if (task != producer)
        reader = task;
    auto &reads = (*accesses)[reader];
    if (reads.size() != 1 || reads.front().writes)
      continue;
    IRMapping identity;
    auto tile = tileOf(reads.front().view, identity, nullptr);
    if (!tile)
      continue;
    llvm::SmallPtrSet<Operation *, 4> indexed;
    bool regular = true;
    auto shape = cast<MemRefType>(buffer.getType()).getShape();
    for (unsigned d = 0; d < shape.size(); ++d) {
      if (auto offset = getConstantIntValue(tile->offsets[d])) {
        regular &= *offset == 0 && tile->sizes[d] == shape[d];
        continue;
      }
      auto value = dyn_cast_if_present<Value>(tile->offsets[d]);
      auto loop = value ? scf::getForInductionVarOwner(value) : nullptr;
      auto bounds = loop ? boundsOf(loop) : std::nullopt;
      if (!bounds || (*bounds)[0] != 0 || (*bounds)[1] != shape[d] ||
          (*bounds)[2] != tile->sizes[d] || shape[d] % tile->sizes[d]) {
        regular = false;
        break;
      }
      indexed.insert(loop);
    }
    bool repeats = false;
    for (Operation *op = reads.front().op->getParentOp(); op != reader;
         op = op->getParentOp()) {
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        auto bounds = boundsOf(loop);
        if (!bounds || (*bounds)[2] <= 0) {
          regular = false;
          break;
        }
        if (!indexed.count(loop) && (*bounds)[1] - (*bounds)[0] > (*bounds)[2])
          repeats = true;
      } else {
        regular = false;
        break;
      }
    }
    if (!regular || !repeats)
      continue;
    Prefix written = prefixOf((*accesses)[producer], buffer, producer);
    Prefix read = prefixOf(reads, buffer, reader);
    sharePrefix(written, read);
    SmallVector<int64_t> dims, footprint(shape);
    for (Step step : written) {
      auto bounds = *boundsOf(step.loop);
      if (bounds[0] != 0 || bounds[1] != shape[step.dim] ||
          shape[step.dim] % bounds[2]) {
        regular = false;
        break;
      }
      dims.push_back(step.dim);
      footprint[step.dim] = bounds[2];
    }
    if (!regular)
      continue;
    auto element = cast<MemRefType>(buffer.getType()).getElementType();
    if (!element.isIntOrFloat())
      continue;
    // Two shared slots plus a conservative allowance for a private accumulator.
    uint64_t bytes = 3 * ((element.getIntOrFloatBitWidth() + 7) / 8);
    for (int64_t size : footprint) {
      if (size <= 0 || uint64_t(size) > std::numeric_limits<uint64_t>::max() / bytes) {
        bytes = 0;
        break;
      }
      bytes *= size;
    }
    auto decision = alloc.emitRemark("reuse_buffer ");
    decision << (bytes && bytes <= remaining ? "retained" : "over_budget")
             << " footprint=[";
    llvm::interleaveComma(footprint, decision);
    decision << "] required_bytes=" << bytes << " remaining_bytes=" << remaining;
    if (!bytes || bytes > remaining)
      continue;
    remaining -= bytes;
    alloc->setAttr("sodap.reuse_dims",
                   DenseI64ArrayAttr::get(dispatch.getContext(), dims));
    alloc->setAttr("sodap.reuse_shape",
                   DenseI64ArrayAttr::get(dispatch.getContext(), footprint));
  }

  if (serial) {
    // The same demand-driven nest, including the same inferred reuse scopes,
    // now retains all tasks' operations in dependency order. DMA copies and
    // compute execute synchronously in one process; there are no inter-task
    // token channels, ping-pong slots, or speculative read issuers.
    auto scheduled = std::make_unique<Block>();
    Scheduler scheduler(store, loadTask, scheduled.get(), dom, true);
    scheduler.sources = sources;
    scheduler.writers = writers;
    for (TaskOp task : tasks)
      for (auto alloc : task.getBody().front().getOps<memref::AllocOp>())
        scheduler.builder.clone(*alloc, scheduler.map);
    if (failed(scheduler.walk(store, &store.getBody().front(), Pins())))
      return failure();
    // Keep the new task after every dispatch allocation. Original compute
    // tasks may be interleaved with the buffers they produce.
    OpBuilder builder(store);
    auto sequential = builder.create<TaskOp>(dispatch.getLoc(), TypeRange());
    sequential->setAttr("serial_lcs", builder.getUnitAttr());
    Block *block = builder.createBlock(&sequential.getBody());
    builder.create<YieldOp>(dispatch.getLoc());
    block->getOperations().splice(block->getTerminator()->getIterator(),
                                 scheduled->getOperations());
    for (TaskOp task : llvm::reverse(tasks))
      task.erase();
    return compactSerialBuffers(dispatch, sequential);
  }

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
  using ScheduleBase::ScheduleBase;
  void runOnOperation() override {
    DominanceInfo &dom = getAnalysis<DominanceInfo>();
    for (DispatchOp dispatch :
         llvm::to_vector(getOperation().front().getOps<DispatchOp>()))
      if (failed(scheduleDispatch(dispatch, dom, reuseBufferBudget, serial)))
        return signalPassFailure();
  }
};
} // namespace
