//===- PingPong.cpp - Ping-pong buffers and depth-1 tokens ----------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Turn each buffer shared by the tasks of a dispatch into two slots of the
// tile they exchange, with a depth-1 token stream per reader carrying the
// slot (see the pass description, and "Ping-pong buffers and the load
// schedule" in docs/LinalgToLCSPlan.md).
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "Prefix.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_PINGPONG
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

using namespace mlir;
using namespace mlir::sodap::dataflow;

namespace {
/// Where a task uses a shared buffer a tile at a time: the innermost block
/// holding all its accesses, run once per tile, and the first and last op of
/// that block holding one.
struct Side {
  TaskOp task;
  SmallVector<Access> accesses;
  Block *level;
  Operation *first, *last;
  bool writes = false, reads = false;
};

Side sideOf(TaskOp task, ArrayRef<Access> accesses) {
  Side side{task, SmallVector<Access>(accesses),
            accesses.front().op->getBlock(), nullptr, nullptr};
  for (const Access &access : accesses) {
    while (!side.level->findAncestorOpInBlock(*access.op))
      side.level = side.level->getParentOp()->getBlock();
    side.writes |= access.writes;
    side.reads |= hasEffect<MemoryEffects::Read>(access.op, access.view);
  }
  for (const Access &access : accesses) {
    Operation *op = side.level->findAncestorOpInBlock(*access.op);
    if (!side.first || op->isBeforeInBlock(side.first))
      side.first = op;
    if (!side.last || side.last->isBeforeInBlock(op))
      side.last = op;
  }
  return side;
}

/// How many times the level of `side` runs, if the loops around it have
/// constant trip counts.
std::optional<int64_t> blocksOf(const Side &side) {
  int64_t blocks = 1;
  for (Operation *op = side.level->getParentOp(); op != side.task;
       op = op->getParentOp())
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      std::optional<int64_t> trips = constantTripCount(
          loop.getLowerBound(), loop.getUpperBound(), loop.getStep());
      if (!trips)
        return std::nullopt;
      blocks *= *trips;
    }
  return blocks;
}

/// The view of slot `slot` (an i8) of the ping-pong buffer `buffer`.
Value slotView(OpBuilder &builder, Location loc, Value buffer, Value slot) {
  auto type = cast<MemRefType>(buffer.getType());
  ArrayRef<int64_t> tile = type.getShape().drop_front();
  SmallVector<OpFoldResult> offsets(type.getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes{builder.getIndexAttr(1)};
  SmallVector<OpFoldResult> strides(type.getRank(), builder.getIndexAttr(1));
  for (int64_t size : tile)
    sizes.push_back(builder.getIndexAttr(size));
  offsets.front() =
      builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), slot)
          .getResult();
  auto viewType =
      cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          tile, type, offsets, sizes, strides));
  return builder.create<memref::SubViewOp>(loc, viewType, buffer, offsets,
                                           sizes, strides);
}

LogicalResult pingPong(DispatchOp dispatch) {
  Block &body = dispatch.getBody().front();
  OpBuilder builder(dispatch.getContext());
  Type i8 = builder.getI8Type();
  for (auto alloc : llvm::to_vector(body.getOps<memref::AllocOp>())) {
    Value buffer = alloc.getResult();
    auto accesses = accessesOf(buffer, dispatch);
    if (failed(accesses))
      return failure();
    if (accesses->size() < 2)
      continue;

    // The writer and the readers, and the tile they exchange.
    std::optional<Side> writer;
    SmallVector<Side> readers;
    MemRefType tile;
    for (auto &[task, list] : *accesses) {
      Side side = sideOf(task, list);
      for (const Access &access : list) {
        auto type = cast<MemRefType>(access.view.getType());
        if (!tile)
          tile = type;
        if (type.getShape() != tile.getShape())
          return access.op->emitOpError(
              "expected the tasks sharing a buffer to use it a tile of one "
              "shape at a time (sodap-dataflow-schedule)");
      }
      if (!side.writes) {
        readers.push_back(side);
      } else if (writer) {
        return alloc.emitOpError(
            "expected one task writing a buffer shared by tasks");
      } else {
        writer = side;
      }
    }
    if (!writer)
      return alloc.emitOpError("expected a task writing a buffer shared by "
                               "tasks");
    // ponytail: checks the counts only, not that the tiles come in the same
    // order; sodap-dataflow-schedule makes both agree.
    for (Side &reader : readers) {
      std::optional<int64_t> written = blocksOf(*writer),
                             read = blocksOf(reader);
      if (written && read && *written != *read)
        return reader.task.emitOpError()
               << "expected to read as many tiles of a shared buffer as its "
                  "writer writes, got "
               << *read << " and " << *written << " (sodap-dataflow-schedule)";
    }

    // The two slots, and a token stream per reader.
    Location loc = alloc.getLoc();
    SmallVector<int64_t> shape{2};
    llvm::append_range(shape, tile.getShape());
    builder.setInsertionPoint(alloc);
    Value slots = builder.create<memref::AllocOp>(
        loc, MemRefType::get(shape, tile.getElementType()));
    builder.setInsertionPointToStart(&body);
    SmallVector<Value> streams;
    for (unsigned i = 0; i < readers.size(); ++i)
      streams.push_back(builder.create<StreamOp>(
          loc, StreamType::get(i8.getContext(), i8, 1)));

    // The writer owns the slot variable: it writes the tile in the slot,
    // sends the slot to every reader, then switches to the other one. One
    // that also reads it (an accumulation) works on a private tile, copied in
    // the slot at the end: Bambu wants each task on a ping-pong buffer to be
    // a writer or a reader, not both.
    builder.setInsertionPointToStart(&writer->task.getBody().front());
    Value var = builder.create<memref::AllocaOp>(loc, MemRefType::get({}, i8));
    builder.create<memref::StoreOp>(
        loc, builder.create<arith::ConstantIntOp>(loc, 0, 8), var);
    Value target;
    if (writer->reads)
      target = builder.create<memref::AllocOp>(
          loc, MemRefType::get(tile.getShape(), tile.getElementType()));
    if (writer->reads)
      builder.setInsertionPointAfter(writer->last);
    else
      builder.setInsertionPoint(writer->first);
    Value slot = builder.create<memref::LoadOp>(loc, var);
    Value view = slotView(builder, loc, slots, slot);
    if (writer->reads)
      builder.create<linalg::CopyOp>(loc, target, view);
    else
      target = view;
    for (const Access &access : writer->accesses)
      access.op->replaceUsesOfWith(access.view, target);
    if (!writer->reads)
      builder.setInsertionPointAfter(writer->last);
    for (Value stream : streams)
      builder.create<StreamWriteOp>(loc, stream, slot);
    Value next = builder.create<arith::XOrIOp>(
        loc, slot, builder.create<arith::ConstantIntOp>(loc, 1, 8));
    builder.create<memref::StoreOp>(loc, next, var);

    // A reader peeks the slot at the start of its tile and frees it at the
    // end.
    for (auto [reader, stream] : llvm::zip(readers, streams)) {
      builder.setInsertionPoint(reader.first);
      Value peek = builder.create<StreamPeekOp>(loc, i8, stream);
      Value view = slotView(builder, loc, slots, peek);
      for (const Access &access : reader.accesses)
        access.op->replaceUsesOfWith(access.view, view);
      builder.setInsertionPointAfter(reader.last);
      builder.create<StreamReadOp>(loc, i8, stream);
    }
  }

  // The views of the old buffers, and the buffers.
  IRRewriter rewriter(dispatch.getContext());
  for (TaskOp task : body.getOps<TaskOp>())
    (void)runRegionDCE(rewriter, task->getRegions());
  for (auto alloc : llvm::make_early_inc_range(body.getOps<memref::AllocOp>()))
    if (alloc.use_empty())
      alloc.erase();
  return success();
}

struct PingPong : public mlir::sodap::dataflow::impl::PingPongBase<PingPong> {
  void runOnOperation() override {
    for (DispatchOp dispatch :
         llvm::to_vector(getOperation().front().getOps<DispatchOp>()))
      if (failed(pingPong(dispatch)))
        return signalPassFailure();
  }
};
} // namespace
