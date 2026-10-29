//===-- SODAPTransforms.cpp - SODAP transform dialect ops -----------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements SODAP transform dialect extension operations.
//
//===----------------------------------------------------------------------===//

#include "sodap/SODAPTransforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "sodap/Dialect/Dataflow/Dataflow.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <string>

class SODAPTransformsExtension
    : public ::mlir::transform::TransformDialectExtension<
          SODAPTransformsExtension> {
public:
  using Base::Base;
  void init();
};

void SODAPTransformsExtension::init() {
  declareGeneratedDialect<::mlir::affine::AffineDialect>();
  declareGeneratedDialect<::mlir::memref::MemRefDialect>();
  declareGeneratedDialect<::mlir::arith::ArithDialect>();
  declareGeneratedDialect<::mlir::sodap::dataflow::DataflowDialect>();
  registerTransformOps<
#define GET_OP_LIST
#include "sodap/SODAPTransforms.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "sodap/SODAPTransforms.cpp.inc"

::mlir::DiagnosedSilenceableFailure
mlir::transform::TagOpsOp::apply(::mlir::transform::TransformRewriter &rewriter,
                                 ::mlir::transform::TransformResults &results,
                                 ::mlir::transform::TransformState &state) {

  auto payload = state.getPayloadOps(getTarget());
  unsigned counter = 0;
  for (Operation *payloadOp : payload) {
    std::string uid = getPrefix().str() + "_" + std::to_string(counter++);
    payloadOp->setAttr("uid", StringAttr::get(payloadOp->getContext(), uid));
  }
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TagOpsOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

/// The outermost scf.for enclosing `op`, or null if `op` is not in a loop.
static ::mlir::scf::ForOp outermostLoop(::mlir::Operation *op) {
  ::mlir::scf::ForOp outer;
  for (auto loop = op->getParentOfType<::mlir::scf::ForOp>(); loop;
       loop = loop->getParentOfType<::mlir::scf::ForOp>())
    outer = loop;
  return outer;
}

/// Whether `value` is an argument of the function it is in.
static bool isFunctionArgument(::mlir::Value value) {
  using namespace ::mlir;
  auto arg = dyn_cast<BlockArgument>(value);
  auto func = arg ? dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp())
                  : func::FuncOp();
  return func && arg.getOwner() == &func.getBody().front();
}

/// The subview of a function argument that `value` is, if any.
static ::mlir::memref::SubViewOp argumentTile(::mlir::Value value) {
  using namespace ::mlir;
  auto subview = value.getDefiningOp<memref::SubViewOp>();
  if (!subview)
    return {};
  auto arg = dyn_cast<BlockArgument>(subview.getSource());
  auto func = subview->getParentOfType<func::FuncOp>();
  if (!arg || !func || arg.getOwner() != &func.getBody().front())
    return {};
  return subview;
}

::mlir::DiagnosedSilenceableFailure mlir::transform::PromoteTilesOp::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {
  // The linalg operands of each tile, in program order. A tile is a subview
  // of a function argument, keyed with no nest, or a scalar argument, which
  // cannot be tiled, keyed with the outermost loop: one buffer per nest.
  struct Tile {
    SmallVector<OpOperand *> uses;
    bool copyIn = false;
    Operation *lastWriter = nullptr;
  };
  llvm::MapVector<std::pair<Value, Operation *>, Tile> tiles;

  SmallVector<Operation *> targets;
  llvm::SmallPtrSet<Operation *, 8> targetSet;
  llvm::SetVector<func::FuncOp> funcs;
  for (Operation *op : state.getPayloadOps(getTarget())) {
    if (!isa<linalg::LinalgOp>(op))
      return emitSilenceableError() << "expected linalg ops";
    targets.push_back(op);
    targetSet.insert(op);
    if (auto func = op->getParentOfType<func::FuncOp>())
      funcs.insert(func);
  }

  // Walk in program order, whatever the order of the handle: the first user
  // of a tile decides whether it is copied in.
  SmallVector<linalg::LinalgOp> ordered;
  for (func::FuncOp func : funcs)
    func.walk([&](linalg::LinalgOp op) {
      if (targetSet.contains(op) && outermostLoop(op))
        ordered.push_back(op);
    });

  for (linalg::LinalgOp linalgOp : ordered) {
    Operation *op = linalgOp;
    for (OpOperand &operand : op->getOpOperands()) {
      Value region = operand.get();
      Operation *nest = nullptr;
      if (!argumentTile(region)) {
        // Any other argument used whole is a tiling left out: kept as is, for
        // the DMA to reject.
        auto type = dyn_cast<MemRefType>(region.getType());
        if (!isFunctionArgument(region) || !type || type.getRank() != 0)
          continue;
        nest = outermostLoop(op);
        if (linalgOp.isDpsInit(&operand)) {
          DiagnosedSilenceableFailure diag =
              emitSilenceableError() << "cannot promote a written scalar";
          diag.attachNote(region.getLoc()) << "scalar";
          return diag;
        }
      }
      if (!cast<MemRefType>(region.getType()).hasStaticShape()) {
        DiagnosedSilenceableFailure diag = emitSilenceableError()
                                           << "cannot promote a tile with a "
                                              "dynamic shape";
        diag.attachNote(region.getLoc()) << "tile";
        return diag;
      }
      bool reads = linalgOp.isDpsInput(&operand) ||
                   linalgOp.payloadUsesValueFromOperand(&operand);
      Tile &tile = tiles[{region, nest}];
      // Copy in only if the first user reads what is in memory.
      if (tile.uses.empty())
        tile.copyIn = reads;
      if (linalgOp.isDpsInit(&operand))
        tile.lastWriter = op;
      tile.uses.push_back(&operand);
    }
  }

  for (auto &[key, tile] : tiles) {
    auto [region, nest] = key;
    MemRefType type = cast<MemRefType>(region.getType());
    Location loc = region.getLoc();
    rewriter.setInsertionPoint(outermostLoop(tile.uses.front()->getOwner()));
    Value buffer = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(type.getShape(), type.getElementType()));

    // A scalar is the same for the whole nest: copied in once, before it.
    if (nest) {
      rewriter.create<memref::CopyOp>(loc, region, buffer);
      for (OpOperand *use : tile.uses)
        rewriter.modifyOpInPlace(use->getOwner(), [&] { use->set(buffer); });
      continue;
    }

    // The tile is the same for the whole block of the subview, so the copies
    // wrap the users at that level: a C tile defined in the j loop is copied
    // around the k loop, not at every k.
    Block *block = region.getDefiningOp()->getBlock();
    if (tile.copyIn) {
      rewriter.setInsertionPoint(
          block->findAncestorOpInBlock(*tile.uses.front()->getOwner()));
      rewriter.create<memref::CopyOp>(loc, region, buffer);
    }
    if (tile.lastWriter) {
      rewriter.setInsertionPointAfter(
          block->findAncestorOpInBlock(*tile.lastWriter));
      rewriter.create<memref::CopyOp>(loc, buffer, region);
    }
    for (OpOperand *use : tile.uses)
      rewriter.modifyOpInPlace(use->getOwner(), [&] { use->set(buffer); });
  }

  results.set(cast<OpResult>(getTransformed()), targets);
  return DiagnosedSilenceableFailure::success();
}

/// Multibuffers `alloc` with the slot linearized over the loop nest around its
/// users (see the op description). Checks everything before touching the IR.
static ::mlir::FailureOr<::mlir::memref::AllocOp>
multiBufferOverNest(::mlir::RewriterBase &rewriter,
                    ::mlir::memref::AllocOp alloc, int64_t factor) {
  using namespace ::mlir;
  MemRefType type = alloc.getType();
  if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
      alloc.use_empty())
    return failure();
  if (llvm::any_of(alloc->getUsers(), [](Operation *user) {
        return !isa<linalg::LinalgOp, memref::CopyOp>(user);
      }))
    return failure();

  // The innermost loop holding every user: the slot changes once per
  // iteration of it.
  auto level = (*alloc->getUsers().begin())->getParentOfType<scf::ForOp>();
  while (level && llvm::any_of(alloc->getUsers(), [&](Operation *user) {
           return !level->isProperAncestor(user);
         }))
    level = level->getParentOfType<scf::ForOp>();
  if (!level || level->isAncestor(alloc))
    return failure();

  // The loops the slot counts, outermost first.
  SmallVector<scf::ForOp> loops;
  for (scf::ForOp loop = level; loop && !loop->isAncestor(alloc);
       loop = loop->getParentOfType<scf::ForOp>())
    loops.insert(loops.begin(), loop);

  // index = sum_d ((iv_d - lb_d) floordiv step_d) * (trip counts inside d)
  MLIRContext *ctx = rewriter.getContext();
  AffineExpr index = getAffineConstantExpr(0, ctx);
  int64_t stride = 1;
  for (int d = loops.size() - 1; d >= 0; --d) {
    std::optional<int64_t> lb = getConstantIntValue(loops[d].getLowerBound());
    std::optional<int64_t> ub = getConstantIntValue(loops[d].getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(loops[d].getStep());
    if (!lb || !ub || !step)
      return failure();
    index = index + (getAffineDimExpr(d, ctx) - *lb).floorDiv(*step) * stride;
    stride *= llvm::divideCeil(*ub - *lb, *step);
  }

  Location loc = alloc.getLoc();
  SmallVector<int64_t> shape{factor};
  llvm::append_range(shape, type.getShape());
  rewriter.setInsertionPoint(alloc);
  auto newAlloc = rewriter.create<memref::AllocOp>(
      loc,
      MemRefType::get(shape, type.getElementType(), MemRefLayoutAttrInterface{},
                      type.getMemorySpace()),
      alloc.getAlignmentAttr());

  rewriter.setInsertionPointToStart(level.getBody());
  SmallVector<Value> ivs = llvm::map_to_vector(
      loops, [](scf::ForOp loop) { return loop.getInductionVar(); });
  Value slot = rewriter.create<affine::AffineApplyOp>(
      loc, AffineMap::get(loops.size(), 0, index % factor), ivs);
  SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
  offsets[0] = slot;
  SmallVector<OpFoldResult> sizes = getAsIndexOpFoldResult(ctx, shape);
  sizes[0] = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));
  auto viewType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
      type.getShape(), newAlloc.getType(), offsets, sizes, strides));
  Value view = rewriter.create<memref::SubViewOp>(loc, viewType, newAlloc,
                                                  offsets, sizes, strides);

  rewriter.replaceAllUsesWith(alloc, view);
  rewriter.eraseOp(alloc);
  return newAlloc;
}

::mlir::DiagnosedSilenceableFailure mlir::transform::MultiBufferOp::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {
  SmallVector<Operation *> newAllocs;
  for (Operation *op : state.getPayloadOps(getTarget())) {
    // Filled once before the nest: nothing to alternate.
    if (llvm::any_of(op->getUsers(), [](Operation *user) {
          return isa<memref::CopyOp>(user) &&
                 !user->getParentOfType<scf::ForOp>();
        }))
      continue;
    FailureOr<memref::AllocOp> newAlloc =
        multiBufferOverNest(rewriter, cast<memref::AllocOp>(op), getFactor());
    if (failed(newAlloc)) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "op failed to multibuffer";
      diag.attachNote(op->getLoc()) << "multibuffer candidate";
      return diag;
    }
    newAllocs.push_back(*newAlloc);
  }
  results.set(cast<OpResult>(getResult()), newAllocs);
  return DiagnosedSilenceableFailure::success();
}

/// The allocation `value` is a view of, if any.
static ::mlir::memref::AllocOp rootAlloc(::mlir::Value value) {
  using namespace ::mlir;
  while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
    value = view.getViewSource();
  return value.getDefiningOp<memref::AllocOp>();
}

/// Whether `op`, a stage op (a copy or a linalg op), writes into `alloc`.
static bool writesBuffer(::mlir::Operation *op, ::mlir::memref::AllocOp alloc) {
  using namespace ::mlir;
  if (auto copy = dyn_cast<memref::CopyOp>(op))
    return rootAlloc(copy.getTarget()) == alloc;
  return llvm::any_of(cast<linalg::LinalgOp>(op).getDpsInits(),
                      [&](Value init) { return rootAlloc(init) == alloc; });
}

namespace {
/// How a stage accesses a buffer.
struct Access {
  bool reads = false;
  bool writes = false;
};

/// A stage: its side-effecting ops, and the buffers they access.
struct Stage {
  ::mlir::SmallVector<::mlir::Operation *> ops;
  ::llvm::MapVector<::mlir::memref::AllocOp, Access> buffers;
};

/// The token of a buffer shared by a producer and a consumer stage. A buffer
/// loaded once before the nest has no view nor slot: its token is for the
/// whole task, and carries 0.
struct Token {
  ::mlir::memref::AllocOp alloc;
  ::mlir::memref::SubViewOp view;
  ::mlir::Value slot;
  unsigned producer;
  unsigned consumer;
};
} // namespace

/// Replaces `nest` with a dispatch of one task per stage (see the op
/// description). Checks everything before touching the IR; on failure,
/// `reason` says why.
static ::mlir::FailureOr<::mlir::sodap::dataflow::DispatchOp>
splitStages(::mlir::RewriterBase &rewriter, ::mlir::scf::ForOp nest,
            std::string &reason) {
  using namespace ::mlir;
  namespace df = ::mlir::sodap::dataflow;

  // The side-effecting ops of the nest, by kind.
  SmallVector<Operation *> loads, stores;
  SmallVector<linalg::LinalgOp> computes;
  Operation *unsupported = nullptr;
  nest.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      computes.push_back(linalgOp);
      return WalkResult::skip();
    }
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      if (argumentTile(copy.getSource()) ||
          isFunctionArgument(copy.getSource()))
        loads.push_back(op);
      else if (argumentTile(copy.getTarget()) ||
               isFunctionArgument(copy.getTarget()))
        stores.push_back(op);
      else
        unsupported = op;
    } else if (!isa<scf::ForOp, scf::YieldOp>(op) && !isMemoryEffectFree(op)) {
      unsupported = op;
    }
    return unsupported ? WalkResult::interrupt() : WalkResult::advance();
  });
  if (unsupported) {
    reason = "unsupported op in the nest: " +
             unsupported->getName().getStringRef().str();
    return failure();
  }

  // The loads done once before the nest (a scalar), into a buffer it uses.
  SmallVector<Operation *> prologue;
  for (Operation &op : llvm::make_range(nest->getBlock()->begin(),
                                        nest->getIterator())) {
    auto copy = dyn_cast<memref::CopyOp>(op);
    memref::AllocOp alloc = copy ? rootAlloc(copy.getTarget()) : nullptr;
    if (alloc && isFunctionArgument(copy.getSource()) &&
        llvm::any_of(alloc->getUsers(), [&](Operation *user) {
          return nest->isAncestor(user);
        }))
      prologue.push_back(copy);
  }
  loads.insert(loads.begin(), prologue.begin(), prologue.end());

  // Compute stages: the linalg ops writing the same buffers, in program order.
  llvm::EquivalenceClasses<Operation *> written;
  for (linalg::LinalgOp op : computes) {
    Operation *first = nullptr;
    for (OpOperand &init : op.getDpsInitsMutable()) {
      memref::AllocOp alloc = rootAlloc(init.get());
      if (!alloc) {
        reason = "a linalg op writes no local buffer";
        return failure();
      }
      written.insert(alloc);
      if (first)
        written.unionSets(first, alloc);
      first = alloc;
    }
  }
  SmallVector<Stage> stages(1);
  stages[0].ops = loads;
  DenseMap<Operation *, unsigned> computeStage;
  for (linalg::LinalgOp op : computes) {
    Operation *leader = written.getLeaderValue(
        rootAlloc(op.getDpsInitOperand(0)->get()).getOperation());
    auto [it, inserted] = computeStage.try_emplace(leader, stages.size());
    if (inserted)
      stages.emplace_back();
    stages[it->second].ops.push_back(op);
  }
  stages.emplace_back();
  stages.back().ops = stores;

  // What each stage reads and writes.
  for (Stage &stage : stages)
    for (Operation *op : stage.ops) {
      auto record = [&](Value value, bool reads, bool writes) {
        if (memref::AllocOp alloc = rootAlloc(value)) {
          Access &access = stage.buffers[alloc];
          access.reads |= reads;
          access.writes |= writes;
        }
      };
      if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        record(copy.getSource(), true, false);
        record(copy.getTarget(), false, true);
        continue;
      }
      auto linalgOp = cast<linalg::LinalgOp>(op);
      for (OpOperand &operand : op->getOpOperands())
        record(operand.get(),
               linalgOp.isDpsInput(&operand) ||
                   linalgOp.payloadUsesValueFromOperand(&operand),
               linalgOp.isDpsInit(&operand));
    }

  // One token per consumer of each buffer shared between stages.
  llvm::MapVector<memref::AllocOp, SmallVector<unsigned>> users;
  for (auto [idx, stage] : llvm::enumerate(stages))
    for (auto &entry : stage.buffers)
      users[entry.first].push_back(idx);
  SmallVector<Token> tokens;
  // The views of the shared buffers their producer also reads.
  SmallVector<std::pair<unsigned, memref::SubViewOp>> privates;
  for (auto &[alloc, stageIdxs] : users) {
    if (stageIdxs.size() < 2)
      continue;
    auto writers = llvm::to_vector(llvm::make_filter_range(
        stageIdxs, [&](unsigned s) { return stages[s].buffers[alloc].writes; }));
    if (writers.size() != 1) {
      reason = "a shared buffer must be written by exactly one stage";
      return failure();
    }
    // Loaded once before the nest: one token for the whole task.
    if (llvm::any_of(prologue, [shared = alloc](Operation *copy) {
          return rootAlloc(cast<memref::CopyOp>(copy).getTarget()) == shared;
        })) {
      for (unsigned consumer : stageIdxs)
        if (consumer != writers.front())
          tokens.push_back({alloc, {}, {}, writers.front(), consumer});
      continue;
    }
    auto view = alloc->hasOneUse()
                    ? dyn_cast<memref::SubViewOp>(*alloc->user_begin())
                    : memref::SubViewOp();
    Value slot = view ? llvm::dyn_cast_if_present<Value>(
                            view.getMixedOffsets().front())
                      : Value();
    if (!slot || !nest->isProperAncestor(view)) {
      reason = "a shared buffer is not multibuffered in the nest";
      return failure();
    }
    for (unsigned consumer : stageIdxs)
      if (consumer != writers.front())
        tokens.push_back({alloc, view, slot, writers.front(), consumer});
    if (stages[writers.front()].buffers[alloc].reads)
      privates.push_back({writers.front(), view});
  }

  Location loc = nest.getLoc();
  Type i8 = rewriter.getI8Type();
  rewriter.setInsertionPoint(nest);
  auto dispatch = rewriter.create<df::DispatchOp>(loc, TypeRange{});
  rewriter.setInsertionPointToStart(&dispatch.getBody().emplaceBlock());
  SmallVector<Value> streams;
  for (unsigned i = 0; i < tokens.size(); ++i)
    streams.push_back(rewriter.create<df::StreamOp>(
        loc, df::StreamType::get(rewriter.getContext(), i8, 1)));
  auto dispatchYield = rewriter.create<df::YieldOp>(loc, ValueRange{});

  for (auto [s, stage] : llvm::enumerate(stages)) {
    if (stage.ops.empty())
      continue;
    rewriter.setInsertionPoint(dispatchYield);
    auto task = rewriter.create<df::TaskOp>(loc, TypeRange{});
    rewriter.createBlock(&task.getBody());
    auto taskYield = rewriter.create<df::YieldOp>(loc, ValueRange{});
    rewriter.setInsertionPoint(taskYield);

    IRMapping map;
    if (s == 0)
      for (Operation *op : prologue)
        rewriter.clone(*op, map);
    rewriter.clone(*nest, map);
    for (auto [other, otherStage] : llvm::enumerate(stages))
      if (other != s)
        for (Operation *op : otherStage.ops)
          if (Operation *clone = map.lookupOrNull(op))
            rewriter.eraseOp(clone);

    // Where a token lives: the loop body of its slot, or the whole task.
    auto levelOf = [&](Token token) {
      return token.view ? map.lookup(token.view.getOperation())->getBlock()
                        : &task.getBody().front();
    };

    // A producer that also reads its shared buffer (an accumulation) works on
    // a private one, copied into the slot at the end of the slot's loop body,
    // ahead of the tokens: Bambu wants every stage connected to a ping-pong
    // buffer to be a writer or a reader, not both.
    for (auto [producer, view] : privates) {
      if (producer != s)
        continue;
      auto slotView = cast<memref::SubViewOp>(map.lookup(view.getOperation()));
      MemRefType type = slotView.getType();
      rewriter.setInsertionPointToStart(&task.getBody().front());
      Value buffer = rewriter.create<memref::AllocOp>(
          loc, MemRefType::get(type.getShape(), type.getElementType()));
      rewriter.replaceAllUsesWith(slotView.getResult(), buffer);
      rewriter.setInsertionPoint(slotView->getBlock()->getTerminator());
      rewriter.create<linalg::CopyOp>(loc, buffer, slotView.getResult());
    }

    // Releases at the end of the slot's loop body; peeks go in reverse so
    // that, each one inserted at the start, they end up in token order.
    for (auto [token, stream] : llvm::reverse(llvm::zip(tokens, streams))) {
      if (token.consumer != s)
        continue;
      Block *level = levelOf(token);
      rewriter.setInsertionPointToStart(level);
      rewriter.create<df::StreamPeekOp>(loc, i8, stream);
    }
    for (auto [token, stream] : llvm::zip(tokens, streams)) {
      if (token.consumer != s)
        continue;
      Block *level = levelOf(token);
      rewriter.setInsertionPoint(level->getTerminator());
      rewriter.create<df::StreamReadOp>(loc, i8, stream);
    }
    // Signals before the first loop after the stage's last write of the slot
    // in that body, or at its end: a load at an outer level must not wait for
    // the inner loops after it, which wait for the consumer, which waits for
    // the signal. Not right after the write: a signal blocks until the slot's
    // previous reader releases it, and the copies of one level should all be
    // issued by then. A private buffer is written to the slot at the end of
    // the body, so it signals there.
    for (auto [token, stream] : llvm::zip(tokens, streams)) {
      if (token.producer != s)
        continue;
      Block *level = levelOf(token);
      rewriter.setInsertionPoint(level->getTerminator());
      bool isPrivate = llvm::is_contained(
          privates, std::make_pair(token.producer, token.view));
      Operation *last = nullptr;
      for (Operation *op : isPrivate ? ArrayRef<Operation *>() : stage.ops) {
        if (!writesBuffer(op, token.alloc))
          continue;
        Operation *ancestor = level->findAncestorOpInBlock(*map.lookup(op));
        if (ancestor && (!last || last->isBeforeInBlock(ancestor)))
          last = ancestor;
      }
      if (last) {
        Operation *next = last->getNextNode();
        while (!isa<scf::ForOp>(next) && next != level->getTerminator())
          next = next->getNextNode();
        rewriter.setInsertionPoint(next);
      }
      Value slot =
          token.slot
              ? rewriter.create<arith::IndexCastOp>(loc, i8,
                                                    map.lookup(token.slot))
                    .getResult()
              : rewriter.create<arith::ConstantIntOp>(loc, 0, 8).getResult();
      rewriter.create<df::StreamWriteOp>(loc, stream, slot);
    }

    // What the stage does not use: other stages' views, loops left empty.
    (void)runRegionDCE(rewriter, task->getRegions());
  }

  for (Operation *op : prologue)
    rewriter.eraseOp(op);
  rewriter.eraseOp(nest);
  return dispatch;
}

::mlir::DiagnosedSilenceableFailure mlir::transform::SplitStagesOp::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {
  // Each nest once, through its outermost loop, whichever loops the handle has.
  llvm::SetVector<scf::ForOp> nests;
  for (Operation *op : state.getPayloadOps(getTarget())) {
    scf::ForOp outer = outermostLoop(op);
    nests.insert(outer ? outer : cast<scf::ForOp>(op));
  }
  SmallVector<Operation *> dispatches;
  for (scf::ForOp nest : nests) {
    Operation *op = nest;
    std::string reason;
    FailureOr<sodap::dataflow::DispatchOp> dispatch =
        splitStages(rewriter, nest, reason);
    if (failed(dispatch)) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "op failed to split into stages: " << reason;
      diag.attachNote(op->getLoc()) << "nest";
      return diag;
    }
    dispatches.push_back(*dispatch);
  }
  results.set(cast<OpResult>(getDispatch()), dispatches);
  return DiagnosedSilenceableFailure::success();
}

namespace mlir {
namespace sodap {
void registerSODAPTransforms(::mlir::DialectRegistry &registry) {
  registry.addExtensions<SODAPTransformsExtension>();
}
} // namespace sodap
} // namespace mlir
