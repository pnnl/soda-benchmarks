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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "sodap/Dialect/Dataflow/Dataflow.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <numeric>
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
  declareGeneratedDialect<::mlir::bufferization::BufferizationDialect>();
  declareGeneratedDialect<::mlir::scf::SCFDialect>();
  declareGeneratedDialect<::mlir::tensor::TensorDialect>();
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

/// Tile sizes of `op`: `size` for its parallel dims (or for the others, if
/// `parallel` is false), 0 for the rest.
static ::mlir::SmallVector<::mlir::OpFoldResult>
tileSizes(::mlir::OpBuilder &builder, ::mlir::linalg::LinalgOp op, int64_t size,
          bool parallel) {
  using namespace ::mlir;
  SmallVector<OpFoldResult> sizes;
  for (utils::IteratorType iterator : op.getIteratorTypesArray())
    sizes.push_back(builder.getIndexAttr(
        (iterator == utils::IteratorType::parallel) == parallel ? size : 0));
  return sizes;
}

::mlir::DiagnosedSilenceableFailure mlir::transform::TileAndFuseOp::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {
  // The op producing each yielded value, before anything changes, with the
  // tile of its function.
  llvm::MapVector<Operation *, int64_t> lasts;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto task = dyn_cast<sodap::dataflow::TaskOp>(target);
    auto func = task ? task->getParentOfType<func::FuncOp>() : nullptr;
    if (!func)
      return emitSilenceableError()
             << "expected a dataflow.task in a func.func";

    // The GCD of every loop range in the function, so every loop tiles
    // evenly and every task gets the same tile.
    int64_t gcd = 0;
    Operation *dynamic = nullptr;
    func.walk([&](linalg::LinalgOp op) {
      for (int64_t range : op.getStaticLoopRanges()) {
        if (ShapedType::isDynamic(range))
          dynamic = op;
        else
          gcd = std::gcd(gcd, range);
      }
    });
    if (dynamic) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "expected static loop ranges";
      diag.attachNote(dynamic->getLoc()) << "op";
      return diag;
    }
    int64_t size = std::min<int64_t>(getMaxTileSize(), std::max<int64_t>(gcd, 1));
    while (gcd % size != 0)
      --size;

    Value invalid;
    for (Value result : task.getYieldOp().getOperands()) {
      auto last = result.getDefiningOp<linalg::LinalgOp>();
      if (!last || last.getNumParallelLoops() == 0)
        invalid = result;
      else
        lasts.insert({last, size});
    }
    if (invalid) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "expected a result produced by a linalg op with a parallel dim";
      diag.attachNote(invalid.getLoc()) << "result";
      return diag;
    }
  }

  for (auto [last, size] : lasts) {
    scf::SCFTilingOptions parallelTiling;
    parallelTiling.setTileSizes(
        tileSizes(rewriter, cast<linalg::LinalgOp>(last), size, true));
    scf::SCFTileAndFuseOptions options;
    options.setTilingOptions(parallelTiling);
    // Fuse only the tileable producers in the task. LLVM 19 clones a producer
    // before checking that it can be tiled and leaves the clone behind when it
    // can't: a task reading another task's result would get a copy of that
    // whole task in its loop.
    Operation *task = last->getParentOfType<sodap::dataflow::TaskOp>();
    options.setFusionControlFn(
        [task](tensor::ExtractSliceOp, OpResult producer, bool) {
          Operation *op = producer.getOwner();
          return std::make_tuple(
              isa<TilingInterface>(op) && task->isAncestor(op), false);
        });
    rewriter.setInsertionPoint(last);
    FailureOr<scf::SCFTileAndFuseResult> nest =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, cast<TilingInterface>(last), options);
    if (failed(nest))
      return emitDefiniteFailure() << "failed to tile and fuse";

    SmallVector<Operation *> replaced{last};
    llvm::append_range(replaced, nest->fusedProducers);
    for (Operation *op : replaced) {
      for (OpResult result : op->getResults())
        if (Value replacement = nest->replacements.lookup(result))
          rewriter.replaceAllUsesWith(result, replacement);
      if (op->use_empty())
        rewriter.eraseOp(op);
    }

    SmallVector<linalg::LinalgOp> reductions;
    nest->loops.front().getOperation()->walk([&](linalg::LinalgOp op) {
      if (op.getNumReductionLoops() > 0)
        reductions.push_back(op);
    });
    for (linalg::LinalgOp op : reductions) {
      scf::SCFTilingOptions reductionTiling;
      reductionTiling.setTileSizes(tileSizes(rewriter, op, size, false));
      rewriter.setInsertionPoint(op);
      FailureOr<scf::SCFTilingResult> tiled = scf::tileUsingSCF(
          rewriter, cast<TilingInterface>(op.getOperation()), reductionTiling);
      if (failed(tiled))
        return emitDefiniteFailure() << "failed to tile a reduction";
      rewriter.replaceOp(op, tiled->replacements);
    }
  }
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TileAndFuseOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

::mlir::DiagnosedSilenceableFailure mlir::transform::CreateTasksOp::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {
  namespace df = ::mlir::sodap::dataflow;
  SmallVector<Operation *> dispatches;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto func = dyn_cast<func::FuncOp>(target);
    if (!func || !func.getBody().hasOneBlock())
      return emitSilenceableError() << "expected a func.func with one block";
    Block &body = func.getBody().front();

    // The compute ops, in program order.
    SmallVector<Operation *> computes;
    for (Operation &op : body.without_terminator()) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (linalgOp && linalgOp.hasPureTensorSemantics())
        computes.push_back(&op);
      else if (!isa<tensor::EmptyOp, arith::ConstantOp>(op)) {
        DiagnosedSilenceableFailure diag =
            emitSilenceableError()
            << "expected only linalg ops on tensors, tensor.empty and "
               "constants";
        diag.attachNote(op.getLoc()) << "op";
        return diag;
      }
    }
    if (computes.empty())
      return emitSilenceableError() << "expected a linalg op";

    // The partition: ops chained through their inits, then the groups.
    llvm::EquivalenceClasses<Operation *> partition;
    auto isCompute = [&](Operation *op) {
      return op && partition.findValue(op) != partition.end();
    };
    for (Operation *op : computes) {
      partition.insert(op);
      for (OpOperand &init : cast<linalg::LinalgOp>(op).getDpsInitsMutable())
        if (Operation *producer = init.get().getDefiningOp();
            isCompute(producer))
          partition.unionSets(op, producer);
    }
    for (Value handle : getGroups()) {
      Operation *first = nullptr;
      for (Operation *op : state.getPayloadOps(handle)) {
        if (!func->isAncestor(op))
          continue;
        if (!isCompute(op)) {
          DiagnosedSilenceableFailure diag =
              emitSilenceableError()
              << "expected grouped ops to be linalg ops of the function";
          diag.attachNote(op->getLoc()) << "op";
          return diag;
        }
        if (first)
          partition.unionSets(first, op);
        first = op;
      }
    }

    // The groups, in program order of their first op.
    SmallVector<SmallVector<Operation *>> groups;
    DenseMap<Operation *, unsigned> groupOfLeader;
    for (Operation *op : computes) {
      auto [it, inserted] = groupOfLeader.try_emplace(
          partition.getLeaderValue(op), groups.size());
      if (inserted)
        groups.emplace_back();
      groups[it->second].push_back(op);
    }
    auto groupOf = [&](Operation *op) {
      return groupOfLeader.lookup(partition.getLeaderValue(op));
    };

    // The task order: a group after the groups whose results it uses, in
    // program order when there is a choice.
    SmallVector<llvm::SmallSetVector<unsigned, 4>> deps(groups.size());
    for (auto [i, ops] : llvm::enumerate(groups))
      for (Operation *op : ops)
        for (Value operand : op->getOperands())
          if (Operation *def = operand.getDefiningOp();
              isCompute(def) && groupOf(def) != i)
            deps[i].insert(groupOf(def));
    SmallVector<unsigned> order;
    SmallVector<bool> placed(groups.size(), false);
    while (order.size() < groups.size()) {
      unsigned next = 0;
      while (next < groups.size() &&
             (placed[next] || !llvm::all_of(deps[next], [&](unsigned dep) {
               return placed[dep];
             })))
        ++next;
      if (next == groups.size())
        return emitSilenceableError() << "expected groups that don't depend "
                                         "on each other";
      placed[next] = true;
      order.push_back(next);
    }

    // What each group yields, its results used outside it, and the init each
    // one is written into: the root of its chain of inits, a tensor.empty.
    SmallVector<llvm::SmallSetVector<Value, 4>> yields(groups.size());
    SmallVector<llvm::SmallSetVector<OpOperand *, 4>> edges(groups.size());
    for (unsigned i = 0; i < groups.size(); ++i)
      for (Operation *op : groups[i])
        for (OpResult result : op->getResults()) {
          if (llvm::all_of(result.getUsers(), [&](Operation *user) {
                return isCompute(user) && groupOf(user) == i;
              }))
            continue;
          yields[i].insert(result);
          OpOperand *init = nullptr;
          for (OpResult value = result; value && isCompute(value.getOwner());
               value = dyn_cast<OpResult>(init->get()))
            init = cast<DestinationStyleOpInterface>(value.getOwner())
                       .getTiedOpOperand(value);
          if (!init->get().getDefiningOp<tensor::EmptyOp>()) {
            DiagnosedSilenceableFailure diag =
                emitSilenceableError()
                << "expected an edge written into a tensor.empty";
            diag.attachNote(op->getLoc()) << "op";
            return diag;
          }
          edges[i].insert(init);
        }

    // The values the function returns, from the dispatch.
    auto ret = cast<func::ReturnOp>(body.getTerminator());
    llvm::SmallSetVector<Value, 4> outputs;
    for (Value value : ret.getOperands())
      if (isCompute(value.getDefiningOp()))
        outputs.insert(value);

    Location loc = func.getLoc();
    rewriter.setInsertionPoint(ret);
    auto dispatch = rewriter.create<df::DispatchOp>(
        loc, TypeRange(ValueRange(outputs.getArrayRef())));
    Block *dispatchBlock = rewriter.createBlock(&dispatch.getBody());
    DenseMap<Value, Value> taskResults;
    for (unsigned i : order) {
      rewriter.setInsertionPointToEnd(dispatchBlock);
      for (OpOperand *edge : edges[i]) {
        auto empty = edge->get().getDefiningOp<tensor::EmptyOp>();
        Value buffer = rewriter.create<bufferization::AllocTensorOp>(
            empty.getLoc(), empty.getType(), empty.getDynamicSizes());
        rewriter.modifyOpInPlace(edge->getOwner(), [&] { edge->set(buffer); });
      }

      auto task = rewriter.create<df::TaskOp>(
          loc, TypeRange(ValueRange(yields[i].getArrayRef())));
      Block *taskBlock = rewriter.createBlock(&task.getBody());
      for (Operation *op : groups[i]) {
        rewriter.moveOpBefore(op, taskBlock, taskBlock->end());
        // A private init: a tensor.empty of its own, in the task.
        for (OpOperand &operand : op->getOpOperands())
          if (auto empty = operand.get().getDefiningOp<tensor::EmptyOp>();
              empty && !task->isAncestor(empty)) {
            rewriter.setInsertionPoint(op);
            Value copy = rewriter.clone(*empty)->getResult(0);
            rewriter.modifyOpInPlace(op, [&] { operand.set(copy); });
          }
      }
      rewriter.setInsertionPointToEnd(taskBlock);
      rewriter.create<df::YieldOp>(loc, yields[i].getArrayRef());

      for (auto [value, result] : llvm::zip(yields[i], task.getResults())) {
        rewriter.replaceUsesWithIf(value, result, [&](OpOperand &use) {
          return !task->isAncestor(use.getOwner());
        });
        taskResults[value] = result;
      }
    }
    rewriter.setInsertionPointToEnd(dispatchBlock);
    SmallVector<Value> dispatchYields;
    for (Value output : outputs)
      dispatchYields.push_back(taskResults.lookup(output));
    rewriter.create<df::YieldOp>(loc, dispatchYields);
    for (auto [value, result] :
         llvm::zip(dispatchYields, dispatch.getResults()))
      rewriter.replaceUsesWithIf(value, result, [&](OpOperand &use) {
        return use.getOwner() == ret;
      });

    for (Operation &op : llvm::make_early_inc_range(body))
      if (isa<tensor::EmptyOp>(op) && op.use_empty())
        rewriter.eraseOp(&op);
    dispatches.push_back(dispatch);
  }
  results.set(cast<OpResult>(getDispatch()), dispatches);
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::CreateTasksOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  onlyReadsHandle(getGroupsMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
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

/// Prove that this argument and its subviews are only read. This permits a
/// whole static input left by fusion to be copied once before the nest. Unknown
/// users and any write forbid hoisting, including writes through a subview.
static bool isReadOnlyBuffer(::mlir::Value value) {
  using namespace ::mlir;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto view = dyn_cast<memref::SubViewOp>(user)) {
      if (!isReadOnlyBuffer(view.getResult()))
        return false;
    } else if (auto op = dyn_cast<linalg::LinalgOp>(user)) {
      if (!op.isDpsInput(&use))
        return false;
    } else if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      if (use.getOperandNumber() != 0)
        return false;
    } else {
      return false;
    }
  }
  return true;
}

::mlir::DiagnosedSilenceableFailure mlir::transform::PromoteTilesOp::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {
  // The linalg operands of each tile, in program order. A tile is a subview
  // of a function argument, keyed with no nest, or a whole read-only argument,
  // keyed with the outermost loop: one buffer per nest.
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
        // Fusion can leave whole static inputs in the nest. Loading them once
        // is a conservative fallback; choosing smaller tiles is separate work.
        auto type = dyn_cast<MemRefType>(region.getType());
        if (!isFunctionArgument(region) || !type)
          continue;
        if (type.getRank() != 0 && !isReadOnlyBuffer(region))
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

    // Whole read-only arguments are copied once, before the nest.
    if (nest) {
      rewriter.create<memref::CopyOp>(loc, region, buffer);
      for (OpOperand *use : tile.uses)
        rewriter.modifyOpInPlace(use->getOwner(), [&] { use->set(buffer); });
      continue;
    }

    // The tile is the same for the whole block of the subview, so the copies
    // wrap the users at that level: a C tile defined in the j loop is copied
    // around the k loop, not at every k. In as soon as the subview exists, so
    // that the C tile comes in before the k loop even if only the epilogue
    // after it reads it.
    Block *block = region.getDefiningOp()->getBlock();
    if (tile.copyIn) {
      rewriter.setInsertionPointAfter(region.getDefiningOp());
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

/// Stores in `var` the slot after `slot`: var = (slot + 1) % factor.
static void toggleSlot(::mlir::RewriterBase &rewriter, ::mlir::Location loc,
                       ::mlir::Value var, ::mlir::Value slot, int64_t factor) {
  using namespace ::mlir;
  Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 8);
  Value inc = rewriter.create<arith::AddIOp>(loc, slot, one);
  Value size = rewriter.create<arith::ConstantIntOp>(loc, factor, 8);
  Value next = rewriter.create<arith::RemUIOp>(loc, inc, size);
  rewriter.create<memref::StoreOp>(loc, next, var);
}

/// Subviews of intermediate tensors must follow the selected bank, including
/// their offset/stride types. Visit parents before children when rebasing.
static void replaceBufferAndRebaseSubviews(::mlir::RewriterBase &rewriter,
                                           ::mlir::Value from,
                                           ::mlir::Value to) {
  using namespace ::mlir;
  SmallVector<memref::SubViewOp> views;
  auto collect = [&](auto &&self, Value value) -> void {
    for (Operation *user : value.getUsers())
      if (auto view = dyn_cast<memref::SubViewOp>(user)) {
        views.push_back(view);
        self(self, view.getResult());
      }
  };
  collect(collect, from);
  rewriter.replaceAllUsesWith(from, to);
  for (memref::SubViewOp view : views) {
    auto type = memref::SubViewOp::inferRankReducedResultType(
        view.getType().getShape(), view.getSourceType(), view.getMixedOffsets(),
        view.getMixedSizes(), view.getMixedStrides());
    rewriter.modifyOpInPlace(view, [&] {
      view.getResult().setType(cast<MemRefType>(type));
    });
  }
}

/// Multibuffers `alloc` with a slot switched once per iteration of the common
/// loop around all direct and subview users. Checks before touching the IR.
static ::mlir::FailureOr<::mlir::memref::AllocOp>
multiBufferOverNest(::mlir::RewriterBase &rewriter,
                    ::mlir::memref::AllocOp alloc, int64_t factor) {
  using namespace ::mlir;
  MemRefType type = alloc.getType();
  if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
      alloc.use_empty())
    return failure();
  SmallVector<Operation *> users;
  auto collectUsers = [&](auto &&self, Value value) -> LogicalResult {
    for (Operation *user : value.getUsers()) {
      users.push_back(user);
      if (auto view = dyn_cast<memref::SubViewOp>(user)) {
        if (failed(self(self, view.getResult())))
          return failure();
      } else if (!isa<linalg::LinalgOp, memref::CopyOp>(user)) {
        return failure();
      }
    }
    return success();
  };
  if (failed(collectUsers(collectUsers, alloc)))
    return failure();

  // The innermost loop holding every user: the slot changes once per
  // iteration of it.
  auto level = users.front()->getParentOfType<scf::ForOp>();
  while (level && llvm::any_of(users, [&](Operation *user) {
           return !level->isProperAncestor(user);
         }))
    level = level->getParentOfType<scf::ForOp>();
  if (!level || level->isAncestor(alloc))
    return failure();

  MLIRContext *ctx = rewriter.getContext();
  Location loc = alloc.getLoc();
  SmallVector<int64_t> shape{factor};
  llvm::append_range(shape, type.getShape());
  rewriter.setInsertionPoint(alloc);
  auto newAlloc = rewriter.create<memref::AllocOp>(
      loc,
      MemRefType::get(shape, type.getElementType(), MemRefLayoutAttrInterface{},
                      type.getMemorySpace()),
      alloc.getAlignmentAttr());
  Value var = rewriter.create<memref::AllocaOp>(
      loc, MemRefType::get(ArrayRef<int64_t>{}, rewriter.getI8Type()));
  rewriter.create<memref::StoreOp>(
      loc, rewriter.create<arith::ConstantIntOp>(loc, 0, 8), var);

  rewriter.setInsertionPointToStart(level.getBody());
  Value slot = rewriter.create<memref::LoadOp>(loc, var);
  SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
  offsets[0] = rewriter
                   .create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                               slot)
                   .getResult();
  SmallVector<OpFoldResult> sizes = getAsIndexOpFoldResult(ctx, shape);
  sizes[0] = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));
  auto viewType = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
      type.getShape(), newAlloc.getType(), offsets, sizes, strides));
  Value view = rewriter.create<memref::SubViewOp>(loc, viewType, newAlloc,
                                                  offsets, sizes, strides);

  rewriter.setInsertionPoint(level.getBody()->getTerminator());
  toggleSlot(rewriter, loc, var, slot, factor);

  replaceBufferAndRebaseSubviews(rewriter, alloc, view);
  rewriter.eraseOp(alloc);
  return newAlloc;
}

::mlir::DiagnosedSilenceableFailure mlir::transform::MultiBufferOp::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {
  if (getFactor() != 2)
    return emitSilenceableError() << "only factor 2 is supported";

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

/// The slot variable of a multibuffered buffer, as `sodap.multibuffer` makes it.
struct Slot {
  ::mlir::memref::AllocOp alloc;
  ::mlir::memref::SubViewOp view;
  ::mlir::Value var;
  ::mlir::Value value;
  ::mlir::memref::StoreOp init;
  ::mlir::memref::StoreOp toggle;
};

/// The token of a buffer shared by a producer and a consumer stage. A buffer
/// loaded once before the nest has no view nor slot: its token is for the
/// whole task, and carries 0.
struct Token {
  ::mlir::memref::AllocOp alloc;
  ::mlir::memref::SubViewOp view;
  ::mlir::Value slot;
  ::mlir::memref::StoreOp toggle;
  unsigned producer;
  unsigned consumer;
};
} // namespace

/// The slot variable of `alloc`, if it is multibuffered in `nest`.
static std::optional<Slot> findSlot(::mlir::memref::AllocOp alloc,
                                    ::mlir::scf::ForOp nest) {
  using namespace ::mlir;
  auto view = alloc->hasOneUse()
                  ? dyn_cast<memref::SubViewOp>(*alloc->user_begin())
                  : memref::SubViewOp();
  if (!view || !nest->isProperAncestor(view))
    return std::nullopt;
  Value offset =
      llvm::dyn_cast_if_present<Value>(view.getMixedOffsets().front());
  auto index = offset ? offset.getDefiningOp<arith::IndexCastOp>()
                      : arith::IndexCastOp();
  auto load = index ? index.getIn().getDefiningOp<memref::LoadOp>()
                    : memref::LoadOp();
  if (!load)
    return std::nullopt;

  Slot slot{alloc, view, load.getMemRef(), load.getResult(), {}, {}};
  for (Operation *user : slot.var.getUsers())
    if (auto store = dyn_cast<memref::StoreOp>(user))
      (nest->isAncestor(store) ? slot.toggle : slot.init) = store;
  if (!slot.init || !slot.toggle)
    return std::nullopt;
  return slot;
}

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
    } else if (!isa<scf::ForOp, scf::YieldOp, memref::LoadOp, memref::StoreOp>(
                   op) &&
               !isMemoryEffectFree(op)) {
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
  SmallVector<Slot> slots;
  // The views of the shared buffers their producer also reads.
  SmallVector<std::pair<unsigned, memref::SubViewOp>> privates;
  for (auto &[alloc, stageIdxs] : users) {
    std::optional<Slot> slot = findSlot(alloc, nest);
    if (slot)
      slots.push_back(*slot);
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
          tokens.push_back({alloc, {}, {}, {}, writers.front(), consumer});
      continue;
    }
    if (!slot) {
      reason = "a shared buffer is not multibuffered in the nest";
      return failure();
    }
    // The producer switches the slot after writing the token, so there is one.
    if (stageIdxs.size() != 2) {
      reason = "a multibuffered buffer must have exactly one consumer";
      return failure();
    }
    for (unsigned consumer : stageIdxs)
      if (consumer != writers.front())
        tokens.push_back({alloc, slot->view, slot->value, slot->toggle,
                          writers.front(), consumer});
    if (stages[writers.front()].buffers[alloc].reads)
      privates.push_back({writers.front(), slot->view});
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
    // The stage writing a buffer owns its slot variable: the others have no
    // switch, and a consumer gets the slot from the token.
    auto ownsSlot = [&buffers = stage.buffers](Slot &slot) {
      return buffers.lookup(slot.alloc).writes;
    };
    for (Slot &slot : slots) {
      if (!ownsSlot(slot))
        continue;
      Value var = rewriter.create<memref::AllocaOp>(
          loc, cast<MemRefType>(slot.var.getType()));
      rewriter.create<memref::StoreOp>(
          loc, rewriter.create<arith::ConstantIntOp>(loc, 0, 8), var);
      map.map(slot.var, var);
    }
    rewriter.clone(*nest, map);
    for (auto [other, otherStage] : llvm::enumerate(stages))
      if (other != s)
        for (Operation *op : otherStage.ops)
          if (Operation *clone = map.lookupOrNull(op))
            rewriter.eraseOp(clone);
    for (Slot &slot : slots)
      if (!ownsSlot(slot))
        rewriter.eraseOp(map.lookup(slot.toggle.getOperation()));

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
      replaceBufferAndRebaseSubviews(rewriter, slotView.getResult(), buffer);
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
      Value peek = rewriter.create<df::StreamPeekOp>(loc, i8, stream);
      if (token.slot)
        rewriter.replaceAllUsesWith(map.lookup(token.slot), peek);
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
              ? map.lookup(token.slot)
              : rewriter.create<arith::ConstantIntOp>(loc, 0, 8).getResult();
      rewriter.create<df::StreamWriteOp>(loc, stream, slot);
      if (token.toggle) {
        Operation *toggle = map.lookup(token.toggle.getOperation());
        Value var = cast<memref::StoreOp>(toggle).getMemRef();
        rewriter.eraseOp(toggle);
        toggleSlot(rewriter, loc, var, slot, token.alloc.getType().getDimSize(0));
      }
    }

    // What the stage does not use: other stages' views, loops left empty.
    (void)runRegionDCE(rewriter, task->getRegions());
  }

  for (Operation *op : prologue)
    rewriter.eraseOp(op);
  rewriter.eraseOp(nest);
  for (Slot &slot : slots) {
    rewriter.eraseOp(slot.init);
    rewriter.eraseOp(slot.var.getDefiningOp());
  }
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
