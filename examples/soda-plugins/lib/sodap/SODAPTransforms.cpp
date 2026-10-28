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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
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

/// The subview of a function argument that `operand` reads or writes, if any.
static ::mlir::memref::SubViewOp argumentTile(::mlir::OpOperand &operand) {
  using namespace ::mlir;
  auto subview = operand.get().getDefiningOp<memref::SubViewOp>();
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
  // The linalg operands of each tile, in program order.
  struct Tile {
    SmallVector<OpOperand *> uses;
    bool copyIn = false;
    Operation *lastWriter = nullptr;
  };
  llvm::MapVector<memref::SubViewOp, Tile> tiles;

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
      memref::SubViewOp subview = argumentTile(operand);
      if (!subview)
        continue;
      if (!subview.getType().hasStaticShape()) {
        DiagnosedSilenceableFailure diag = emitSilenceableError()
                                           << "cannot promote a tile with a "
                                              "dynamic shape";
        diag.attachNote(subview.getLoc()) << "tile";
        return diag;
      }
      bool reads = linalgOp.isDpsInput(&operand) ||
                   linalgOp.payloadUsesValueFromOperand(&operand);
      Tile &tile = tiles[subview];
      // Copy in only if the first user reads what is in memory.
      if (tile.uses.empty())
        tile.copyIn = reads;
      if (linalgOp.isDpsInit(&operand))
        tile.lastWriter = op;
      tile.uses.push_back(&operand);
    }
  }

  for (auto &[subview, tile] : tiles) {
    MemRefType type = subview.getType();
    Location loc = subview.getLoc();
    rewriter.setInsertionPoint(outermostLoop(tile.uses.front()->getOwner()));
    Value buffer = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(type.getShape(), type.getElementType()));

    // The tile is the same for the whole block of the subview, so the copies
    // wrap the users at that level: a C tile defined in the j loop is copied
    // around the k loop, not at every k.
    Block *block = subview->getBlock();
    if (tile.copyIn) {
      rewriter.setInsertionPoint(
          block->findAncestorOpInBlock(*tile.uses.front()->getOwner()));
      rewriter.create<memref::CopyOp>(loc, subview, buffer);
    }
    if (tile.lastWriter) {
      rewriter.setInsertionPointAfter(
          block->findAncestorOpInBlock(*tile.lastWriter));
      rewriter.create<memref::CopyOp>(loc, buffer, subview);
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

namespace mlir {
namespace sodap {
void registerSODAPTransforms(::mlir::DialectRegistry &registry) {
  registry.addExtensions<SODAPTransformsExtension>();
}
} // namespace sodap
} // namespace mlir
