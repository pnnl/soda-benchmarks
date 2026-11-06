//===- Prefix.h - The loops a task walks a shared buffer with ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The prefix of an access to a buffer shared by the tasks of a dispatch: the
// outer loops of the task stepping through the buffer a block at a time (see
// "Ping-pong buffers and the load scheduler" in docs/LinalgToLCSPlan.md).
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_DIALECT_DATAFLOW_TRANSFORMS_PREFIX_H
#define SODAP_DIALECT_DATAFLOW_TRANSFORMS_PREFIX_H

#include "sodap/Dialect/Dataflow/Dataflow.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/MapVector.h"

#include <array>

namespace mlir {
namespace sodap {
namespace dataflow {

/// A loop of a prefix, and the dim of the buffer it steps through.
struct Step {
  scf::ForOp loop;
  unsigned dim;
};
using Prefix = SmallVector<Step>;

/// An op of a task using the buffer, through `view`.
struct Access {
  Operation *op;
  Value view;
  bool writes;
};

/// Lower bound, upper bound and step, if constant.
inline std::optional<std::array<int64_t, 3>> boundsOf(scf::ForOp loop) {
  std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lb || !ub || !step)
    return std::nullopt;
  return std::array<int64_t, 3>{*lb, *ub, *step};
}

/// The loops of `task`, from the outermost, whose induction variable is the
/// offset of `access` along a dim of `buffer`, with the step as size.
inline Prefix prefixOf(const Access &access, Value buffer, TaskOp task) {
  auto type = cast<MemRefType>(buffer.getType());
  unsigned rank = type.getRank();
  SmallVector<memref::SubViewOp> chain;
  for (Value view = access.view; view != buffer;) {
    auto subview = view.getDefiningOp<memref::SubViewOp>();
    if (!subview || subview.getType().getRank() != rank ||
        !llvm::all_of(subview.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; }))
      return {};
    chain.push_back(subview);
    view = subview.getSource();
  }

  // Per dim: the one value the offset varies with (null if none, or if the
  // dim can't be a block dim), and the size of the access.
  SmallVector<Value> ivs(rank);
  SmallVector<int64_t> sizes(type.getShape());
  SmallVector<bool> fixed(rank, false);
  for (memref::SubViewOp subview : llvm::reverse(chain))
    for (unsigned d = 0; d < rank; ++d) {
      OpFoldResult offset = subview.getMixedOffsets()[d];
      if (std::optional<int64_t> constant = getConstantIntValue(offset))
        fixed[d] = fixed[d] || *constant != 0;
      else if (ivs[d])
        fixed[d] = true;
      else
        ivs[d] = offset.get<Value>();
      if (std::optional<int64_t> size =
              getConstantIntValue(subview.getMixedSizes()[d]))
        sizes[d] = *size;
      else
        fixed[d] = true;
    }

  SmallVector<scf::ForOp> loops;
  for (Operation *op = access.op->getParentOp(); op != task;
       op = op->getParentOp())
    if (auto loop = dyn_cast<scf::ForOp>(op))
      loops.push_back(loop);
  Prefix prefix;
  for (scf::ForOp loop : llvm::reverse(loops)) {
    auto bounds = boundsOf(loop);
    auto *it = llvm::find(ivs, loop.getInductionVar());
    if (!bounds || it == ivs.end())
      break;
    unsigned d = it - ivs.begin();
    if (fixed[d] || sizes[d] != (*bounds)[2])
      break;
    prefix.push_back({loop, d});
  }
  return prefix;
}

/// The prefix all the accesses of a task share.
inline Prefix prefixOf(ArrayRef<Access> accesses, Value buffer, TaskOp task) {
  Prefix common = prefixOf(accesses.front(), buffer, task);
  for (const Access &access : accesses.drop_front()) {
    Prefix prefix = prefixOf(access, buffer, task);
    unsigned n = 0;
    while (n < common.size() && n < prefix.size() &&
           common[n].loop == prefix[n].loop && common[n].dim == prefix[n].dim)
      ++n;
    common.resize(n);
  }
  return common;
}

/// Cuts the prefixes of a writer and a reader where they stop agreeing (the
/// same dims, over loops with the same bounds): their shared prefix.
inline void sharePrefix(Prefix &writes, Prefix &reads) {
  unsigned n = 0;
  while (n < writes.size() && n < reads.size() &&
         writes[n].dim == reads[n].dim &&
         boundsOf(writes[n].loop) == boundsOf(reads[n].loop))
    ++n;
  writes.resize(n);
  reads.resize(n);
}

/// Replace a whole copy with a loop nest over `steps`, copying a block per
/// iteration. Returns the block copy.
inline memref::CopyOp splitCopy(OpBuilder &builder, memref::CopyOp copy,
                                ArrayRef<Step> steps) {
  if (steps.empty())
    return copy;
  Location loc = copy.getLoc();
  auto type = cast<MemRefType>(copy.getSource().getType());
  SmallVector<OpFoldResult> offsets(type.getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(type.getRank(), builder.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes;
  for (int64_t size : type.getShape())
    sizes.push_back(builder.getIndexAttr(size));
  builder.setInsertionPoint(copy);
  for (Step step : steps) {
    auto [lb, ub, stride] = *boundsOf(step.loop);
    auto loop = builder.create<scf::ForOp>(
        loc, builder.create<arith::ConstantIndexOp>(loc, lb),
        builder.create<arith::ConstantIndexOp>(loc, ub),
        builder.create<arith::ConstantIndexOp>(loc, stride));
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    offsets[step.dim] = loop.getInductionVar();
    sizes[step.dim] = builder.getIndexAttr(stride);
  }
  Value source = builder.create<memref::SubViewOp>(loc, copy.getSource(),
                                                   offsets, sizes, strides);
  Value target = builder.create<memref::SubViewOp>(loc, copy.getTarget(),
                                                   offsets, sizes, strides);
  auto block = builder.create<memref::CopyOp>(loc, source, target);
  copy.erase();
  return block;
}

/// The accesses of each task of `dispatch` to `buffer`, through views of it.
inline FailureOr<llvm::MapVector<TaskOp, SmallVector<Access>>>
accessesOf(Value buffer, DispatchOp dispatch) {
  Block &body = dispatch.getBody().front();
  llvm::MapVector<TaskOp, SmallVector<Access>> accesses;
  SmallVector<Value> views{buffer};
  while (!views.empty()) {
    Value view = views.pop_back_val();
    for (OpOperand &use : view.getUses()) {
      Operation *op = use.getOwner();
      if (auto viewOp = dyn_cast<ViewLikeOpInterface>(op);
          viewOp && viewOp.getViewSource() == view) {
        views.push_back(op->getResult(0));
        continue;
      }
      auto task = dyn_cast_or_null<TaskOp>(body.findAncestorOpInBlock(*op));
      if (!task) {
        if (isa<YieldOp>(op))
          continue;
        op->emitOpError("expected the buffers of a dispatch to be used in "
                        "its tasks");
        return failure();
      }
      accesses[task].push_back(
          {op, view, hasEffect<MemoryEffects::Write>(op, view)});
    }
  }
  return accesses;
}

} // namespace dataflow
} // namespace sodap
} // namespace mlir

#endif // SODAP_DIALECT_DATAFLOW_TRANSFORMS_PREFIX_H
