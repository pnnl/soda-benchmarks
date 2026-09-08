//===- Utils.cpp - Affine access-pattern helpers --------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sodap/Conversion/AffineToDataflow/Utils.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::sodap;
using namespace mlir::affine;

namespace {

/// Rewrite an access map's symbols as trailing dimensions, so that symbolic
/// and dimensional operands can be treated uniformly below.
AffineMap fixAccessMap(AffineMap map) {
  SmallVector<AffineExpr, 2> replacements;
  for (unsigned i = 0, e = map.getNumSymbols(); i < e; ++i)
    replacements.push_back(
        getAffineDimExpr(map.getNumDims() + i, map.getContext()));
  return map.replaceDimsAndSymbols(/*dimReplacements=*/{}, replacements,
                                   map.getNumDims() + map.getNumSymbols(),
                                   /*numResultSyms=*/0);
}

/// Return true when `values` is a permutation of [0, values.size()).
bool isPermutation(ArrayRef<unsigned> values) {
  llvm::SmallDenseSet<unsigned> seen;
  for (auto v : values) {
    if (v >= values.size() || !seen.insert(v).second)
      return false;
  }
  return true;
}

} // namespace

AffineForOp sodap::getLoopBandFromInnermost(AffineForOp forOp,
                                            AffineLoopBand &band) {
  AffineLoopBand reverseBand;
  auto current = forOp;
  while (current) {
    reverseBand.push_back(current);
    auto parent = current->getParentOfType<AffineForOp>();
    if (!parent)
      break;
    current = parent;
  }
  band.assign(reverseBand.rbegin(), reverseBand.rend());
  return band.empty() ? AffineForOp() : band.front();
}

AffineMap sodap::getMinimalAccessPattern(Operation *op) {
  SmallVector<Value, 4> mapOperands;
  AffineMap accessMap;
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    mapOperands = loadOp.getMapOperands();
    accessMap = loadOp.getAffineMap();
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    mapOperands = storeOp.getMapOperands();
    accessMap = storeOp.getAffineMap();
  } else {
    return AffineMap();
  }

  if (accessMap.getNumSymbols() > 0)
    accessMap = fixAccessMap(accessMap);

  auto innermost = op->getParentOfType<AffineForOp>();
  if (!innermost)
    return AffineMap();

  AffineLoopBand band;
  getLoopBandFromInnermost(innermost, band);

  // Keep the enclosing loops whose induction variable actually indexes the
  // buffer, in loop-nest order.
  SmallVector<Value, 4> relevantOperands;
  for (auto forOp : band) {
    auto iv = forOp.getInductionVar();
    if (llvm::is_contained(mapOperands, iv))
      relevantOperands.push_back(iv);
  }

  // Anything indexed by a value that is not an enclosing induction variable is
  // outside what this analysis can describe.
  if (relevantOperands.size() != mapOperands.size())
    return AffineMap();

  // Map each access operand to its position in loop-nest order.
  SmallVector<unsigned, 4> loopOrder;
  for (auto operand : mapOperands) {
    auto it = llvm::find(relevantOperands, operand);
    if (it == relevantOperands.end())
      return AffineMap();
    loopOrder.push_back(std::distance(relevantOperands.begin(), it));
  }
  if (!isPermutation(loopOrder))
    return AffineMap();

  auto reordered = accessMap.compose(
      AffineMap::getPermutationMap(loopOrder, op->getContext()));

  // The order in which the buffer's dimensions vary, outermost loop first.
  SmallVector<unsigned, 4> locations;
  for (auto expr : reordered.getResults()) {
    expr.walk([&](AffineExpr e) {
      if (auto dim = llvm::dyn_cast<AffineDimExpr>(e)) {
        unsigned pos = dim.getPosition();
        if (!llvm::is_contained(locations, pos))
          locations.push_back(pos);
      }
    });
  }

  if (!isPermutation(locations))
    return AffineMap();

  return AffineMap::getPermutationMap(locations, op->getContext());
}
