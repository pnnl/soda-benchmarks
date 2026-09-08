//===- Utils.h - Affine access-pattern helpers ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_CONVERSION_AFFINETODATAFLOW_UTILS_H
#define SODAP_CONVERSION_AFFINETODATAFLOW_UTILS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace sodap {

/// Loop band, outermost first.
using AffineLoopBand = llvm::SmallVector<affine::AffineForOp, 6>;

/// Fill `band` with the loops enclosing `forOp`, outermost first, and return
/// the outermost one.
affine::AffineForOp getLoopBandFromInnermost(affine::AffineForOp forOp,
                                             AffineLoopBand &band);

/// Return a permutation describing the order in which the affine load or store
/// `op` traverses its buffer, relative to the loops enclosing it.
///
/// Two accesses that yield the same permutation walk their buffer in the same
/// order, which is the condition under which a FIFO can replace the buffer.
/// Returns a null AffineMap when the access cannot be characterised -- for
/// instance when it is indexed by something other than enclosing loop
/// induction variables. Callers are expected to treat that as "leave this IR
/// alone" rather than as an error.
AffineMap getMinimalAccessPattern(Operation *op);

} // namespace sodap
} // namespace mlir

#endif // SODAP_CONVERSION_AFFINETODATAFLOW_UTILS_H
