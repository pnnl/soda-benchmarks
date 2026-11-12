//===- Reuse.h - Static tile analysis for bounded reuse --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SODAP_DATAFLOW_REUSE_H
#define SODAP_DATAFLOW_REUSE_H
#include "Prefix.h"
#include "mlir/IR/IRMapping.h"
namespace mlir::sodap::dataflow {
/// The buffer `value` is a view of, and the views in between, innermost
/// first.
inline Value rootOf(Value value,
                    SmallVectorImpl<Operation *> *chain = nullptr) {
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
inline std::optional<Tile> tileOf(Value view, IRMapping &map,
                                  OpBuilder *builder) {
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

} // namespace mlir::sodap::dataflow
#endif
