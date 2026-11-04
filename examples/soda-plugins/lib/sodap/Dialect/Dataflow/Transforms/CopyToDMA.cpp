//===- CopyToDMA.cpp - Copies of the top's tiles to DMA transfers ---------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replaces the memref arguments of the top with the four streams of a DMA
// engine, and every copy of one of their tiles with the transfer of the tile:
// a request per row, then the row's data.
//
// The arguments form one address space, in words, in argument order: for
// matmul(A: 8x12, B: 12x16, C: 8x16), A starts at 0, B at 96, C at 288. A 4x4
// tile of A at (i, k) is then 4 requests, one per row r, at i*12 + k + r*12,
// each of length 4. A copy of a whole argument is a tile of it at 0; a
// scalar, memref<f32>, is one request of length 1.
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_COPYTODMA
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

#define DEBUG_TYPE "sodap-dataflow-copy-to-dma"

using namespace mlir;
using namespace mlir::sodap;
using namespace mlir::sodap::dataflow;

namespace {

/// The four streams of the DMA engine.
struct DMAStreams {
  Value readCtrl;
  Value readData;
  Value writeCtrl;
  Value writeData;
};

/// One copy of a tile of a top argument, and where that argument starts.
struct Transfer {
  memref::CopyOp copy;
  memref::SubViewOp tile; // Null for a copy of the whole argument.
  int64_t base;
  bool read;
};

} // namespace

/// Checks that `arg` is a DMA region the pass can handle and collects the
/// copies of its tiles.
static LogicalResult collectTransfers(BlockArgument arg, int64_t base,
                                      SmallVectorImpl<Transfer> &transfers) {
  auto type = cast<MemRefType>(arg.getType());
  Location loc = arg.getLoc();
  if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
      type.getElementTypeBitWidth() != 32)
    return emitError(loc) << "argument #" << arg.getArgNumber()
                          << " is not a static 32-bit memref with the "
                             "identity layout: "
                          << type;

  for (Operation *user : arg.getUsers()) {
    if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      transfers.push_back({copy, {}, base, copy.getSource() == arg});
      continue;
    }
    auto tile = dyn_cast<memref::SubViewOp>(user);
    if (!tile || !tile.hasUnitStride() ||
        ShapedType::isDynamicShape(tile.getStaticSizes()))
      return user->emitError()
             << "argument #" << arg.getArgNumber()
             << " is used other than by a copy or a static tile with unit "
                "strides";
    for (Operation *tileUser : tile->getUsers()) {
      auto copy = dyn_cast<memref::CopyOp>(tileUser);
      if (!copy)
        return tileUser->emitError()
               << "a tile of argument #" << arg.getArgNumber()
               << " is used other than by a memref.copy";
      transfers.push_back({copy, tile, base, copy.getSource() == tile});
    }
  }
  return success();
}

/// Emits `transfer` in place of its copy: for each row of the tile along the
/// innermost dimension, the request, then the row's elements.
static void lowerTransfer(OpBuilder &b, Transfer &transfer,
                          const DMAStreams &dma, unsigned wordStride) {
  memref::SubViewOp tile = transfer.tile;
  Location loc = transfer.copy.getLoc();
  Value buffer = transfer.read ? transfer.copy.getTarget()
                               : transfer.copy.getSource();
  // A whole argument: a tile the shape of the buffer, at 0.
  auto argType = tile ? tile.getSourceType()
                      : cast<MemRefType>(buffer.getType());
  SmallVector<int64_t> sizes = llvm::to_vector(
      tile ? tile.getStaticSizes() : argType.getShape());
  SmallVector<OpFoldResult> offsets(argType.getRank(), b.getIndexAttr(0));
  if (tile)
    offsets = tile.getMixedOffsets();
  SmallVector<int64_t> strides = computeSuffixProduct(argType.getShape());
  Value ctrl = transfer.read ? dma.readCtrl : dma.writeCtrl;
  Value data = transfer.read ? dma.readData : dma.writeData;
  unsigned rank = sizes.size();
  b.setInsertionPoint(transfer.copy);

  // A loop per outer dimension of the tile: one request each.
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> ivs;
  for (unsigned d = 0; d + 1 < rank; ++d) {
    Value size = b.create<arith::ConstantIndexOp>(loc, sizes[d]);
    auto loop = b.create<scf::ForOp>(loc, c0, size, c1);
    b.setInsertionPointToStart(loop.getBody());
    ivs.push_back(loop.getInductionVar());
  }

  // address = base + sum_d (offset_d + iv_d) * stride_d, over the dims
  // (offsets..., ivs...); the innermost dimension has no iv here.
  SmallVector<Value> operands =
      getValueOrCreateConstantIndexOp(b, loc, offsets);
  llvm::append_range(operands, ivs);
  AffineExpr address = b.getAffineConstantExpr(transfer.base);
  for (unsigned d = 0; d < rank; ++d)
    address = address + b.getAffineDimExpr(d) * (strides[d] * wordStride);
  for (unsigned d = 0; d + 1 < rank; ++d)
    address =
        address + b.getAffineDimExpr(rank + d) * (strides[d] * wordStride);
  Value addr = b.create<affine::AffineApplyOp>(
      loc, AffineMap::get(operands.size(), 0, address), operands);

  // A scalar is a row of one element, with no index.
  int64_t length = rank ? sizes.back() : 1;
  Value request = b.create<arith::OrIOp>(
      loc, b.create<arith::ConstantIntOp>(loc, (length * wordStride) << 32, 64),
      b.create<arith::IndexCastOp>(loc, b.getI64Type(), addr));
  b.create<StreamWriteOp>(loc, ctrl, request);

  Value rowLength = b.create<arith::ConstantIndexOp>(loc, length);
  auto row = b.create<scf::ForOp>(loc, c0, rowLength, c1);
  b.setInsertionPointToStart(row.getBody());
  SmallVector<Value> indices(ivs);
  if (rank)
    indices.push_back(row.getInductionVar());
  Type element = cast<MemRefType>(buffer.getType()).getElementType();
  Type word = b.getI32Type();
  if (transfer.read) {
    Value value = b.create<StreamReadOp>(loc, word, data).getResult();
    if (wordStride == 2)
      b.create<StreamReadOp>(loc, word, data); // Discard the padding lane.
    if (element != word)
      value = b.create<arith::BitcastOp>(loc, element, value);
    b.create<memref::StoreOp>(loc, value, buffer, indices);
  } else {
    Value value = b.create<memref::LoadOp>(loc, buffer, indices);
    if (element != word)
      value = b.create<arith::BitcastOp>(loc, word, value);
    b.create<StreamWriteOp>(loc, data, value);
    if (wordStride == 2)
      b.create<StreamWriteOp>(loc, data,
                              b.create<arith::ConstantIntOp>(loc, 0, 32));
  }
}

namespace {

struct CopyToDMA : public dataflow::impl::CopyToDMABase<CopyToDMA> {
  using CopyToDMABase::CopyToDMABase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (wordStride != 1 && wordStride != 2) {
      module.emitError("dma-word-stride must be 1 or 2");
      return signalPassFailure();
    }
    auto top = module.lookupSymbol<func::FuncOp>(topFunc);
    if (!top || top.isExternal()) {
      module.emitError() << "top function '" << topFunc << "' not found";
      return signalPassFailure();
    }

    // The memref arguments, laid out one after the other.
    SmallVector<Transfer> transfers;
    llvm::BitVector regions(top.getNumArguments());
    int64_t base = 0;
    for (BlockArgument arg : top.getArguments()) {
      auto type = dyn_cast<MemRefType>(arg.getType());
      if (!type)
        continue;
      if (failed(collectTransfers(arg, base, transfers)))
        return signalPassFailure();
      regions.set(arg.getArgNumber());
      base += type.getNumElements() * wordStride;
    }

    // A copy between two DMA regions needs local staging. Reject it before
    // mutating anything, rather than lowering/erasing the same copy twice.
    llvm::SmallPtrSet<Operation *, 16> copies;
    llvm::SetVector<Operation *> tiles;
    for (Transfer &transfer : transfers) {
      if (!copies.insert(transfer.copy.getOperation()).second) {
        transfer.copy.emitError("a DMA copy must have exactly one top-level "
                                "memory operand; stage argument-to-argument "
                                "copies through a local buffer");
        return signalPassFailure();
      }
      if (transfer.tile)
        tiles.insert(transfer.tile.getOperation());
    }

    MLIRContext *ctx = &getContext();
    Location loc = top.getLoc();
    auto ctrlType = StreamType::get(ctx, IntegerType::get(ctx, 64), 8);
    auto dataType = StreamType::get(ctx, IntegerType::get(ctx, 32), 16);
    unsigned next = top.getNumArguments();
    for (Type type : {ctrlType, dataType, ctrlType, dataType})
      top.insertArgument(next++, type, DictionaryAttr::get(ctx), loc);
    unsigned first = top.getNumArguments() - 4;
    DMAStreams dma{top.getArgument(first), top.getArgument(first + 1),
                   top.getArgument(first + 2), top.getArgument(first + 3)};

    OpBuilder b(ctx);
    for (Transfer &transfer : transfers) {
      lowerTransfer(b, transfer, dma, wordStride);
      transfer.copy.erase();
    }
    for (Operation *tile : tiles)
      if (tile->use_empty())
        tile->erase();

    regions.resize(top.getNumArguments());
    top.eraseArguments(regions);
  }
};

} // namespace
