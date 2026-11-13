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
// A tile from sodap-join-multiple-tiles (arguments with
// sodap.descriptor_field) has no fixed layout: the load task first reads the
// descriptor (ReusableGemmABI.h) at address 0 and forwards it to the other
// tasks on a stream each. The index arguments become its fields, and a tile
// of a matrix starts at its field's address, rows its row-stride field apart.
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"
#include "sodap/ExecutionEngine/ReusableGemmABI.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/ScopeExit.h"
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
  /// With a descriptor: the fields of the argument's address and row stride.
  int64_t baseField = -1, ldField = -1;
};

} // namespace

/// Checks that `arg` is a DMA region the pass can handle and collects the
/// copies of its tiles.
static LogicalResult collectTransfers(BlockArgument arg, int64_t base,
                                      SmallVectorImpl<Transfer> &transfers) {
  auto type = cast<MemRefType>(arg.getType());
  Location loc = arg.getLoc();
  auto func = cast<func::FuncOp>(arg.getOwner()->getParentOp());
  auto field = func.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                                  "sodap.descriptor_field");
  auto ld = func.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                               "sodap.descriptor_ld");
  if (field && (!ld || type.getRank() != 2))
    return emitError(loc) << "argument #" << arg.getArgNumber()
                          << ": a descriptor field needs a matrix with "
                             "sodap.descriptor_ld";
  if ((!field && !type.hasStaticShape()) || !type.getLayout().isIdentity() ||
      type.getElementTypeBitWidth() != 32)
    return emitError(loc) << "argument #" << arg.getArgNumber()
                          << " is not a static 32-bit memref with the "
                             "identity layout: "
                          << type;

  size_t first = transfers.size();
  auto described = llvm::make_scope_exit([&] {
    if (field)
      for (Transfer &t : llvm::drop_begin(transfers, first)) {
        t.baseField = field.getInt();
        t.ldField = ld.getInt();
      }
  });
  for (Operation *user : arg.getUsers()) {
    if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      if (field)
        return copy.emitError("a copy of a whole matrix with a descriptor "
                              "field: expected tiles");
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
                          const DMAStreams &dma, unsigned wordStride,
                          ArrayRef<Value> fields = {}) {
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
  SmallVector<int64_t> strides;
  if (transfer.baseField < 0)
    strides = computeSuffixProduct(argType.getShape());
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
  Value addr;
  if (transfer.baseField >= 0) {
    // From the descriptor: base + (row offset + iv) * ld + col offset * 2.
    Value row = b.create<arith::AddIOp>(loc, operands[0], ivs[0]);
    Value col = b.create<arith::MulIOp>(
        loc, operands[1], b.create<arith::ConstantIndexOp>(loc, wordStride));
    addr = b.create<arith::AddIOp>(
        loc, fields[transfer.baseField],
        b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(loc, row, fields[transfer.ldField]),
            col));
  } else {
    AffineExpr address = b.getAffineConstantExpr(transfer.base);
    for (unsigned d = 0; d < rank; ++d)
      address = address + b.getAffineDimExpr(d) * (strides[d] * wordStride);
    for (unsigned d = 0; d + 1 < rank; ++d)
      address =
          address + b.getAffineDimExpr(rank + d) * (strides[d] * wordStride);
    addr = b.create<affine::AffineApplyOp>(
        loc, AffineMap::get(operands.size(), 0, address), operands);
  }

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

/// Each task of `dispatch` gets the descriptor at its start: the load task
/// reads it from the DMA at address 0 (one 64-bit slot per field) and writes
/// it to a stream per other task, which read it from there. Returns each
/// task's fields, as index values.
static FailureOr<DenseMap<Operation *, SmallVector<Value>>>
readDescriptor(OpBuilder &b, DispatchOp dispatch, const DMAStreams &dma) {
  MLIRContext *ctx = b.getContext();
  Location loc = dispatch.getLoc();
  Block &body = dispatch.getBody().front();
  SmallVector<TaskOp> tasks = llvm::to_vector(body.getOps<TaskOp>());
  auto load = llvm::find_if(tasks, [](TaskOp t) { return t->hasAttr("load"); });
  if (load == tasks.end())
    return dispatch.emitOpError("expected a load task to read the descriptor");
  Type i32 = b.getI32Type();
  b.setInsertionPointToStart(&body);
  DenseMap<Operation *, Value> streams;
  for (TaskOp task : tasks)
    if (task != *load)
      streams[task] = b.create<StreamOp>(
          loc, StreamType::get(ctx, i32, GEMM_FIELD_COUNT));

  DenseMap<Operation *, SmallVector<Value>> fields;
  for (TaskOp task : tasks) {
    b.setInsertionPointToStart(&task.getBody().front());
    Value local = b.create<memref::AllocaOp>(
        loc, MemRefType::get({GEMM_FIELD_COUNT}, i32));
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    Value count = b.create<arith::ConstantIndexOp>(loc, GEMM_FIELD_COUNT);
    if (task == *load)
      b.create<StreamWriteOp>(
          loc, dma.readCtrl,
          b.create<arith::ConstantIntOp>(
              loc, int64_t(GEMM_FIELD_COUNT * GEMM_WORD_STRIDE) << 32, 64));
    auto loop = b.create<scf::ForOp>(loc, c0, count, c1);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value value;
      if (task == *load) {
        value = b.create<StreamReadOp>(loc, i32, dma.readData).getResult();
        b.create<StreamReadOp>(loc, i32, dma.readData); // The padding lane.
        for (TaskOp other : tasks)
          if (other != *load)
            b.create<StreamWriteOp>(loc, streams[other], value);
      } else {
        value = b.create<StreamReadOp>(loc, i32, streams[task]).getResult();
      }
      b.create<memref::StoreOp>(loc, value, local, loop.getInductionVar());
    }
    // ponytail: no magic/version check; add William's guard (no work on an
    // invalid descriptor) once the runtime can send one.
    for (int64_t f = 0; f < GEMM_FIELD_COUNT; ++f) {
      Value word = b.create<memref::LoadOp>(
          loc, local, ValueRange{b.create<arith::ConstantIndexOp>(loc, f)});
      fields[task].push_back(
          b.create<arith::IndexCastUIOp>(loc, b.getIndexType(), word));
    }
  }
  return fields;
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
    std::string topName = dataflow::getTopName(module, topFunc);
    auto top = module.lookupSymbol<func::FuncOp>(topName);
    if (!top || top.isExternal()) {
      module.emitError() << "top function '" << topName << "' not found";
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
      if (type.hasStaticShape())
        base += type.getNumElements() * wordStride;
    }
    // The index arguments read from the descriptor.
    SmallVector<std::pair<BlockArgument, int64_t>> scalars;
    for (BlockArgument arg : top.getArguments())
      if (auto field = top.getArgAttrOfType<IntegerAttr>(
              arg.getArgNumber(), "sodap.descriptor_field")) {
        if (isa<IndexType>(arg.getType())) {
          scalars.push_back({arg, field.getInt()});
          regions.set(arg.getArgNumber());
        }
      }
    bool descriptor = llvm::any_of(
        transfers, [](const Transfer &t) { return t.baseField >= 0; });
    if (descriptor && wordStride != GEMM_WORD_STRIDE) {
      module.emitError("a descriptor needs dma-word-stride=2");
      return signalPassFailure();
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
    auto ctrlType = StreamType::get(ctx, IntegerType::get(ctx, 64), 4);
    auto dataType = StreamType::get(ctx, IntegerType::get(ctx, 32), 1);
    unsigned next = top.getNumArguments();
    for (Type type : {ctrlType, dataType, ctrlType, dataType})
      top.insertArgument(next++, type, DictionaryAttr::get(ctx), loc);
    unsigned first = top.getNumArguments() - 4;
    DMAStreams dma{top.getArgument(first), top.getArgument(first + 1),
                   top.getArgument(first + 2), top.getArgument(first + 3)};

    OpBuilder b(ctx);
    DenseMap<Operation *, SmallVector<Value>> fields;
    if (descriptor) {
      auto dispatches = llvm::to_vector(top.front().getOps<DispatchOp>());
      if (dispatches.size() != 1) {
        top.emitError("a descriptor needs one dataflow.dispatch");
        return signalPassFailure();
      }
      auto read = readDescriptor(b, dispatches.front(), dma);
      if (failed(read))
        return signalPassFailure();
      fields = std::move(*read);
      Block &body = dispatches.front().getBody().front();
      for (auto [arg, field] : scalars)
        for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
          Operation *task = body.findAncestorOpInBlock(*use.getOwner());
          if (!task || !fields.count(task)) {
            use.getOwner()->emitError("a use of a descriptor field outside "
                                      "the tasks");
            return signalPassFailure();
          }
          use.set(fields[task][field]);
        }
    }
    llvm::SmallPtrSet<Operation *, 4> receivers;
    llvm::SmallPtrSet<Operation *, 4> writers;
    for (Transfer &transfer : transfers) {
      if (transfer.read)
        if (auto task = transfer.copy->getParentOfType<TaskOp>())
          receivers.insert(task);
      if (!transfer.read)
        if (auto task = transfer.copy->getParentOfType<TaskOp>())
          writers.insert(task);
      ArrayRef<Value> taskFields;
      if (transfer.baseField >= 0)
        taskFields = fields[transfer.copy->getParentOfType<TaskOp>()];
      lowerTransfer(b, transfer, dma, wordStride, taskFields);
      transfer.copy.erase();
    }
    // A stalled tile publication must complete before the receiver admits
    // another DMA burst. Different ac_channel writes otherwise have no data
    // dependence, and HLS may issue a request while the receiver is blocked
    // publishing the preceding tile. Lower these writes with an explicit
    // accepted-result loop, making completion a control-flow dependency.
    for (Operation *receiver : receivers)
      receiver->walk([&](StreamWriteOp write) {
        write->setAttr("sodap.dma_ordered", b.getUnitAttr());
      });
    // The store's peek acquires a ready output tile while retaining its
    // ownership token until the copy finishes. Address-only DMA commands
    // otherwise have no data dependency on this peek and HLS may issue them
    // before compute has produced the tile. Make readiness an explicit
    // control-flow barrier; keep the existing PLM and token lifetime.
    for (Operation *writer : writers)
      writer->walk([&](StreamPeekOp peek) {
        peek->setAttr("sodap.dma_ordered", b.getUnitAttr());
      });
    for (Operation *tile : tiles)
      if (tile->use_empty())
        tile->erase();

    regions.resize(top.getNumArguments());
    top.eraseArguments(regions);
  }
};

} // namespace
