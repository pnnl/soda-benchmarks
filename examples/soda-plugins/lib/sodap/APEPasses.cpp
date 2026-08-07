//===- APEPasses.cpp - Affine Prefetching Engine passes --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"

#include <functional>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include "sodap/SODAPPasses.h"

using namespace mlir;

namespace mlir::sodap {
#define GEN_PASS_DEF_LINALGAPEANALYSIS
#define GEN_PASS_DEF_AFFINEAPEINSERTION
#define GEN_PASS_DEF_GENADDRFUNCTIONPASS
#include "sodap/SODAPPasses.h.inc"

namespace {

constexpr llvm::StringLiteral kIssueAPERequest = "issue_APE_request";
constexpr llvm::StringLiteral kPreIssueAPERequest = "pre_issue_APE_request";
constexpr llvm::StringLiteral kAPETensorInfoAttr = "ape.tensor_info";

// addr_gen2.py C runtime entry points
constexpr llvm::StringLiteral kApeIncomplete  = "ape_incomplete";
constexpr llvm::StringLiteral kApeBroadcast   = "ape_broadcast";
constexpr llvm::StringLiteral kApeGather      = "ape_gather_from_memref";
constexpr llvm::StringLiteral kApeElementwise = "ape_elementwise_trace";

constexpr int64_t CACHE_CAPACITY = 20; //32 * 1024; // 32KB L1 cache

static func::FuncOp ensureIssueAPERequestDeclaration(ModuleOp module) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kIssueAPERequest))
    return existing;

  OpBuilder moduleBuilder(module.getBodyRegion());
  MLIRContext *ctx = module.getContext();

  auto i32Type = moduleBuilder.getI32Type();
  auto i64Type = moduleBuilder.getI64Type();
  auto i1Type = moduleBuilder.getI1Type();
  auto indexType = moduleBuilder.getIndexType();

  SmallVector<Type, 10> inputs = {
      indexType, // base_addr
      i32Type,   // tensor_id
      i32Type,   // strategy
      i1Type,    // is_write
      i64Type,   // element_bytes
      i64Type,   // total_elements
      i64Type,   // rank
      indexType, // shape_desc_ptr
      indexType, // strides_desc_ptr
  };

  auto fnType = FunctionType::get(ctx, inputs, TypeRange{});
  auto fn = moduleBuilder.create<func::FuncOp>(module.getLoc(), kIssueAPERequest,
                                               fnType);
  fn.setPrivate();
  return fn;
}

static func::FuncOp ensurePreIssueAPERequestDeclaration(ModuleOp module) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kPreIssueAPERequest))
    return existing;

  OpBuilder moduleBuilder(module.getBodyRegion());
  MLIRContext *ctx = module.getContext();
  auto fnType = FunctionType::get(ctx, TypeRange{}, TypeRange{});
  auto fn = moduleBuilder.create<func::FuncOp>(module.getLoc(),
                                               kPreIssueAPERequest, fnType);
  fn.setPrivate();
  return fn;
}

static func::FuncOp ensureApeIncompleteDecl(ModuleOp module) {
  if (auto f = module.lookupSymbol<func::FuncOp>(kApeIncomplete)) return f;
  OpBuilder b(module.getBodyRegion());
  auto fn = b.create<func::FuncOp>(module.getLoc(), kApeIncomplete,
      FunctionType::get(module.getContext(), TypeRange{}, TypeRange{}));
  fn.setPrivate();
  return fn;
}

static func::FuncOp ensureApeBroadcastDecl(ModuleOp module) {
  if (auto f = module.lookupSymbol<func::FuncOp>(kApeBroadcast)) return f;
  OpBuilder b(module.getBodyRegion());
  MLIRContext *ctx = module.getContext();
  // (aligned, offset, s0, s1, str0, str1, elem_bytes) — all i64
  SmallVector<Type, 7> args(7, IntegerType::get(ctx, 64));
  auto fn = b.create<func::FuncOp>(module.getLoc(), kApeBroadcast,
      FunctionType::get(ctx, args, TypeRange{}));
  fn.setPrivate();
  return fn;
}

static func::FuncOp ensureApeGatherDecl(ModuleOp module) {
  if (auto f = module.lookupSymbol<func::FuncOp>(kApeGather)) return f;
  OpBuilder b(module.getBodyRegion());
  MLIRContext *ctx = module.getContext();
  // 3 memrefs × (aligned,offset,s0,s1,str0,str1) + L1_cache_size + elem_bytes = 20 i64
  SmallVector<Type, 20> args(20, IntegerType::get(ctx, 64));
  auto fn = b.create<func::FuncOp>(module.getLoc(), kApeGather,
      FunctionType::get(ctx, args, TypeRange{}));
  fn.setPrivate();
  return fn;
}

static func::FuncOp ensureApeElementwiseDecl(ModuleOp module) {
  if (auto f = module.lookupSymbol<func::FuncOp>(kApeElementwise)) return f;
  OpBuilder b(module.getBodyRegion());
  MLIRContext *ctx = module.getContext();
  // (n_operands: i64, L1_cache_size: i64, elem_bytes: i64, desc_ptr: index)
  SmallVector<Type, 4> args = {
      IntegerType::get(ctx, 64), IntegerType::get(ctx, 64),
      IntegerType::get(ctx, 64), IndexType::get(ctx)};
  auto fn = b.create<func::FuncOp>(module.getLoc(), kApeElementwise,
      FunctionType::get(ctx, args, TypeRange{}));
  fn.setPrivate();
  return fn;
}

static int64_t getElementByteWidth(Type elementType) {
  if (auto intTy = dyn_cast<IntegerType>(elementType))
    return std::max<int64_t>(1, intTy.getWidth() / 8);
  if (auto floatTy = dyn_cast<FloatType>(elementType))
    return std::max<int64_t>(1, floatTy.getWidth() / 8);
  return 0;
}

static int32_t inferContiguousVectorLength(MemRefType memrefType,
                                           int64_t elementBytes) {
  if (!memrefType.hasRank() || memrefType.getRank() == 0)
    return 1;
  int64_t last = memrefType.getShape().back();
  if (ShapedType::isDynamic(last))
    return 4;
  if (elementBytes <= 0)
    return 1;

  // Heuristic: request chunks aligned to 16B (common SIMD / burst granularity).
  int64_t elemsPerChunk = std::max<int64_t>(1, 16 / elementBytes);
  return static_cast<int32_t>(std::max<int64_t>(1, std::min(last, elemsPerChunk)));
}

static StringRef classifyAccessPattern(AffineMap indexingMap) {
  if (indexingMap.getNumResults() == 0)
    return "unknown";

  // Constant-indexed dimensions indicate broadcast-style access.
  for (AffineExpr e : indexingMap.getResults()) {
    if (isa<AffineConstantExpr>(e))
      return "broadcast";
  }

  bool allDimExpr = true;
  llvm::DenseSet<unsigned> usedDims;
  SmallVector<unsigned, 8> dimOrder;
  for (AffineExpr e : indexingMap.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(e);
    if (!dim) {
      allDimExpr = false;
      break;
    }
    unsigned pos = dim.getPosition();
    usedDims.insert(pos);
    dimOrder.push_back(pos);
  }

  if (!allDimExpr)
    return "strided";

  if (usedDims.size() != indexingMap.getNumResults())
    return "strided";

  bool isIdentity = true;
  for (unsigned i = 0; i < dimOrder.size(); ++i) {
    if (dimOrder[i] != i) {
      isIdentity = false;
      break;
    }
  }

  if (isIdentity)
    return dimOrder.size() == 1 ? "linear" : "identity";

  return "permuted";
}

static bool isDenseAccessPattern(StringRef accessPattern) {
  return accessPattern == "linear" || accessPattern == "identity";
}

static int32_t inferPrefetchStrategy(StringRef accessPattern,
                                     int32_t contiguousVecLen) {
  // 0 = dense/sequential access, 1 = non-contiguous or gather-like access.
  bool dense = isDenseAccessPattern(accessPattern);
  return (dense && contiguousVecLen > 1) ? 0 : 1;
}

static int64_t inferTotalElements(MemRefType memrefType) {
  if (!memrefType.hasRank() || memrefType.getRank() == 0)
    return 1;

  int64_t total = 1;
  for (int64_t dim : memrefType.getShape()) {
    if (ShapedType::isDynamic(dim))
      return 1;
    total *= std::max<int64_t>(1, dim);
  }
  return std::max<int64_t>(1, total);
}

// Classify a linalg op into the addr_gen2.py dispatch category.
static StringRef classifyLinalgOpKind(linalg::LinalgOp linalgOp) {
  // Named matmul ops → gather pattern (A×B→C with reduction over K)
  if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(linalgOp.getOperation()))
    return "gather";

  // Any reduction iterator → not handled by elementwise/broadcast
  auto iterTypes = linalgOp.getIteratorTypesArray();
  bool hasReduction = llvm::any_of(iterTypes, [](utils::IteratorType t) {
    return t == utils::IteratorType::reduction;
  });
  if (hasReduction)
    return "incomplete";

  SmallVector<AffineMap, 4> maps = linalgOp.getIndexingMapsArray();
  int inputCount = linalgOp.getNumDpsInputs();
  int broadcastInputs = 0;
  bool anyUnsupported = false;

  for (int i = 0; i < static_cast<int>(maps.size()); ++i) {
    StringRef pat = classifyAccessPattern(maps[i]);
    if (pat == "broadcast") {
      if (i < inputCount) ++broadcastInputs;
    } else if (pat != "identity" && pat != "linear") {
      anyUnsupported = true;
      break;
    }
  }
  if (anyUnsupported)
    return "incomplete";

  // Single input with broadcast map + identity output → broadcast_op
  if (inputCount == 1 && broadcastInputs == 1 &&
      linalgOp.getNumDpsInits() == 1) {
    StringRef outPat = classifyAccessPattern(maps[static_cast<size_t>(inputCount)]);
    if (outPat == "identity" || outPat == "linear")
      return "broadcast_op";
  }

  if (broadcastInputs == 0)
    return "elementwise";

  return "incomplete"; // mixed broadcast + non-broadcast (e.g. outer product)
}

static DictionaryAttr buildTensorInfoDict(Builder &b, int32_t tensorId,
                                          StringRef accessKind,
                                          StringRef role,
                                          MemRefType memrefType,
                                          AffineMap indexingMap) {
  SmallVector<Attribute, 8> shapeAttrs;
  SmallVector<Attribute, 8> strideAttrs;

  int64_t elemBytes = getElementByteWidth(memrefType.getElementType());
  for (int64_t d : memrefType.getShape())
    shapeAttrs.push_back(b.getI64IntegerAttr(d));

  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (succeeded(getStridesAndOffset(memrefType, strides, offset))) {
    for (int64_t s : strides)
      strideAttrs.push_back(b.getI64IntegerAttr(s));
  } else {
    for (int64_t i = 0; i < memrefType.getRank(); ++i)
      strideAttrs.push_back(b.getI64IntegerAttr(ShapedType::kDynamic));
  }

  int32_t contiguousVecLen = inferContiguousVectorLength(memrefType, elemBytes);
  int64_t rank = memrefType.getRank();
  StringRef accessPattern = classifyAccessPattern(indexingMap);
  int32_t issueStrategy = inferPrefetchStrategy(accessPattern, contiguousVecLen);
  int64_t issueTotalElements = inferTotalElements(memrefType);

  std::string elementTypeStr;
  {
    llvm::raw_string_ostream os(elementTypeStr);
    os << memrefType.getElementType();
  }

  return b.getDictionaryAttr({
      b.getNamedAttr("tensor_id", b.getI32IntegerAttr(tensorId)),
      b.getNamedAttr("role", b.getStringAttr(role)),
      b.getNamedAttr("access_kind", b.getStringAttr(accessKind)),
      b.getNamedAttr("element_type", b.getStringAttr(elementTypeStr)),
      b.getNamedAttr("element_bytes", b.getI64IntegerAttr(elemBytes)),
      b.getNamedAttr("rank", b.getI64IntegerAttr(rank)),
      b.getNamedAttr("shape", b.getArrayAttr(shapeAttrs)),
      b.getNamedAttr("strides", b.getArrayAttr(strideAttrs)),
      b.getNamedAttr("access_pattern", b.getStringAttr(accessPattern)),
      b.getNamedAttr("contiguous_vector_len",
                     b.getI32IntegerAttr(contiguousVecLen)),
      // Precompute all request-level fields at linalg time so affine insertion
      // can emit calls without reconstructing memref/indexing details.
      b.getNamedAttr("issue_strategy", b.getI32IntegerAttr(issueStrategy)),
      b.getNamedAttr("issue_total_elements",
                     b.getI64IntegerAttr(issueTotalElements)),
  });
}

// Compact per-operand attribute for pre_issue_APE_request calls.
static DictionaryAttr buildCompactOperandInfo(Builder &b, int32_t tensorId,
                                               StringRef accessKind,
                                               StringRef accessPattern,
                                               int64_t elementBytes) {
  return b.getDictionaryAttr({
      b.getNamedAttr("tensor_id",      b.getI32IntegerAttr(tensorId)),
      b.getNamedAttr("access_kind",    b.getStringAttr(accessKind)),
      b.getNamedAttr("access_pattern", b.getStringAttr(accessPattern)),
      b.getNamedAttr("element_bytes",  b.getI64IntegerAttr(elementBytes)),
  });
}


struct TensorMetadata {
  int32_t tensorId = -1;
  int32_t contiguousVectorLen = 1;
};

static std::optional<TensorMetadata> readTensorMetadata(Value memref,
                                                        func::FuncOp funcOp) {
  TensorMetadata md;

  auto readFromDict = [&](DictionaryAttr info) {
    if (!info)
      return;
    if (auto id = dyn_cast_or_null<IntegerAttr>(info.get("tensor_id")))
      md.tensorId = static_cast<int32_t>(id.getInt());
    if (auto vecLen =
            dyn_cast_or_null<IntegerAttr>(info.get("contiguous_vector_len")))
      md.contiguousVectorLen = static_cast<int32_t>(vecLen.getInt());
  };

  if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
    if (blockArg.getOwner() == &funcOp.getBody().front()) {
      readFromDict(
          dyn_cast_or_null<DictionaryAttr>(funcOp.getArgAttrOfType<Attribute>(
              blockArg.getArgNumber(), kAPETensorInfoAttr)));
    }
  } else if (Operation *def = memref.getDefiningOp()) {
    readFromDict(dyn_cast_or_null<DictionaryAttr>(def->getAttr(kAPETensorInfoAttr)));
  }

  if (md.tensorId < 0)
    return std::nullopt;

  return md;
}

// static SmallVector<affine::AffineForOp, 6> getEnclosingAffineLoops(Operation *op) {
//   SmallVector<affine::AffineForOp, 6> loops;
//   for (Operation *cur = op->getParentOp(); cur; cur = cur->getParentOp()) {
//     if (auto loop = dyn_cast<affine::AffineForOp>(cur))
//       loops.push_back(loop);
//   }
//   llvm::reverse(loops);
//   return loops;
// }

static Value getI64Const(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64Type(),
                                           builder.getI64IntegerAttr(value));
}


class LinalgAPEAnalysis
    : public impl::LinalgAPEAnalysisBase<LinalgAPEAnalysis> {
public:
  using impl::LinalgAPEAnalysisBase<LinalgAPEAnalysis>::LinalgAPEAnalysisBase;

  void runOnOperation() final {
    auto funcOp = dyn_cast<func::FuncOp>(getOperation());
    if (!funcOp)
      return;
    if (funcOp.isExternal() || funcOp.getBody().empty())
      return;

    ensureIssueAPERequestDeclaration(funcOp->getParentOfType<ModuleOp>());

    ensurePreIssueAPERequestDeclaration(funcOp->getParentOfType<ModuleOp>());

    Builder b(funcOp.getContext());
    OpBuilder callBuilder(funcOp.getContext());
    DenseMap<Value, int32_t> tensorIds;
    int32_t nextTensorId = 0;

    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      SmallVector<Attribute, 8> perOperandInfos;  // full info → linalg op attr
      SmallVector<Attribute, 8> compactInfos;      // compact info → call attr
      SmallVector<Value, 8> operandValues;  // Add this

      auto processOperand = [&](OpOperand *operand, StringRef role,
                                StringRef accessKind, AffineMap indexingMap) {
        Value v = operand->get();
        auto memrefTy = dyn_cast<MemRefType>(v.getType());
        if (!memrefTy)
          return;

        int32_t tensorId;
        auto it = tensorIds.find(v);
        if (it == tensorIds.end()) {
          tensorIds[v] = nextTensorId;
          tensorId = nextTensorId++;
        } else {
          tensorId = it->second;
        }

        DictionaryAttr info = buildTensorInfoDict(b, tensorId, accessKind, role,
                                                  memrefTy, indexingMap);
        perOperandInfos.push_back(info);

        // Build compact call attribute: only fields needed at issue time.
        // Shape, strides, rank are omitted — derivable from the memref arg.
        int64_t elemBytes = getElementByteWidth(memrefTy.getElementType());
        StringRef accessPattern = classifyAccessPattern(indexingMap);
        compactInfos.push_back(buildCompactOperandInfo(
            b, tensorId, accessKind, accessPattern, elemBytes));
        operandValues.push_back(v);

        // Attach compact per-tensor metadata to the value source so later
        // passes can recover a memref from tensor_id even after linalg is
        // lowered away.
        auto tensorInfo = b.getDictionaryAttr(
            {b.getNamedAttr("tensor_id", b.getI32IntegerAttr(tensorId)),
             b.getNamedAttr("contiguous_vector_len",
                  info.get("contiguous_vector_len"))});
        if (auto blockArg = dyn_cast<BlockArgument>(v)) {
          if (blockArg.getOwner() == &funcOp.getBody().front() &&
              !funcOp.getArgAttr(blockArg.getArgNumber(), kAPETensorInfoAttr)) {
            funcOp.setArgAttr(blockArg.getArgNumber(), kAPETensorInfoAttr,
                              tensorInfo);
          }
        } else if (Operation *def = v.getDefiningOp()) {
          if (!def->getAttr(kAPETensorInfoAttr))
            def->setAttr(kAPETensorInfoAttr, tensorInfo);
        }
      };

      for (auto [idx, input] : llvm::enumerate(linalgOp.getDpsInputs())) {
        AffineMap map = linalgOp.getMatchingIndexingMap(
            linalgOp.getDpsInputOperand(idx));
        processOperand(linalgOp.getDpsInputOperand(idx), "input", "read", map);
      }
      for (auto [idx, output] : llvm::enumerate(linalgOp.getDpsInits())) {
        (void)output;
        AffineMap map = linalgOp.getMatchingIndexingMap(
            linalgOp.getDpsInitOperand(idx));
        processOperand(linalgOp.getDpsInitOperand(idx), "output", "write", map);
      }

      if (!compactInfos.empty()) {
        // Attach full info to the linalg op for downstream reference.
        DictionaryAttr opInfo = b.getDictionaryAttr(
            {b.getNamedAttr("operands", b.getArrayAttr(perOperandInfos))});
        linalgOp->setAttr(kAPETensorInfoAttr, opInfo);

        StringRef opKind = classifyLinalgOpKind(linalgOp);
        DictionaryAttr callInfo = b.getDictionaryAttr({
            b.getNamedAttr("op_kind",   b.getStringAttr(opKind)),
            b.getNamedAttr("operands",  b.getArrayAttr(compactInfos))});

        callBuilder.setInsertionPoint(linalgOp);
        auto preIssue = callBuilder.create<func::CallOp>(
            linalgOp.getLoc(), TypeRange{},
            SymbolRefAttr::get(callBuilder.getContext(), kPreIssueAPERequest),
            ValueRange{});
        preIssue->setAttr("ape.pre_issue_info", callInfo);
      }
    });
  }
};

class AffineAPEInsertion
    : public impl::AffineAPEInsertionBase<AffineAPEInsertion> {
public:
  using impl::AffineAPEInsertionBase<AffineAPEInsertion>::AffineAPEInsertionBase;

  void runOnOperation() final {
    auto funcOp = dyn_cast<func::FuncOp>(getOperation());
    if (!funcOp)
      return;
    if (funcOp.isExternal() || funcOp.getBody().empty())
      return;

    ModuleOp module = funcOp->getParentOfType<ModuleOp>();
    ensureApeIncompleteDecl(module);
    ensureApeBroadcastDecl(module);
    ensureApeGatherDecl(module);
    ensureApeElementwiseDecl(module);

    MLIRContext *ctx = funcOp.getContext();
    OpBuilder builder(ctx);
    DenseMap<int32_t, Value> tensorIdToMemref;

    // Build tensor_id → live SSA memref map from annotated block args and ops.
    for (BlockArgument arg : funcOp.getArguments()) {
      if (!isa<MemRefType>(arg.getType()))
        continue;
      if (auto md = readTensorMetadata(arg, funcOp))
        tensorIdToMemref.try_emplace(md->tensorId, arg);
    }
    funcOp.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        if (!isa<MemRefType>(result.getType()))
          continue;
        if (auto md = readTensorMetadata(result, funcOp))
          tensorIdToMemref.try_emplace(md->tensorId, result);
      }
    });

    // Extract (aligned_i64, offset_i64, s0_i64, s1_i64, str0_i64, str1_i64)
    // from a live memref SSA value. `firstDim` lets us skip batch dims on rank-3.
    auto extractMemrefArgs =
        [&](OpBuilder &b, Location loc, Value memrefVal,
            int firstDim) -> SmallVector<Value, 6> {
      auto ty = cast<MemRefType>(memrefVal.getType());
      auto shape = ty.getShape();
      int rank = static_cast<int>(shape.size());

      SmallVector<int64_t, 4> strides;
      int64_t offset = 0;
      if (failed(getStridesAndOffset(ty, strides, offset))) {
        offset = 0;
        strides.assign(static_cast<size_t>(rank), 0);
      }

      Value alignedIdx =
          b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memrefVal);
      Value aligned_i64 =
          b.create<arith::IndexCastOp>(loc, b.getI64Type(), alignedIdx);

      int d0 = firstDim, d1 = firstDim + 1;
      auto safeShape = [&](int d) -> int64_t {
        if (d < 0 || d >= rank) return 1;
        return ShapedType::isDynamic(shape[d]) ? 1 : shape[d];
      };
      auto safeStride = [&](int d) -> int64_t {
        if (d < 0 || d >= static_cast<int>(strides.size())) return 0;
        return ShapedType::isDynamic(strides[d]) ? 0 : strides[d];
      };

      return {aligned_i64,
              getI64Const(b, loc, offset),
              getI64Const(b, loc, safeShape(d0)),
              getI64Const(b, loc, safeShape(d1)),
              getI64Const(b, loc, safeStride(d0)),
              getI64Const(b, loc, safeStride(d1))};
    };

    funcOp.walk([&](func::CallOp preIssueCall) {
      if (preIssueCall.getCallee() != kPreIssueAPERequest)
        return;

      auto info = dyn_cast_or_null<DictionaryAttr>(
          preIssueCall->getAttr("ape.pre_issue_info"));
      if (!info)
        return;

      auto opKindAttr = dyn_cast_or_null<StringAttr>(info.get("op_kind"));
      StringRef opKind = opKindAttr ? opKindAttr.getValue() : "incomplete";

      auto operandsAttr = dyn_cast_or_null<ArrayAttr>(info.get("operands"));
      if (!operandsAttr)
        return;

      // Collect element_bytes from the first operand (all same element type).
      int64_t elemBytes = 4;
      if (!operandsAttr.empty())
        if (auto first = dyn_cast<DictionaryAttr>(operandsAttr[0]))
          if (auto eb = dyn_cast_or_null<IntegerAttr>(first.get("element_bytes")))
            elemBytes = std::max<int64_t>(1, eb.getInt());

      // Collect live memref SSA values in operand order (inputs then outputs).
      SmallVector<Value, 4> memrefs;
      bool missingMemref = false;
      for (Attribute opAttr : operandsAttr) {
        auto opDict = dyn_cast<DictionaryAttr>(opAttr);
        if (!opDict) { missingMemref = true; break; }
        auto tidAttr = dyn_cast_or_null<IntegerAttr>(opDict.get("tensor_id"));
        if (!tidAttr) { missingMemref = true; break; }
        auto it = tensorIdToMemref.find(static_cast<int32_t>(tidAttr.getInt()));
        if (it == tensorIdToMemref.end()) { missingMemref = true; break; }
        memrefs.push_back(it->second);
      }

      // TODO: KIM change call insertion level — currently inserted right before
      // pre_issue_APE_request (outermost scope, not inside any affine loop).
      Location loc = preIssueCall.getLoc();
      builder.setInsertionPoint(preIssueCall);

      auto emitIncomplete = [&]() {
        builder.create<func::CallOp>(loc, TypeRange{},
            SymbolRefAttr::get(ctx, kApeIncomplete), ValueRange{});
      };

      if (missingMemref || opKind == "incomplete") {
        emitIncomplete();
        return;
      }

      // --- gather: matmul / batch_matmul --- operands: [A, B, C]
      if (opKind == "gather") {
        if (memrefs.size() < 3) { emitIncomplete(); return; }
        SmallVector<Value> callArgs;
        for (int oi = 0; oi < 3; ++oi) {
          Value mv = memrefs[static_cast<size_t>(oi)];
          int r = cast<MemRefType>(mv.getType()).getRank();
          auto args = extractMemrefArgs(builder, loc, mv, r == 3 ? 1 : 0);
          callArgs.append(args.begin(), args.end());
        }
        callArgs.push_back(getI64Const(builder, loc, CACHE_CAPACITY)); // TODO: KIM pass L1 cache size
        callArgs.push_back(getI64Const(builder, loc, elemBytes));
        builder.create<func::CallOp>(loc, TypeRange{},
            SymbolRefAttr::get(ctx, kApeGather), callArgs);
        return;
      }

      // --- broadcast_op: single-input broadcast scan ---
      if (opKind == "broadcast_op") {
        if (memrefs.empty()) { emitIncomplete(); return; }
        Value mv = memrefs[0];
        int r = cast<MemRefType>(mv.getType()).getRank();
        auto args = extractMemrefArgs(builder, loc, mv, r == 3 ? 1 : 0);
        args.push_back(getI64Const(builder, loc, elemBytes));
        builder.create<func::CallOp>(loc, TypeRange{},
            SymbolRefAttr::get(ctx, kApeBroadcast), args);
        return;
      }

      // --- elementwise: all-identity/linear ops ---
      if (opKind == "elementwise") {
        int64_t nOps = static_cast<int64_t>(memrefs.size());
        constexpr int64_t kFields = 6; // fields per operand descriptor
        int64_t bufSize = std::max<int64_t>(1, nOps * kFields);
        auto bufType = MemRefType::get({bufSize}, builder.getI64Type());
        Value buf = builder.create<memref::AllocaOp>(loc, bufType);

        for (int64_t oi = 0; oi < nOps; ++oi) {
          Value mv = memrefs[static_cast<size_t>(oi)];
          int r = cast<MemRefType>(mv.getType()).getRank();
          auto fields = extractMemrefArgs(builder, loc, mv, r == 3 ? 1 : 0);
          for (int64_t fi = 0; fi < kFields; ++fi) {
            Value idx = builder.create<arith::ConstantIndexOp>(
                loc, oi * kFields + fi);
            builder.create<memref::StoreOp>(
                loc, fields[static_cast<size_t>(fi)], buf, ValueRange{idx});
          }
        }

        Value descPtr =
            builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buf);
        builder.create<func::CallOp>(loc, TypeRange{},
            SymbolRefAttr::get(ctx, kApeElementwise),
            ValueRange{getI64Const(builder, loc, nOps),
                       getI64Const(builder, loc, CACHE_CAPACITY), // TODO: KIM pass L1 cache size
                       getI64Const(builder, loc, elemBytes),
                       descPtr});
        return;
      }

      emitIncomplete();
    });
  }
};

} // namespace

void populateAPE(OpPassManager &pm) {
  pm.addPass(createLinalgAPEAnalysis());
  pm.addPass(createAffineAPEInsertion());
}

struct AddressGeneratorKey {
  MemRefType memrefType;
  AffineMap indexingMap;

  bool operator==(const AddressGeneratorKey &other) const {
    return memrefType == other.memrefType &&
           indexingMap == other.indexingMap;
  }
};

struct GenAddrFunctionPass
    : public impl::GenAddrFunctionPassBase<GenAddrFunctionPass> {
  using impl::GenAddrFunctionPassBase<GenAddrFunctionPass>::GenAddrFunctionPassBase;

  struct OperandAccessPattern {
    MemRefType memrefType;
    AffineMap indexingMap;
    bool isOutput;
  };

  struct LinalgTraceSite {
    linalg::LinalgOp linalgOp;
    SmallVector<OperandAccessPattern> operandPatterns;
  };

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    Location loc = module.getLoc();

    OpBuilder moduleBuilder(context);
    moduleBuilder.setInsertionPointToEnd(module.getBody());

    auto i64Type = IntegerType::get(context, 64);

    //
    // Declare the runtime formatting function:
    //
    //   func.func @format_address_pair(i64, i64) -> i64
    //
    func::FuncOp formatAddressPair =
      module.lookupSymbol<func::FuncOp>("format_address_pair");

    if (!formatAddressPair) {
      auto formatType =
        moduleBuilder.getFunctionType({i64Type, i64Type}, {i64Type});

      formatAddressPair =
          moduleBuilder.create<func::FuncOp>(
              loc,
          "format_address_pair",
              formatType);

      // This is an external declaration. Keep it private so that the
      // declaration is valid at module scope.
      formatAddressPair.setPrivate();
    }

    //
    // Each operand may have a different indexing map, even when the
    // operands have the same memref type.
    //
    SmallVector<OperandAccessPattern> accessPatterns;
    SmallVector<LinalgTraceSite> traceSites;
    std::unordered_set<std::string> seenPatterns;

    module.walk([&](linalg::LinalgOp linalgOp) {
      SmallVector<OperandAccessPattern> operandPatterns;
      SmallVector<AffineMap, 4> indexingMaps = linalgOp.getIndexingMapsArray();

      if (indexingMaps.size() != linalgOp->getNumOperands()) {
        linalgOp.emitError()
            << "number of indexing maps does not match number of operands";
        signalPassFailure();
        return;
      }

      for (auto it : llvm::enumerate(linalgOp->getOperands())) {
        Value operand = it.value();

        auto memrefType =
            mlir::dyn_cast<MemRefType>(operand.getType());

        if (!memrefType)
          continue;

        AffineMap indexingMap = indexingMaps[it.index()];

        //
        // affine.apply can evaluate dimension expressions directly.
        // Symbols require additional runtime values, which are not part
        // of this helper's current ABI.
        //
        if (indexingMap.getNumSymbols() != 0) {
          linalgOp.emitError()
              << "symbolic indexing maps are not currently supported";
          signalPassFailure();
          return;
        }

        //
        // Linalg indexing maps use the same number of loop dimensions as
        // the enclosing op iteration space.
        //
        if (indexingMap.getNumDims() != linalgOp.getNumLoops()) {
          linalgOp.emitError()
              << "indexing map dimension count does not match "
                 "linalg iteration rank";
          signalPassFailure();
          return;
        }

        if (indexingMap.getNumResults() != memrefType.getRank()) {
          linalgOp.emitError()
              << "indexing map result rank does not match memref rank";
          signalPassFailure();
          return;
        }

        bool isOutput =
          static_cast<int64_t>(it.index()) >= linalgOp.getNumDpsInputs();

        operandPatterns.push_back({memrefType, indexingMap, isOutput});

        std::string key;
        llvm::raw_string_ostream keyStream(key);

        memrefType.print(keyStream);
        keyStream << "|";
        indexingMap.print(keyStream);
        keyStream << "|loops=" << linalgOp.getNumLoops();
        keyStream.flush();

        if (seenPatterns.insert(key).second) {
          accessPatterns.push_back({memrefType, indexingMap, isOutput});
        }
      }

      if (!operandPatterns.empty())
        traceSites.push_back({linalgOp, std::move(operandPatterns)});
    });

    //
    // Generate one helper for every distinct:
    //
    //   memref type + indexing map
    //
    for (const OperandAccessPattern &pattern : accessPatterns) {
      createAddressFunction(
          module,
          pattern.memrefType,
          pattern.indexingMap,
          formatAddressPair);
    }

    for (auto &traceSite : traceSites) {
      func::FuncOp traceFunction = createTraceFunction(
          module,
          traceSite.linalgOp,
          traceSite.operandPatterns,
          traceSite.linalgOp->getLoc(),
          formatAddressPair,
          &traceSite - traceSites.data());

      if (!traceFunction) {
        signalPassFailure();
        return;
      }

      SmallVector<Value> traceOperands;
      for (Value operand : traceSite.linalgOp->getOperands()) {
        if (isa<MemRefType>(operand.getType()))
          traceOperands.push_back(operand);
      }

      OpBuilder builder(traceSite.linalgOp);
      builder.create<func::CallOp>(
          traceSite.linalgOp.getLoc(),
          TypeRange{},
          SymbolRefAttr::get(context, traceFunction.getSymName()),
          traceOperands);
    }
  }

private:
  static std::string getFunctionName(
      MemRefType memrefType,
      AffineMap indexingMap) {
    std::string typeString;
    llvm::raw_string_ostream stream(typeString);

    memrefType.print(stream);
    stream << "|";
    indexingMap.print(stream);
    stream.flush();

    std::size_t hashValue =
        std::hash<std::string>{}(typeString);

    return "gen_addr_" + llvm::utohexstr(hashValue);
  }

  static std::string getTraceFunctionName(
      ArrayRef<OperandAccessPattern> operandPatterns,
      unsigned ordinal) {
    std::string key;
    llvm::raw_string_ostream stream(key);

    stream << "trace|op=" << ordinal;
    for (const OperandAccessPattern &pattern : operandPatterns) {
      stream << "|";
      pattern.memrefType.print(stream);
      stream << "|";
      pattern.indexingMap.print(stream);
    }
    stream.flush();

    std::size_t hashValue =
        std::hash<std::string>{}(key);

    return "gen_trace_" + llvm::utohexstr(hashValue);
  }

  static FailureOr<int64_t> getElementSizeInBytes(
      Type elementType) {
    int64_t bitWidth = 0;

    if (auto integerType =
            mlir::dyn_cast<IntegerType>(elementType)) {
      bitWidth = integerType.getWidth();
    } else if (auto floatType =
                   mlir::dyn_cast<FloatType>(elementType)) {
      bitWidth = floatType.getWidth();
    } else if (elementType.isIndex()) {
      //
      // This example assumes 64-bit indexes.
      //
      bitWidth = 64;
    } else if (auto vectorType =
                   mlir::dyn_cast<VectorType>(elementType)) {
      auto elementSize =
          getElementSizeInBytes(
              vectorType.getElementType());

      if (failed(elementSize))
        return failure();

      return *elementSize * vectorType.getNumElements();
    } else {
      return failure();
    }

    if (bitWidth <= 0 || bitWidth % 8 != 0)
      return failure();

    return bitWidth / 8;
  }

  static SmallVector<Value> buildIterationSizes(
      OpBuilder &builder,
      Location loc,
      AffineMap indexingMap,
      memref::ExtractStridedMetadataOp metadata) {
    unsigned iterationRank = indexingMap.getNumDims();

    Value one =
        builder.create<arith::ConstantIndexOp>(
            loc,
            1);

    SmallVector<Value> iterationSizes(
        iterationRank,
        one);

    auto memrefSizes = metadata.getSizes();
    for (unsigned resultIndex = 0;
         resultIndex < indexingMap.getNumResults();
         ++resultIndex) {
      auto dimExpr =
          mlir::dyn_cast<AffineDimExpr>(
              indexingMap.getResult(resultIndex));

      if (!dimExpr)
        continue;

      unsigned dimPos = dimExpr.getPosition();
      if (dimPos >= iterationRank)
        continue;

      iterationSizes[dimPos] = memrefSizes[resultIndex];
    }

    return iterationSizes;
  }

  static FailureOr<SmallVector<Value>> buildLinalgDimensionSizes(
      OpBuilder &builder,
      Location loc,
      ArrayRef<OperandAccessPattern> operandPatterns,
      Block *entry) {
    if (operandPatterns.empty())
      return SmallVector<Value>{};

    unsigned iterationRank = operandPatterns.front().indexingMap.getNumDims();
    SmallVector<Value> dimensionSizes(iterationRank);
    SmallVector<bool> seenDimensions(iterationRank, false);

    for (auto it : llvm::enumerate(operandPatterns)) {
      auto metadata = builder.create<memref::ExtractStridedMetadataOp>(
          loc,
          entry->getArgument(it.index()));

      auto memrefSizes = metadata.getSizes();
      AffineMap indexingMap = it.value().indexingMap;

      for (unsigned resultIndex = 0;
           resultIndex < indexingMap.getNumResults();
           ++resultIndex) {
        auto dimExpr =
            mlir::dyn_cast<AffineDimExpr>(indexingMap.getResult(resultIndex));

        if (!dimExpr)
          continue;

        unsigned dimPos = dimExpr.getPosition();
        if (dimPos >= iterationRank || seenDimensions[dimPos])
          continue;

        dimensionSizes[dimPos] = memrefSizes[resultIndex];
        seenDimensions[dimPos] = true;
      }
    }

    for (bool seen : seenDimensions) {
      if (!seen)
        return failure();
    }

    return dimensionSizes;
  }

  static SmallVector<unsigned> collectLoopDimsByType(
      ArrayRef<utils::IteratorType> iteratorTypes,
      bool collectReductionDims) {
    SmallVector<unsigned> dimensions;

    for (auto it : llvm::enumerate(iteratorTypes)) {
      bool isReduction =
          it.value() == utils::IteratorType::reduction;
      if (isReduction == collectReductionDims)
        dimensions.push_back(it.index());
    }

    return dimensions;
  }

  static SmallVector<Value> decodeLinearIndex(
      OpBuilder &builder,
      Location loc,
      Value linearIndex,
      ArrayRef<Value> extents) {
    SmallVector<Value> coordinates(extents.size());
    Value remainingIndex = linearIndex;

    for (int64_t dimension = static_cast<int64_t>(extents.size()) - 1;
         dimension >= 0;
         --dimension) {
      if (dimension == 0) {
        coordinates[dimension] = remainingIndex;
        continue;
      }

      coordinates[dimension] = builder.create<arith::RemUIOp>(
          loc,
          remainingIndex,
          extents[dimension]);

      remainingIndex = builder.create<arith::DivUIOp>(
          loc,
          remainingIndex,
          extents[dimension]);
    }

    return coordinates;
  }

  static Value buildOperandFlatIndexFromLinalgCoords(
      OpBuilder &builder,
      Location loc,
      const OperandAccessPattern &pattern,
      ArrayRef<Value> linalgCoords,
      ArrayRef<Value> linalgDimensionSizes) {
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    unsigned iterationRank = pattern.indexingMap.getNumDims();

    SmallVector<bool> usesDimension(iterationRank, false);
    for (unsigned resultIndex = 0;
         resultIndex < pattern.indexingMap.getNumResults();
         ++resultIndex) {
      auto dimExpr =
          mlir::dyn_cast<AffineDimExpr>(pattern.indexingMap.getResult(resultIndex));
      if (!dimExpr)
        continue;

      unsigned dimPos = dimExpr.getPosition();
      if (dimPos < iterationRank)
        usesDimension[dimPos] = true;
    }

    Value flatIndex = zero;
    for (unsigned dimension = 0; dimension < iterationRank; ++dimension) {
      Value size = usesDimension[dimension] ? linalgDimensionSizes[dimension] : one;
      Value coordinate = usesDimension[dimension] ? linalgCoords[dimension] : zero;

      flatIndex = builder.create<arith::MulIOp>(loc, flatIndex, size);
      flatIndex = builder.create<arith::AddIOp>(loc, flatIndex, coordinate);
    }

    return flatIndex;
  }

  static Value buildIterationCount(
      OpBuilder &builder,
      Location loc,
      ArrayRef<Value> iterationSizes) {
    Value totalIterations =
        builder.create<arith::ConstantIndexOp>(
            loc,
            1);

    for (Value size : iterationSizes) {
      totalIterations =
          builder.create<arith::MulIOp>(
              loc,
              totalIterations,
              size);
    }

    return totalIterations;
  }

  static func::FuncOp createTraceFunction(
      ModuleOp module,
      linalg::LinalgOp linalgOp,
      ArrayRef<OperandAccessPattern> operandPatterns,
      Location loc,
      func::FuncOp formatAddressPair,
      unsigned ordinal) {
    MLIRContext *context = module.getContext();

    OpBuilder moduleBuilder(context);
    moduleBuilder.setInsertionPointToEnd(module.getBody());

    std::string functionName =
        getTraceFunctionName(operandPatterns, ordinal);

    if (auto existing =
            module.lookupSymbol<func::FuncOp>(functionName)) {
      return existing;
    }

    SmallVector<Type> inputTypes;
    for (const OperandAccessPattern &pattern : operandPatterns)
      inputTypes.push_back(pattern.memrefType);

    auto functionType =
        moduleBuilder.getFunctionType(
            inputTypes,
            TypeRange{});

    func::FuncOp function =
        moduleBuilder.create<func::FuncOp>(
            loc,
            functionName,
            functionType);

    function.setPrivate();

    Block *entry = function.addEntryBlock();
    OpBuilder builder(entry, entry->begin());

    Value zero =
        builder.create<arith::ConstantIndexOp>(
            loc,
            0);
    Value one =
        builder.create<arith::ConstantIndexOp>(
            loc,
            1);
    auto i64Type =
        IntegerType::get(context, 64);

    SmallVector<func::FuncOp> addressFunctions;
    SmallVector<unsigned> inputOperandIndices;
    SmallVector<unsigned> outputOperandIndices;

    for (auto it : llvm::enumerate(operandPatterns)) {
      const OperandAccessPattern &pattern = it.value();

      func::FuncOp addressFunction = createAddressFunction(
          module,
          pattern.memrefType,
          pattern.indexingMap,
          formatAddressPair);

      if (!addressFunction) {
        function.erase();
        return func::FuncOp();
      }

      addressFunctions.push_back(addressFunction);

      if (pattern.isOutput)
        outputOperandIndices.push_back(it.index());
      else
        inputOperandIndices.push_back(it.index());
    }

    FailureOr<SmallVector<Value>> linalgDimensionSizes = buildLinalgDimensionSizes(
        builder,
        loc,
        operandPatterns,
        entry);

    if (failed(linalgDimensionSizes)) {
      function.emitError()
          << "cannot infer all linalg loop dimension sizes for trace generation";
      function.erase();
      return func::FuncOp();
    }

    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    SmallVector<unsigned> reductionDims =
        collectLoopDimsByType(iteratorTypes, true);
    SmallVector<unsigned> parallelDims =
        collectLoopDimsByType(iteratorTypes, false);

    SmallVector<Value> parallelDimSizes;
    parallelDimSizes.reserve(parallelDims.size());
    for (unsigned dim : parallelDims)
      parallelDimSizes.push_back((*linalgDimensionSizes)[dim]);

    SmallVector<Value> reductionDimSizes;
    reductionDimSizes.reserve(reductionDims.size());
    for (unsigned dim : reductionDims)
      reductionDimSizes.push_back((*linalgDimensionSizes)[dim]);

    Value groupCount = buildIterationCount(
        builder,
        loc,
        parallelDimSizes);
    Value reductionVolume = buildIterationCount(
        builder,
        loc,
        reductionDimSizes);

    Value inputCount = builder.create<arith::ConstantIndexOp>(
        loc,
        static_cast<int64_t>(inputOperandIndices.size()));
    Value outputCount = builder.create<arith::ConstantIndexOp>(
        loc,
        static_cast<int64_t>(outputOperandIndices.size()));
    Value inputPhaseLength = builder.create<arith::MulIOp>(
        loc,
        reductionVolume,
        inputCount);
    Value groupSpan = builder.create<arith::AddIOp>(
        loc,
        inputPhaseLength,
        outputCount);
    Value totalTraceSteps = builder.create<arith::MulIOp>(
        loc,
        groupCount,
        groupSpan);

    auto forOp =
        builder.create<scf::ForOp>(
            loc,
            zero,
            totalTraceSteps,
            one);

    OpBuilder loopBuilder = OpBuilder::atBlockBegin(forOp.getBody());
    Value groupId = loopBuilder.create<arith::DivUIOp>(
        loc,
        forOp.getInductionVar(),
        groupSpan);
    Value phase = loopBuilder.create<arith::RemUIOp>(
        loc,
        forOp.getInductionVar(),
        groupSpan);

    SmallVector<Value> parallelCoords = decodeLinearIndex(
        loopBuilder,
        loc,
        groupId,
        parallelDimSizes);

    SmallVector<Value> baseLinalgCoords((*linalgDimensionSizes).size(), zero);
    for (auto it : llvm::enumerate(parallelDims))
      baseLinalgCoords[it.value()] = parallelCoords[it.index()];

    Value inInputPhase = loopBuilder.create<arith::CmpIOp>(
        loc,
        arith::CmpIPredicate::ult,
        phase,
        inputPhaseLength);

    auto inputPhaseIf = loopBuilder.create<scf::IfOp>(
        loc,
        inInputPhase,
        true);

    {
      OpBuilder thenBuilder = inputPhaseIf.getThenBodyBuilder();
      Value inputSlot = thenBuilder.create<arith::RemUIOp>(
          loc,
          phase,
          inputCount);
      Value reductionLinear = thenBuilder.create<arith::DivUIOp>(
          loc,
          phase,
          inputCount);

      SmallVector<Value> reductionCoords = decodeLinearIndex(
          thenBuilder,
          loc,
          reductionLinear,
          reductionDimSizes);

      SmallVector<Value> linalgCoords = baseLinalgCoords;
      for (auto it : llvm::enumerate(reductionDims))
        linalgCoords[it.value()] = reductionCoords[it.index()];

      for (auto it : llvm::enumerate(inputOperandIndices)) {
        Value operandIndex = thenBuilder.create<arith::ConstantIndexOp>(
            loc,
            static_cast<int64_t>(it.index()));
        Value matchesOperand = thenBuilder.create<arith::CmpIOp>(
            loc,
            arith::CmpIPredicate::eq,
            inputSlot,
            operandIndex);
        auto operandIf = thenBuilder.create<scf::IfOp>(
            loc,
            matchesOperand,
            false);
        OpBuilder operandBuilder = operandIf.getThenBodyBuilder();
        unsigned operandPos = it.value();
        Value flatIndex = buildOperandFlatIndexFromLinalgCoords(
            operandBuilder,
            loc,
            operandPatterns[operandPos],
            linalgCoords,
            *linalgDimensionSizes);
        Value flatIndexI64 = operandBuilder.create<arith::IndexCastOp>(
            loc,
            i64Type,
            flatIndex);
        operandBuilder.create<func::CallOp>(
            loc,
            TypeRange{i64Type},
            SymbolRefAttr::get(context, addressFunctions[operandPos].getSymName()),
            ValueRange{flatIndexI64, entry->getArgument(operandPos)});
      }

    }

    {
      OpBuilder elseBuilder = inputPhaseIf.getElseBodyBuilder();
      Value outputSlot = elseBuilder.create<arith::SubIOp>(
          loc,
          phase,
          inputPhaseLength);

      for (auto it : llvm::enumerate(outputOperandIndices)) {
        Value operandIndex = elseBuilder.create<arith::ConstantIndexOp>(
            loc,
            static_cast<int64_t>(it.index()));
        Value matchesOperand = elseBuilder.create<arith::CmpIOp>(
            loc,
            arith::CmpIPredicate::eq,
            outputSlot,
            operandIndex);
        auto operandIf = elseBuilder.create<scf::IfOp>(
            loc,
            matchesOperand,
            false);
        OpBuilder operandBuilder = operandIf.getThenBodyBuilder();
        unsigned operandPos = it.value();
        Value flatIndex = buildOperandFlatIndexFromLinalgCoords(
            operandBuilder,
            loc,
            operandPatterns[operandPos],
            baseLinalgCoords,
            *linalgDimensionSizes);
        Value flatIndexI64 = operandBuilder.create<arith::IndexCastOp>(
            loc,
            i64Type,
            flatIndex);
        operandBuilder.create<func::CallOp>(
            loc,
            TypeRange{i64Type},
            SymbolRefAttr::get(context, addressFunctions[operandPos].getSymName()),
            ValueRange{flatIndexI64, entry->getArgument(operandPos)});
      }

    }

    builder.create<func::ReturnOp>(
        loc,
        ValueRange{});

    return function;
  }

  static func::FuncOp createAddressFunction(
      ModuleOp module,
      MemRefType memrefType,
      AffineMap indexingMap,
      func::FuncOp formatAddressPair) {
    MLIRContext *context = module.getContext();
    Location loc = module.getLoc();

    OpBuilder moduleBuilder(context);
    moduleBuilder.setInsertionPointToEnd(module.getBody());

    std::string functionName =
        getFunctionName(memrefType, indexingMap);

    if (auto existing =
            module.lookupSymbol<func::FuncOp>(functionName)) {
      return existing;
    }

    auto i64Type =
        IntegerType::get(context, 64);
    Type indexType =
        IndexType::get(context);
    unsigned iterationRank =
      indexingMap.getNumDims();

    //
    // Function ABI:
    //
    //   (
    //     i64 flatIndex,
    //     memref operand
    //   ) -> i64
    //
    SmallVector<Type> inputTypes;
    inputTypes.push_back(i64Type);
    inputTypes.push_back(memrefType);

    auto functionType =
        moduleBuilder.getFunctionType(
            inputTypes,
            {i64Type});

    func::FuncOp function =
        moduleBuilder.create<func::FuncOp>(
            loc,
            functionName,
            functionType);

    function.setPrivate();

    Block *entry = function.addEntryBlock();
    OpBuilder builder(entry, entry->begin());

    Value flatIndexI64 =
        entry->getArgument(0);

    Value memref =
        entry->getArgument(1);

    // The iteration-size operands were removed from the helper ABI.
    // Reconstruct loop-dimension sizes from memref runtime sizes by mapping
    // each memref result dimension back to the corresponding affine dim when
    // the indexing map result is a plain AffineDimExpr.
    auto metadata =
      builder.create<memref::ExtractStridedMetadataOp>(
        loc,
        memref);

    SmallVector<Value> iterationSizes = buildIterationSizes(
        builder,
        loc,
        indexingMap,
        metadata);

    //
    // Convert the flat index into an MLIR index.
    //
    Value remainingIndex =
        builder.create<arith::IndexCastOp>(
            loc,
            indexType,
            flatIndexI64);

    //
    // Reconstruct coordinates in the linalg iteration space.
    //
    // For a 2D iteration space with sizes [S0, S1]:
    //
    //   d0 = flatIndex / S1
    //   d1 = flatIndex % S1
    //
    // The general row-major reconstruction is performed from the
    // innermost dimension toward the outermost dimension.
    //
    SmallVector<Value> iterationCoords(iterationRank);

    for (int64_t dimension =
             static_cast<int64_t>(iterationRank) - 1;
         dimension >= 0;
         --dimension) {
      if (dimension == 0) {
        iterationCoords[dimension] =
            remainingIndex;
        continue;
      }

      Value size =
          iterationSizes[dimension];

      iterationCoords[dimension] =
          builder.create<arith::RemUIOp>(
              loc,
              remainingIndex,
              size);

      remainingIndex =
          builder.create<arith::DivUIOp>(
              loc,
              remainingIndex,
              size);
    }

    //
    // Apply the linalg operand indexing map.
    //
    // For example:
    //
    //   affine_map<(d0, d1) -> (d1, d0)>
    //
    // generates:
    //
    //   %m0 = affine.apply #map(%d0, %d1)
    //   %m1 = affine.apply #map(%d0, %d1)
    //
    // where the individual one-result maps contain d1 and d0.
    //
    SmallVector<Value> memrefCoords;

    for (unsigned resultIndex = 0;
      // todo: get similar affine map from the memrefs - is possible?
         resultIndex < indexingMap.getNumResults();
         ++resultIndex) {
      SmallVector<AffineExpr> resultExprs;
      resultExprs.push_back(
          indexingMap.getResult(resultIndex));

      AffineMap singleResultMap =
          AffineMap::get(
              indexingMap.getNumDims(),
              indexingMap.getNumSymbols(),
              resultExprs,
              context);

      Value coordinate =
          builder.create<affine::AffineApplyOp>(
              loc,
              indexType,
              singleResultMap,
              iterationCoords)
              .getResult();

      memrefCoords.push_back(coordinate);
    }

    //
    // Extract memref metadata:
    //
    //   offset
    //   strides
    //
    Value elementOffset =
        metadata.getOffset();

    SmallVector<Value> strides(
        metadata.getStrides().begin(),
        metadata.getStrides().end());

    //
    // Compute:
    //
    //   offset + sum(memrefCoord[d] * stride[d])
    //
    for (unsigned dimension = 0;
         dimension < memrefCoords.size();
         ++dimension) {
      Value scaledCoordinate =
          builder.create<arith::MulIOp>(
              loc,
              memrefCoords[dimension],
              strides[dimension]);

      elementOffset =
          builder.create<arith::AddIOp>(
              loc,
              elementOffset,
              scaledCoordinate);
    }

    //
    // Extract the aligned base pointer as an index.
    //
    Value basePointerAsIndex =
        builder.create<
            memref::ExtractAlignedPointerAsIndexOp>(
            loc,
            indexType,
            memref);

    auto elementSize =
        getElementSizeInBytes(
            memrefType.getElementType());

    if (failed(elementSize)) {
      function.emitError()
          << "cannot determine fixed byte size for memref element type "
          << memrefType.getElementType();

      function.erase();
      return func::FuncOp();
    }

    Value elementSizeConstant =
        builder.create<arith::ConstantIndexOp>(
            loc,
            *elementSize);

    Value byteOffset =
        builder.create<arith::MulIOp>(
            loc,
            elementOffset,
            elementSizeConstant);

    Value addressAsIndex =
        builder.create<arith::AddIOp>(
            loc,
            basePointerAsIndex,
            byteOffset);

    Value addressAsI64 =
        builder.create<arith::IndexCastOp>(
            loc,
            i64Type,
            addressAsIndex);

    Value byteOffsetI64 =
      builder.create<arith::IndexCastOp>(
        loc,
        i64Type,
        byteOffset);

    Value formattedAddress =
        builder.create<func::CallOp>(
            loc,
            TypeRange{i64Type},
            SymbolRefAttr::get(
                context,
          formatAddressPair.getSymName()),
        ValueRange{addressAsI64, byteOffsetI64})
            .getResult(0);

    builder.create<func::ReturnOp>(
        loc,
        ValueRange{formattedAddress});

    return function;
  }
};

} // namespace



// // Address gen(base, i)

// // 2 x 3
// // 0-5
// // 0 = [0][0]
// // 1 = [0][1]
// // 2 = [0][2]
// // 3 = [1][0]
// // 4 = [1][1]
// // 5 = [1][2]


// // need to test:: does this work for traversal? with different rowmajor/colummajor

// // **a b matmul, where b is row major but being accessed as column major: can this be handeled??
// // **needs to vary depending on permutation and access patterns (not currently handled) 
// // **linalg loop layer, kernel module for weaving mod val, shapes for i evolving, 

// // ape iter always increases by 1, always want the next address
// // // for every instance of kernel execution, creates a new iterator
// // interweave func()
// //   for ape_iter in range
// //   0 1 2 3 4 5 6 7 8 9 
// //   (i,j,k) -> (i,k) (matrix a) 
// //   (i,j,k) -> (k,j) (matrix b) 
// //   (i,j,k) -> (i,j) (matrix c) 
// //     small_i = ape_iter mod 
// //     A = addr_gen(a, small i)
// //     B = addr...
