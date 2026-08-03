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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include "sodap/SODAPPasses.h"

using namespace mlir;

namespace mlir::sodap {
#define GEN_PASS_DEF_LINALGAPEANALYSIS
#define GEN_PASS_DEF_AFFINEAPEINSERTION
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

static std::optional<int64_t> getConstantTripCount(affine::AffineForOp loop) {
  if (!loop.hasConstantLowerBound() || !loop.hasConstantUpperBound())
    return std::nullopt;
  int64_t lb = loop.getConstantLowerBound();
  int64_t ub = loop.getConstantUpperBound();
  int64_t step = loop.getStepAsInt();
  if (step <= 0 || ub <= lb)
    return int64_t{0};
  return (ub - lb + step - 1) / step;
}

static bool operandUsesIV(AffineMap map, ValueRange mapOperands, Value iv,
                          bool onlyLastResult) {
  SmallVector<unsigned, 4> ivDimPositions;
  unsigned maxDims = std::min<unsigned>(map.getNumDims(), mapOperands.size());
  for (unsigned i = 0; i < maxDims; ++i) {
    if (mapOperands[i] == iv)
      ivDimPositions.push_back(i);
  }
  if (ivDimPositions.empty())
    return false;

  auto usesAnyDim = [&](AffineExpr e) {
    for (unsigned dimPos : ivDimPositions) {
      if (e.isFunctionOfDim(dimPos))
        return true;
    }
    return false;
  };

  if (!onlyLastResult) {
    for (AffineExpr e : map.getResults()) {
      if (usesAnyDim(e))
        return true;
    }
    return false;
  }

  if (map.getNumResults() == 0)
    return false;

  bool inLast = usesAnyDim(map.getResult(map.getNumResults() - 1));
  if (!inLast)
    return false;

  for (unsigned r = 0; r + 1 < map.getNumResults(); ++r) {
    if (usesAnyDim(map.getResult(r)))
      return false;
  }
  return true;
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

static SmallVector<affine::AffineForOp, 6> getEnclosingAffineLoops(Operation *op) {
  SmallVector<affine::AffineForOp, 6> loops;
  for (Operation *cur = op->getParentOp(); cur; cur = cur->getParentOp()) {
    if (auto loop = dyn_cast<affine::AffineForOp>(cur))
      loops.push_back(loop);
  }
  llvm::reverse(loops);
  return loops;
}

static Value getZeroIndex(OpBuilder &builder, Location loc) {
  return builder.create<arith::ConstantIndexOp>(loc, 0);
}

static Value getI32Const(OpBuilder &builder, Location loc, int32_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI32Type(),
                                           builder.getI32IntegerAttr(value));
}

static Value getI64Const(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64Type(),
                                           builder.getI64IntegerAttr(value));
}

static Value getI1Const(OpBuilder &builder, Location loc, bool value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI1Type(),
                                           builder.getBoolAttr(value));
}

static Value getBaseAddrAsIndex(OpBuilder &builder, Location loc, Value memref) {
  return builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
}

static Value materializeI64DescriptorPtr(OpBuilder &builder, Location loc,
                                         ArrayAttr values) {
  int64_t size = values ? static_cast<int64_t>(values.size()) : int64_t{0};
  size = std::max<int64_t>(1, size);
  auto bufferType = MemRefType::get({size}, builder.getI64Type());
  Value buffer = builder.create<memref::AllocaOp>(loc, bufferType);

  auto emitAt = [&](int64_t idx, int64_t value) {
    Value idxVal = builder.create<arith::ConstantIndexOp>(loc, idx);
    Value val = getI64Const(builder, loc, value);
    builder.create<memref::StoreOp>(loc, val, buffer, ValueRange{idxVal});
  };

  if (!values || values.empty()) {
    emitAt(0, 1);
    return getBaseAddrAsIndex(builder, loc, buffer);
  }

  for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
    int64_t value = 1;
    if (auto intAttr = dyn_cast<IntegerAttr>(values[static_cast<size_t>(i)]))
      value = intAttr.getInt();
    if (value <= 0)
      value = 1;
    emitAt(i, value);
  }
  return getBaseAddrAsIndex(builder, loc, buffer);
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
      getStridesAndOffset(ty, strides, offset);

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

} // namespace mlir::sodap
