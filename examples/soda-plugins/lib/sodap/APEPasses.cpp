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

static func::FuncOp ensureIssueAPERequestDeclaration(ModuleOp module) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kIssueAPERequest))
    return existing;

  OpBuilder moduleBuilder(module.getBodyRegion());
  MLIRContext *ctx = module.getContext();

  auto i32Type = moduleBuilder.getI32Type();
  auto i1Type = moduleBuilder.getI1Type();
  auto indexType = moduleBuilder.getIndexType();

  SmallVector<Type, 10> inputs = {
      indexType, // base_addr
      i32Type,   // tensor_id
      i32Type,   // strategy
      indexType, // i0
      indexType, // i1
      indexType, // i2
      indexType, // i3
      i1Type,    // is_write
      i32Type,   // prefetch_ahead
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
  auto fn = moduleBuilder.create<func::FuncOp>(
      module.getLoc(), kPreIssueAPERequest, fnType);
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

static int32_t inferPrefetchDistance(StringRef accessKind) {
  if (accessKind == "write")
    return 1;
  if (accessKind == "readwrite")
    return 2;
  return 2;
}

static DictionaryAttr buildTensorInfoDict(Builder &b, int32_t tensorId,
                                          StringRef accessKind,
                                          StringRef role,
                                          MemRefType memrefType,
                                          StringRef indexingMapText) {
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
  int32_t prefetchDistance = inferPrefetchDistance(accessKind);

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
      b.getNamedAttr("rank", b.getI64IntegerAttr(memrefType.getRank())),
      b.getNamedAttr("shape", b.getArrayAttr(shapeAttrs)),
      b.getNamedAttr("strides", b.getArrayAttr(strideAttrs)),
      b.getNamedAttr("indexing_map", b.getStringAttr(indexingMapText)),
      b.getNamedAttr("contiguous_vector_len",
                     b.getI32IntegerAttr(contiguousVecLen)),
      b.getNamedAttr("suggested_prefetch_distance",
                     b.getI32IntegerAttr(prefetchDistance)),
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
  int32_t suggestedPrefetchDistance = 1;
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
    if (auto prefetch =
            dyn_cast_or_null<IntegerAttr>(info.get("suggested_prefetch_distance")))
      md.suggestedPrefetchDistance = static_cast<int32_t>(prefetch.getInt());
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

static Value getI1Const(OpBuilder &builder, Location loc, bool value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI1Type(),
                                           builder.getBoolAttr(value));
}

static Value getBaseAddrAsIndex(OpBuilder &builder, Location loc, Value memref) {
  return builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
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
      SmallVector<Attribute, 8> perOperandInfos;

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

        std::string mapText;
        {
          llvm::raw_string_ostream os(mapText);
          indexingMap.print(os);
        }

        DictionaryAttr info = buildTensorInfoDict(b, tensorId, accessKind, role,
                                                  memrefTy, mapText);
        perOperandInfos.push_back(info);

        // Attach compact per-tensor metadata to the value source so later
        // passes can recover a memref from tensor_id even after linalg is
        // lowered away.
        auto tensorInfo = b.getDictionaryAttr(
            {b.getNamedAttr("tensor_id", b.getI32IntegerAttr(tensorId)),
             b.getNamedAttr("contiguous_vector_len",
                            info.get("contiguous_vector_len")),
             b.getNamedAttr("suggested_prefetch_distance",
                            info.get("suggested_prefetch_distance"))});
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

      if (!perOperandInfos.empty()) {
        DictionaryAttr opInfo = b.getDictionaryAttr(
            {b.getNamedAttr("operands", b.getArrayAttr(perOperandInfos))});
        linalgOp->setAttr(kAPETensorInfoAttr, opInfo);

        // Keep one compact marker right before each linalg op so the metadata
        // survives lowering without adding per-operand noise.
        callBuilder.setInsertionPoint(linalgOp);
        auto preIssue = callBuilder.create<func::CallOp>(
            linalgOp.getLoc(), TypeRange{},
            SymbolRefAttr::get(callBuilder.getContext(), kPreIssueAPERequest),
            ValueRange{});
        preIssue->setAttr("ape.pre_issue_info", opInfo);
      }
    });
  }
};

struct InsertionKey {
  Operation *loopOp = nullptr;
  Value memref;
  bool isWrite = false;
  int32_t requestOrdinal = 0;

  bool operator==(const InsertionKey &other) const {
    return loopOp == other.loopOp && memref == other.memref &&
           isWrite == other.isWrite && requestOrdinal == other.requestOrdinal;
  }
};

struct InsertionKeyInfo : llvm::DenseMapInfo<InsertionKey> {
  static InsertionKey getEmptyKey() {
    return {llvm::DenseMapInfo<Operation *>::getEmptyKey(),
            llvm::DenseMapInfo<Value>::getEmptyKey(), false, -1};
  }
  static InsertionKey getTombstoneKey() {
    return {llvm::DenseMapInfo<Operation *>::getTombstoneKey(),
            llvm::DenseMapInfo<Value>::getTombstoneKey(), false, -2};
  }
  static unsigned getHashValue(const InsertionKey &k) {
    return llvm::hash_combine(k.loopOp, k.memref.getAsOpaquePointer(), k.isWrite,
                              k.requestOrdinal);
  }
  static bool isEqual(const InsertionKey &a, const InsertionKey &b) {
    return a == b;
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

    ensureIssueAPERequestDeclaration(funcOp->getParentOfType<ModuleOp>());

    OpBuilder builder(funcOp.getContext());
    llvm::DenseSet<std::pair<Operation *, int32_t>> seen;
    DenseMap<int32_t, Value> tensorIdToMemref;

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

    auto getNextAffineFor = [&](Operation *op) -> affine::AffineForOp {
      for (Operation *cur = op->getNextNode(); cur; cur = cur->getNextNode()) {
        if (auto forOp = dyn_cast<affine::AffineForOp>(cur))
          return forOp;
      }
      return {};
    };

    funcOp.walk([&](func::CallOp preIssueCall) {
      if (preIssueCall.getCallee() != kPreIssueAPERequest)
        return;

      auto info = dyn_cast_or_null<DictionaryAttr>(
          preIssueCall->getAttr("ape.pre_issue_info"));
      if (!info)
        return;

      auto operandsAttr = dyn_cast_or_null<ArrayAttr>(info.get("operands"));
      if (!operandsAttr || operandsAttr.empty())
        return;

      affine::AffineForOp insertionLoop = getNextAffineFor(preIssueCall);
      if (!insertionLoop)
        return;

      Location loc = insertionLoop.getLoc();
      builder.setInsertionPointToStart(insertionLoop.getBody());

      // Build IV payload from loops in scope at insertion point. We keep up to
      // 4 dimensions and pad the rest with zero as required by the ABI.
      SmallVector<affine::AffineForOp, 6> loopsInScope;
      for (Operation *cur = insertionLoop.getOperation(); cur;
           cur = cur->getParentOp()) {
        if (auto loop = dyn_cast<affine::AffineForOp>(cur))
          loopsInScope.push_back(loop);
      }
      llvm::reverse(loopsInScope);

      SmallVector<Value, 4> ivPayload;
      for (affine::AffineForOp loop : loopsInScope) {
        ivPayload.push_back(loop.getInductionVar());
        if (ivPayload.size() == 4)
          break;
      }
      while (ivPayload.size() < 4)
        ivPayload.push_back(getZeroIndex(builder, loc));

      // We currently issue requests only for read/readwrite operands because
      // the immediate goal is prefetching incoming matrices.
      for (Attribute operandAttr : operandsAttr) {
        auto operandInfo = dyn_cast<DictionaryAttr>(operandAttr);
        if (!operandInfo)
          continue;

        auto accessKindAttr =
            dyn_cast_or_null<StringAttr>(operandInfo.get("access_kind"));
        if (!accessKindAttr)
          continue;
        StringRef accessKind = accessKindAttr.getValue();
        bool isWrite = (accessKind == "write");
        if (isWrite)
          continue;

        auto tensorIdAttr =
            dyn_cast_or_null<IntegerAttr>(operandInfo.get("tensor_id"));
        if (!tensorIdAttr)
          continue;
        int32_t tensorIdValue = static_cast<int32_t>(tensorIdAttr.getInt());

        std::pair<Operation *, int32_t> key{insertionLoop.getOperation(),
                                            tensorIdValue};
        if (seen.contains(key))
          continue;
        seen.insert(key);

        int32_t contiguousVecLen = 1;
        if (auto vecLen =
                dyn_cast_or_null<IntegerAttr>(operandInfo.get("contiguous_vector_len")))
          contiguousVecLen = static_cast<int32_t>(vecLen.getInt());

        int32_t prefetchAhead = 1;
        if (auto prefetch = dyn_cast_or_null<IntegerAttr>(
                operandInfo.get("suggested_prefetch_distance")))
          prefetchAhead = static_cast<int32_t>(prefetch.getInt());

        int32_t strategy = contiguousVecLen > 1 ? 0 : 1;
        Value baseAddr = getZeroIndex(builder, loc);
        if (auto it = tensorIdToMemref.find(tensorIdValue);
            it != tensorIdToMemref.end()) {
          baseAddr = getBaseAddrAsIndex(builder, loc, it->second);
        }
        Value tensorId = getI32Const(builder, loc, tensorIdValue);
        Value strategyVal = getI32Const(builder, loc, strategy);
        Value isWriteVal = getI1Const(builder, loc, false);
        Value prefetchAheadVal = getI32Const(builder, loc, prefetchAhead);

        auto call = builder.create<func::CallOp>(
            loc, TypeRange{}, SymbolRefAttr::get(builder.getContext(),
                                                 kIssueAPERequest),
            ValueRange{baseAddr, tensorId, strategyVal, ivPayload[0],
                       ivPayload[1], ivPayload[2], ivPayload[3], isWriteVal,
                       prefetchAheadVal});
        call->setAttr("ape.insertion_reason",
                      builder.getStringAttr(
                          "inserted from pre_issue_APE_request metadata"));
      }
    });
  }
};

} // namespace

void populateAPE(OpPassManager &pm) {
  pm.addPass(createLinalgAPEAnalysis());
  pm.addPass(createAffineAPEInsertion());
}

} // namespace mlir::sodap
