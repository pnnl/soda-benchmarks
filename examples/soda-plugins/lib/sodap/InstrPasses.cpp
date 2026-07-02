//===- InstrPasses.cpp - SODAP instrumentation passes -----------*- C++ -*-===//
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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <optional>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include "sodap/SODAPPasses.h"

/// Library for instrumentation functions
constexpr llvm::StringLiteral kAssertLessThen = "sodaInstrAssertLessThen";
constexpr llvm::StringLiteral kInstrHWCounters = "sodaInstrHWCounters";
constexpr llvm::StringLiteral kInstrCollectOpCounts = "sodaInstrCollectOpCounts";
constexpr llvm::StringLiteral kInstrDynamicCounter = "sodaInstrDynamicCounter";
constexpr llvm::StringLiteral kInstrDynamicCounterFlush = "sodaInstrDynamicCounterFlush";
constexpr llvm::StringLiteral kInstrDynamicCounterStartGroup = "sodaInstrDynamicCounterStartGroup";
constexpr llvm::StringLiteral kInstrMarkMatrixAccessStarts =
    "sodaInstrMarkMatrixAccessStarts";
constexpr llvm::StringLiteral kInstrDynamicCounterSetGroupFunctionName =
  "sodaInstrDynamicCounterSetGroupFunctionName";
constexpr llvm::StringLiteral kInstrTraceLinalg = "sodaInstrTraceLinalg";
constexpr llvm::StringLiteral kInstrTraceLinalgMemref =
  "sodaInstrTraceLinalgMemref";

using namespace mlir;

namespace mlir::sodap {
#define GEN_PASS_DEF_INSTRBOUNDS
#define GEN_PASS_DEF_INSTRHWCOUNTERS
#define GEN_PASS_DEF_INSTRDYNAMICOPCOUNTS
#define GEN_PASS_DEF_INSTRDYNAMICCOUNTER
#define GEN_PASS_DEF_INSTRMATRIXACCESSTRACE
#include "sodap/SODAPPasses.h.inc"

namespace {

enum class EmitCInterface : bool { Off = false, On = true };

FlatSymbolRefAttr getFunc(ModuleOp module, StringRef name, TypeRange resultType,
                          ValueRange operands, EmitCInterface emitCInterface) {
  MLIRContext *context = module.getContext();
  auto result = SymbolRefAttr::get(context, name);
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<func::FuncOp>(
        module.getLoc(), name,
        FunctionType::get(context, operands.getTypes(), resultType));
    func.setPrivate();
    if (static_cast<bool>(emitCInterface))
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));
  }
  return result;
}

func::CallOp createFuncCall(OpBuilder &builder, Location loc, StringRef name,
                            TypeRange resultType, ValueRange operands,
                            EmitCInterface emitCInterface) {
  auto module = builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  FlatSymbolRefAttr fn =
      getFunc(module, name, resultType, operands, emitCInterface);
  return builder.create<func::CallOp>(loc, resultType, fn, operands);
}

struct LoopTrackedOpCounts {
  int64_t loads = 0;
  int64_t stores = 0;
  int64_t fpArithmetic = 0;
  int64_t intArithmetic = 0;
};

bool isTrackedFloatType(Type type) {
  auto floatType = dyn_cast<FloatType>(type);
  return floatType && (floatType.isF32() || floatType.isF64());
}

bool isTrackedIntegerType(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType &&
         (integerType.getWidth() == 32 || integerType.getWidth() == 64);
}

LoopTrackedOpCounts countTrackedOpsInLoopBody(scf::ForOp forOp) {
  LoopTrackedOpCounts counts;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<memref::LoadOp>(op)) {
      ++counts.loads;
      continue;
    }
    if (isa<memref::StoreOp>(op)) {
      ++counts.stores;
      continue;
    }
    if (auto addf = dyn_cast<arith::AddFOp>(op);
        addf && isTrackedFloatType(addf.getType())) {
      ++counts.fpArithmetic;
      continue;
    }
    if (auto mulf = dyn_cast<arith::MulFOp>(op);
        mulf && isTrackedFloatType(mulf.getType())) {
      ++counts.fpArithmetic;
      continue;
    }
    if (auto addi = dyn_cast<arith::AddIOp>(op);
        addi && isTrackedIntegerType(addi.getType())) {
      ++counts.intArithmetic;
      continue;
    }
    if (auto muli = dyn_cast<arith::MulIOp>(op);
        muli && isTrackedIntegerType(muli.getType())) {
      ++counts.intArithmetic;
      continue;
    }
  }
  return counts;
}

// Instrument all scf::ForOp in a function with the assertion call
void instrumentForOpsInFunc(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  funcOp.walk([&](scf::ForOp forOp) {
    auto &bodyOps = forOp.getBody()->getOperations();
    if (!bodyOps.empty()) {
      if (auto call = llvm::dyn_cast<func::CallOp>(&bodyOps.front())) {
        if (call.getCallee() == kAssertLessThen)
          return; // Already instrumented
      }
    }
    builder.setInsertionPointToStart(forOp.getBody());
    auto loc = forOp.getLoc();
    auto iv = forOp.getInductionVar();
    auto ub = forOp.getUpperBound();
    createFuncCall(builder, loc, kAssertLessThen, TypeRange{},
                   ValueRange{iv, ub}, EmitCInterface::Off);
  });
}

// Instrument all scf::ForOp in a function with HW counter calls
void instrumentForOpsWithHWCounter(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  int64_t loopId = 0;
  funcOp.walk([&](scf::ForOp forOp) {
    builder.setInsertionPointToStart(forOp.getBody());
    auto loc = forOp.getLoc();
    // Create constants for arguments
    auto runTrue = builder.create<arith::ConstantOp>(
        loc, builder.getIntegerType(1), builder.getBoolAttr(true));
    auto runFalse = builder.create<arith::ConstantOp>(
        loc, builder.getIntegerType(1), builder.getBoolAttr(false));
    auto idVal = builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(), builder.getIndexAttr(loopId++));
    // Insert HW counter start at the beginning
    createFuncCall(builder, loc, kInstrHWCounters, TypeRange{},
                   ValueRange{runTrue, idVal}, EmitCInterface::Off);
    // Insert HW counter stop just before the yield
    auto *terminator = forOp.getBody()->getTerminator();
    builder.setInsertionPoint(terminator);
    createFuncCall(builder, loc, kInstrHWCounters, TypeRange{},
                   ValueRange{runFalse, idVal}, EmitCInterface::Off);
  });
}

// Instrument all scf::ForOp with runtime calls that collect tracked operation
// counts at loop boundaries.
void instrumentForOpsWithDynamicOpCounts(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  int64_t loopId = 0;
  auto i64Type = builder.getI64Type();
  auto createI64Const = [&](Location loc, int64_t value) -> Value {
    return builder.create<arith::ConstantOp>(loc, i64Type,
                                             builder.getI64IntegerAttr(value));
  };
  funcOp.walk([&](scf::ForOp forOp) {
    const LoopTrackedOpCounts counts = countTrackedOpsInLoopBody(forOp);
    auto loc = forOp.getLoc();

    builder.setInsertionPointToStart(forOp.getBody());
    auto runStart = createI64Const(loc, 1);
    auto idVal = builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(), builder.getIndexAttr(loopId++));
    auto zero = createI64Const(loc, 0);

    createFuncCall(builder, loc, kInstrCollectOpCounts, TypeRange{},
                   ValueRange{runStart, idVal, zero, zero, zero, zero},
                   EmitCInterface::Off);

    auto *terminator = forOp.getBody()->getTerminator();
    builder.setInsertionPoint(terminator);
    auto runStop = createI64Const(loc, 0);
    auto loads = createI64Const(loc, counts.loads);
    auto stores = createI64Const(loc, counts.stores);
    auto fpArith = createI64Const(loc, counts.fpArithmetic);
    auto intArith = createI64Const(loc, counts.intArithmetic);

    createFuncCall(builder, loc, kInstrCollectOpCounts, TypeRange{},
                   ValueRange{runStop, idVal, loads, stores, fpArith, intArith},
                   EmitCInterface::Off);
  });
}

enum class DynamicCounterKind : int64_t {
  MemrefLoad = 0,
  MemrefStore = 1,
  ArithInt = 2,
  ArithFloat = 3,
  Scf = 4,
  Affine = 5,
};

struct DynamicCounterSelection {
  bool memrefLoad = true;
  bool memrefStore = true;
  bool arithInt = true;
  bool arithFloat = true;
  bool scf = true;
  bool affine = true;
};

static bool parseTrackedKinds(StringRef trackedKinds,
                              DynamicCounterSelection &selection,
                              std::string &errorMessage) {
  // If not specified (or only whitespace), keep defaults: track all.
  if (trackedKinds.trim().empty()) {
    selection = DynamicCounterSelection{};
    return true;
  }

  // Start from nothing selected; add items as we parse.
  DynamicCounterSelection parsed;
  parsed.memrefLoad = false;
  parsed.memrefStore = false;
  parsed.arithInt = false;
  parsed.arithFloat = false;
  parsed.scf = false;
  parsed.affine = false;

  // Split on commas, then split each piece on '+'.
  llvm::SmallVector<StringRef, 8> commaTokens;
  trackedKinds.split(commaTokens, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  llvm::SmallVector<StringRef, 8> items;
  for (StringRef t : commaTokens)
    t.split(items, '+', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  // If the string was something like "," or "+", treat as "all".
  if (items.empty()) {
    selection = DynamicCounterSelection{};
    return true;
  }

  llvm::StringSet<> seen;
  for (StringRef raw : items) {
    std::string kind = raw.trim().lower();
    if (kind.empty())
      continue;

    // Dedup.
    if (!seen.insert(kind).second)
      continue;

    if (kind == "all") {
      selection = DynamicCounterSelection{}; // defaults are all true
      return true; // "all" overrides everything; we're done.
    } else if (kind == "memref-load") {
      parsed.memrefLoad = true;
    } else if (kind == "memref-store") {
      parsed.memrefStore = true;
    } else if (kind == "memref") {
      parsed.memrefLoad = true;
      parsed.memrefStore = true;
    } else if (kind == "arith-int") {
      parsed.arithInt = true;
    } else if (kind == "arith-float") {
      parsed.arithFloat = true;
    } else if (kind == "arith") {
      parsed.arithInt = true;
      parsed.arithFloat = true;
    } else if (kind == "scf") {
      parsed.scf = true;
    } else if (kind == "affine") {
      parsed.affine = true;
    } else {
      errorMessage =
          ("Unknown tracked kind '" + kind +
           "'. Expected one of: all, memref-load, memref-store, memref, "
           "arith-int, arith-float, arith, scf, affine");
      return false;
    }
  }

  selection = parsed;
  return true;
}

std::optional<DynamicCounterKind>
getDynamicCounterKind(Operation &op, const DynamicCounterSelection &selection) {
  if (isa<func::CallOp>(op) || op.hasAttr("soda.dynamic_counter.instrumented"))
    return std::nullopt;

  if (selection.memrefLoad && isa<memref::LoadOp>(op))
    return DynamicCounterKind::MemrefLoad;
  if (selection.memrefStore && isa<memref::StoreOp>(op))
    return DynamicCounterKind::MemrefStore;

  StringRef dialect = op.getName().getDialectNamespace();
  if (dialect == "arith") {
    if (isa<arith::ConstantOp>(op))
      return std::nullopt;

    bool hasTrackedFloat = false;
    bool hasTrackedInt = false;
    auto markType = [&](Type t) {
      if (isTrackedFloatType(t))
        hasTrackedFloat = true;
      if (isTrackedIntegerType(t))
        hasTrackedInt = true;
    };

    for (Type t : op.getResultTypes())
      markType(t);
    for (Value operand : op.getOperands())
      markType(operand.getType());

    if (selection.arithFloat && hasTrackedFloat)
      return DynamicCounterKind::ArithFloat;
    if (selection.arithInt && hasTrackedInt)
      return DynamicCounterKind::ArithInt;
  }

  if (selection.scf && dialect == "scf" && !isa<scf::YieldOp>(op))
    return DynamicCounterKind::Scf;
  if (selection.affine && dialect == "affine")
    return DynamicCounterKind::Affine;

  return std::nullopt;
}

static std::string getFunctionName(Operation *op) {
  if (auto parentFunc = op->getParentOfType<func::FuncOp>())
    return parentFunc.getSymName().str();
  return "unknown";
}

void instrumentForOpsWithDynamicCounter(
    func::FuncOp funcOp, const DynamicCounterSelection &selection) {
  if (funcOp.isExternal() || funcOp.getBody().empty())
    return;

  llvm::SmallVector<std::pair<Operation *, DynamicCounterKind>, 64> worklist;
  funcOp.walk([&](Operation *op) {
    if (auto kind = getDynamicCounterKind(*op, selection))
      worklist.emplace_back(op, *kind);
  });

  OpBuilder builder(funcOp.getContext());

  for (auto [op, kind] : worklist) {
    builder.setInsertionPoint(op);
    auto loc = op->getLoc();
    auto i64Type = builder.getI64Type();
    Value counterId = builder.create<arith::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(static_cast<int64_t>(kind)));
    Value delta = builder.create<arith::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(1));
    createFuncCall(builder, loc, kInstrDynamicCounter, TypeRange{},
                   ValueRange{counterId, delta}, EmitCInterface::Off);
    op->setAttr("soda.dynamic_counter.instrumented",
                UnitAttr::get(funcOp.getContext()));
  }

  // Insert start/flush calls around each top-level loop.
  int64_t groupId = 0;
  llvm::SmallVector<Operation *, 16> topLevelLoops;
  
  // Collect top-level loops (both scf.for and affine.for) in the function body.
  for (Operation &op : funcOp.getBody().front().getOperations()) {
    if (isa<scf::ForOp, affine::AffineForOp>(&op))
      topLevelLoops.push_back(&op);
  }

  for (auto loopOp : topLevelLoops) {
    auto loc = loopOp->getLoc();
    std::string functionName = getFunctionName(loopOp);

    builder.setInsertionPoint(loopOp);

    for (size_t i = 0; i < functionName.size(); ++i) {
      auto groupIdVal = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(),
        builder.getI64IntegerAttr(static_cast<int64_t>(groupId)));
      auto charCodeVal = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(
                      static_cast<int64_t>(
                        static_cast<unsigned char>(
                          functionName[i]))));
      createFuncCall(builder, loc, kInstrDynamicCounterSetGroupFunctionName,
             TypeRange{},
             ValueRange{groupIdVal, charCodeVal},
             EmitCInterface::Off);
    }

    // Start group before the loop executes.
    builder.setInsertionPoint(loopOp);
    auto startGroupIdVal = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(),
        builder.getI64IntegerAttr(static_cast<int64_t>(groupId)));
    createFuncCall(builder, loc, kInstrDynamicCounterStartGroup, TypeRange{},
                   ValueRange{startGroupIdVal}, EmitCInterface::Off);

    // Flush once after the loop to print loop totals.
    builder.setInsertionPointAfter(loopOp);
    auto flushGroupIdVal = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(),
        builder.getI64IntegerAttr(static_cast<int64_t>(groupId)));
    createFuncCall(builder, loc, kInstrDynamicCounterFlush, TypeRange{},
                   ValueRange{flushGroupIdVal}, EmitCInterface::Off);

    groupId++;
  }
}

enum class LinalgGenericOperandKind : int64_t {
  Input = 0,
  Output = 1,
};

static Value createI64Constant(OpBuilder &builder, Location loc,
                               int64_t value) {
  return builder.create<arith::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(value));
}

static std::string sanitizeSymbolSuffix(StringRef text) {
  std::string sanitized;
  sanitized.reserve(text.size());
  for (char ch : text) {
    if (llvm::isAlnum(static_cast<unsigned char>(ch))) {
      sanitized.push_back(ch);
      continue;
    }
    sanitized.push_back('_');
  }
  return sanitized;
}

static std::string affineMapToString(AffineMap map) {
  std::string out;
  llvm::raw_string_ostream os(out);
  map.print(os);
  return os.str();
}

static std::string indexingMapsToString(linalg::GenericOp genericOp) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << '[';
  llvm::interleaveComma(genericOp.getIndexingMapsArray(), os,
                        [&](AffineMap map) { os << affineMapToString(map); });
  os << ']';
  return os.str();
}

static std::string iteratorTypesToString(linalg::GenericOp genericOp) {
  std::string out;
  llvm::raw_string_ostream os(out);
  genericOp.getIteratorTypesAttr().print(os);
  return os.str();
}

struct StringLiteralValue {
  Value unrankedMemref;
};

static StringLiteralValue getOrCreateStringLiteral(
    OpBuilder &builder, ModuleOp module, Location loc, StringRef prefix,
    StringRef text, llvm::StringMap<std::string> &cache,
    int64_t &nextStringId) {
  std::string cacheKey = text.str();
  auto cacheIt = cache.find(cacheKey);
  std::string symbolName;
  if (cacheIt != cache.end()) {
    symbolName = cacheIt->second;
  } else {
    symbolName =
        (Twine("__soda_trace_") + prefix + "_" + Twine(nextStringId++)).str();
    cache[cacheKey] = symbolName;

    OpBuilder moduleBuilder(module.getBodyRegion());
    auto i8Type = moduleBuilder.getIntegerType(8);
    auto memrefType =
        MemRefType::get({static_cast<int64_t>(text.size())}, i8Type);
    auto tensorType = RankedTensorType::get(
        {static_cast<int64_t>(text.size())}, i8Type);

    SmallVector<APInt> values;
    values.reserve(text.size());
    for (unsigned char ch : text)
      values.emplace_back(8, ch);

    auto init = DenseIntElementsAttr::get(tensorType, values);
    moduleBuilder.create<memref::GlobalOp>(
        loc, symbolName, moduleBuilder.getStringAttr("private"), memrefType,
        init, /*constant=*/true, /*alignment=*/IntegerAttr{});
  }

  auto i8Type = builder.getIntegerType(8);
  auto memrefType = MemRefType::get({static_cast<int64_t>(text.size())}, i8Type);
  auto unrankedType = UnrankedMemRefType::get(i8Type, 0);
  Value global = builder.create<memref::GetGlobalOp>(loc, memrefType, symbolName);
  Value unranked = builder.create<memref::CastOp>(loc, unrankedType, global);
  return {unranked};
}

static void emitMemrefSpec(OpBuilder &builder, Location loc, int64_t opId,
                           LinalgGenericOperandKind operandKind,
                           int64_t operandIndex, Value memref) {
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (!memrefType)
    return;

  Value baseAddrIdx =
      builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
  Value baseAddr =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), baseAddrIdx);

  auto metadata = builder.create<memref::ExtractStridedMetadataOp>(loc, memref);
  Value offset = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                    metadata.getOffset());

  Value opIdVal = createI64Constant(builder, loc, opId);
  Value operandKindVal =
      createI64Constant(builder, loc, static_cast<int64_t>(operandKind));
  Value operandIndexVal = createI64Constant(builder, loc, operandIndex);
  auto i64Type = builder.getI64Type();
  auto dimsType = MemRefType::get({memrefType.getRank()}, i64Type);
  auto dimsUnrankedType = UnrankedMemRefType::get(i64Type, 0);
  Value dimsBuffer = builder.create<memref::AllocaOp>(loc, dimsType);
  Value stridesBuffer = builder.create<memref::AllocaOp>(loc, dimsType);

  for (auto [dimIndex, dimVal] : llvm::enumerate(metadata.getSizes())) {
    Value dimI64 =
        builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), dimVal);
    Value dimPos = builder.create<arith::ConstantIndexOp>(loc, dimIndex);
    builder.create<memref::StoreOp>(loc, dimI64, dimsBuffer, ValueRange{dimPos});
  }

  for (auto [strideIndex, strideVal] : llvm::enumerate(metadata.getStrides())) {
    Value strideI64 =
        builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), strideVal);
    Value stridePos = builder.create<arith::ConstantIndexOp>(loc, strideIndex);
    builder.create<memref::StoreOp>(loc, strideI64, stridesBuffer,
                                    ValueRange{stridePos});
  }

  Value dimsUnranked =
      builder.create<memref::CastOp>(loc, dimsUnrankedType, dimsBuffer);
  Value stridesUnranked =
      builder.create<memref::CastOp>(loc, dimsUnrankedType, stridesBuffer);

  createFuncCall(builder, loc, kInstrTraceLinalgMemref, TypeRange{},
                 ValueRange{opIdVal, operandKindVal, operandIndexVal, baseAddr,
                            offset, dimsUnranked, stridesUnranked},
                 EmitCInterface::Off);
}

static void instrumentMatrixAccessTrace(func::FuncOp funcOp) {
  if (funcOp.isExternal() || funcOp.getBody().empty())
    return;

  ModuleOp module = funcOp->getParentOfType<ModuleOp>();
  llvm::SmallVector<Operation *, 16> linalgOps;
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::LinalgOp>(op))
      linalgOps.push_back(op);
  });
  if (linalgOps.empty())
    return;

  OpBuilder builder(funcOp.getContext());
  int64_t opId = 0;
  int64_t nextStringId = 0;
  llvm::StringMap<std::string> stringLiteralCache;
  const std::string missingMetadata = "-";

  for (Operation *op : linalgOps) {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    Location loc = op->getLoc();
    builder.setInsertionPoint(op);

    std::string opName = op->getName().getStringRef().str();
    if (StringRef(opName).starts_with("linalg."))
      opName = opName.substr(strlen("linalg."));

    std::string mapsText = missingMetadata;
    std::string iteratorTypesText = missingMetadata;
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
      mapsText = indexingMapsToString(genericOp);
      iteratorTypesText = iteratorTypesToString(genericOp);
    }

    auto opNameVal = getOrCreateStringLiteral(builder, module, loc, "opname",
                                              opName, stringLiteralCache,
                                              nextStringId);
    auto mapsVal = getOrCreateStringLiteral(builder, module, loc, "maps",
                                            mapsText, stringLiteralCache,
                                            nextStringId);
    auto iterVal = getOrCreateStringLiteral(builder, module, loc, "iters",
                                            iteratorTypesText,
                                            stringLiteralCache, nextStringId);

    Value opIdVal = createI64Constant(builder, loc, opId);
    Value numInputsVal =
        createI64Constant(builder, loc, linalgOp.getNumDpsInputs());
    Value numOutputsVal =
        createI64Constant(builder, loc, linalgOp.getNumDpsInits());
    createFuncCall(builder, loc, kInstrTraceLinalg, TypeRange{},
                   ValueRange{opIdVal, opNameVal.unrankedMemref, numInputsVal,
                              numOutputsVal, mapsVal.unrankedMemref,
                              iterVal.unrankedMemref},
                   EmitCInterface::Off);

    for (auto [inputIndex, input] : llvm::enumerate(linalgOp.getDpsInputs())) {
      emitMemrefSpec(builder, loc, opId, LinalgGenericOperandKind::Input,
                     inputIndex, input);
    }
    for (auto [outputIndex, output] : llvm::enumerate(linalgOp.getDpsInits())) {
      emitMemrefSpec(builder, loc, opId, LinalgGenericOperandKind::Output,
                     outputIndex, output);
    }

    ++opId;
  }
}

class SODAPInstrBounds : public impl::InstrBoundsBase<SODAPInstrBounds> {
public:
  using impl::InstrBoundsBase<SODAPInstrBounds>::InstrBoundsBase;
  void runOnOperation() final {
    getOperation()->walk(
        [](func::FuncOp funcOp) { instrumentForOpsInFunc(funcOp); });
  }
};

class SODAInstrBoundsWithHWCounters
    : public impl::InstrHWCountersBase<SODAInstrBoundsWithHWCounters> {
public:
  using impl::InstrHWCountersBase<
      SODAInstrBoundsWithHWCounters>::InstrHWCountersBase;
  void runOnOperation() final {
    getOperation()->walk(
        [](func::FuncOp funcOp) { instrumentForOpsWithHWCounter(funcOp); });
  }
};

class SODAPInstrDynamicOpCounts
    : public impl::InstrDynamicOpCountsBase<SODAPInstrDynamicOpCounts> {
public:
  using impl::InstrDynamicOpCountsBase<
      SODAPInstrDynamicOpCounts>::InstrDynamicOpCountsBase;
  void runOnOperation() final {
    getOperation()->walk(
        [](func::FuncOp funcOp) { instrumentForOpsWithDynamicOpCounts(funcOp); });
  }
};

class SODAPInstrDynamicCounter
    : public impl::InstrDynamicCounterBase<SODAPInstrDynamicCounter> {
public:
  using impl::InstrDynamicCounterBase<
      SODAPInstrDynamicCounter>::InstrDynamicCounterBase;

  void runOnOperation() final {
    DynamicCounterSelection selection;
    std::string errorMessage;
    if (!parseTrackedKinds(trackedKinds, selection, errorMessage)) {
      getOperation()->emitError(errorMessage);
      signalPassFailure();
      return;
    }

    getOperation()->walk([&](func::FuncOp funcOp) {
      instrumentForOpsWithDynamicCounter(funcOp, selection);
    });
  }
};

class SODAPInstrMatrixAccessTrace
    : public impl::InstrMatrixAccessTraceBase<SODAPInstrMatrixAccessTrace> {
public:
  using impl::InstrMatrixAccessTraceBase<
      SODAPInstrMatrixAccessTrace>::InstrMatrixAccessTraceBase;

  void runOnOperation() final {
    getOperation()->walk(
        [&](func::FuncOp funcOp) { instrumentMatrixAccessTrace(funcOp); });
  }
};
} // namespace
} // namespace mlir::sodap
