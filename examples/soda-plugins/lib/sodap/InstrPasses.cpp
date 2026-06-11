//===- InstrPasses.cpp - SODAP instrumentation passes -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include "sodap/SODAPPasses.h"

/// Library for instrumentation functions
static constexpr const char *kAssertLessThen = "sodaInstrAssertLessThen";
static constexpr const char *kInstrHWCounters = "sodaInstrHWCounters";
static constexpr const char *kInstrCollectOpCounts = "sodaInstrCollectOpCounts";
static constexpr const char *kInstrDynamicCounter = "sodaInstrDynamicCounter";

using namespace mlir;

namespace mlir::sodap {
#define GEN_PASS_DEF_INSTRBOUNDS
#define GEN_PASS_DEF_INSTRHWCOUNTERS
#define GEN_PASS_DEF_INSTRDYNAMICOPCOUNTS
#define GEN_PASS_DEF_INSTRDYNAMICCOUNTER
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

static bool isTrackedFloatType(Type type) {
  auto floatType = dyn_cast<FloatType>(type);
  return floatType && (floatType.isF32() || floatType.isF64());
}

static bool isTrackedIntegerType(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType &&
         (integerType.getWidth() == 32 || integerType.getWidth() == 64);
}

static LoopTrackedOpCounts countTrackedOpsInLoopBody(scf::ForOp forOp) {
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
static void instrumentForOpsInFunc(func::FuncOp funcOp) {
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
static void instrumentForOpsWithHWCounter(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  int loopId = 0;
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
static void instrumentForOpsWithDynamicOpCounts(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  int loopId = 0;
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
  DynamicCounterSelection parsed{/*memrefLoad=*/false,
                                 /*memrefStore=*/false,
                                 /*arithInt=*/false,
                                 /*arithFloat=*/false,
                                 /*scf=*/false,
                                 /*affine=*/false};

  llvm::SmallVector<StringRef, 8> items;
  llvm::SmallVector<StringRef, 8> commaParts;
  trackedKinds.split(commaParts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (commaParts.empty())
    commaParts.push_back("all");
  for (StringRef part : commaParts)
    part.split(items, '+', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  llvm::StringSet<> seen;
  for (StringRef raw : items) {
    StringRef kind = raw.trim().lower();
    if (kind.empty())
      continue;
    if (!seen.insert(kind).second)
      continue;

    if (kind == "all") {
      parsed = DynamicCounterSelection{};
      continue;
    }
    if (kind == "memref-load") {
      parsed.memrefLoad = true;
      continue;
    }
    if (kind == "memref-store") {
      parsed.memrefStore = true;
      continue;
    }
    if (kind == "memref") {
      parsed.memrefLoad = true;
      parsed.memrefStore = true;
      continue;
    }
    if (kind == "arith-int") {
      parsed.arithInt = true;
      continue;
    }
    if (kind == "arith-float") {
      parsed.arithFloat = true;
      continue;
    }
    if (kind == "arith") {
      parsed.arithInt = true;
      parsed.arithFloat = true;
      continue;
    }
    if (kind == "scf") {
      parsed.scf = true;
      continue;
    }
    if (kind == "affine") {
      parsed.affine = true;
      continue;
    }

    errorMessage =
        ("Unknown tracked kind '" + kind +
         "'. Expected one of: all, memref-load, memref-store, memref, "
         "arith-int, arith-float, arith, scf, affine")
            .str();
    return false;
  }

  selection = parsed;
  return true;
}

static std::optional<DynamicCounterKind>
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

static void instrumentForOpsWithDynamicCounter(
    func::FuncOp funcOp, const DynamicCounterSelection &selection) {
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
      signalPassFailure();
      getOperation()->emitError(errorMessage);
      return;
    }

    getOperation()->walk([&](func::FuncOp funcOp) {
      instrumentForOpsWithDynamicCounter(funcOp, selection);
    });
  }
};
} // namespace
} // namespace mlir::sodap
