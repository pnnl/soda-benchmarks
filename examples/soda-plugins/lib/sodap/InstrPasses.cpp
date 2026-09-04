//===- InstrPasses.cpp - SODAP instrumentation passes -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
static constexpr const char *kInstrChangeLocation = "sodaInstrChangeLocation";
static constexpr const char *kInstrCollectOpCounts = "sodaInstrCollectOpCounts";
static constexpr const char *kInstrDynamicCounter = "sodaInstrDynamicCounter";
static constexpr const char *kVectorDot = "sodaVectorDot";

/// Sentinel location value recognized by the `sodaInstrHWCounters` hardware
/// IP as "print every counter's final value once", instead of a normal
/// start/stop location id. Encoded as all-bits-set (-1) so it can be carried
/// through the existing 2-argument `(action, location)` call signature
/// without changing the interface: real loop ids are small non-negative
/// numbers assigned incrementally, so this value is never produced by the
/// normal instrumentation walk.
static constexpr int64_t kReportSentinelLocation = -1;

using namespace mlir;

namespace mlir::sodap {
#define GEN_PASS_DEF_INSTRBOUNDS
#define GEN_PASS_DEF_INSTRHWCOUNTERS
#define GEN_PASS_DEF_INSTRCHANGELOCATION
#define GEN_PASS_DEF_INSTRDYNAMICOPCOUNTS
#define GEN_PASS_DEF_INSTRDYNAMICCOUNTER
#define GEN_PASS_DEF_SWAPOPTOHW
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

// Insert a single finalize call before every `func.return` in `funcOp` that
// asks the `sodaInstrHWCounters` hardware IP to print every counter's final
// value once, instead of relying on the per-stop `$display` (which prints on
// every loop stop and floods the simulation log).
static void instrumentFuncWithHWCounterReport(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  funcOp.walk([&](func::ReturnOp returnOp) {
    builder.setInsertionPoint(returnOp);
    auto loc = returnOp.getLoc();
    auto runTrue = builder.create<arith::ConstantOp>(
        loc, builder.getIntegerType(1), builder.getBoolAttr(true));
    auto reportLoc = builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(),
        builder.getIndexAttr(kReportSentinelLocation));
    createFuncCall(builder, loc, kInstrHWCounters, TypeRange{},
                   ValueRange{runTrue, reportLoc}, EmitCInterface::Off);
  });
}

// Instrument all scf::ForOp in a function with a single change-location call
// mapped to a single-active-counter SODA hardware IP. Unlike
// `instrumentForOpsWithHWCounter`, this only emits one call per loop (no
// separate stop call): switching the active location implicitly stops the
// previous location and starts the new one in hardware.
static void instrumentForOpsWithChangeLocation(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  int loopId = 0;
  funcOp.walk([&](scf::ForOp forOp) {
    builder.setInsertionPointToStart(forOp.getBody());
    auto loc = forOp.getLoc();
    auto idVal = builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(), builder.getIndexAttr(loopId++));
    createFuncCall(builder, loc, kInstrChangeLocation, TypeRange{},
                   ValueRange{idVal}, EmitCInterface::Off);
  });
}

// Insert a single finalize call before every `func.return` in `funcOp` that
// asks the `sodaInstrChangeLocation` hardware IP to print every location's
// final count once. Because only one counter is ever active at a time, the
// single-active-counter design naturally lends itself to a print-once-at-end
// report (unlike the concurrent-counters IP, which prints on every stop).
static void instrumentFuncWithChangeLocationReport(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  funcOp.walk([&](func::ReturnOp returnOp) {
    builder.setInsertionPoint(returnOp);
    auto loc = returnOp.getLoc();
    auto reportLoc = builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(),
        builder.getIndexAttr(kReportSentinelLocation));
    createFuncCall(builder, loc, kInstrChangeLocation, TypeRange{},
                   ValueRange{reportLoc}, EmitCInterface::Off);
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

static void
instrumentForOpsWithDynamicCounter(func::FuncOp funcOp,
                                   const DynamicCounterSelection &selection) {
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

// Returns true if `type` is a memref of static rank 1 (a fixed-size vector).
static bool isStaticVectorMemRef(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  return memrefType && memrefType.getRank() == 1 && !memrefType.isDynamicDim(0);
}

// Matches a `linalg.dot` op that can be swapped for the `sodaVectorDot`
// hardware module: two same-length, statically-shaped 1-D memref inputs,
// reducing into a rank-0 (scalar) memref output. Dynamically-shaped dot
// products are left untouched, since the HW module is modeled as a
// fixed-size vector engine.
static bool isMatchableVectorDot(linalg::DotOp op) {
  if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
    return false;

  Value a = op.getInputs()[0];
  Value b = op.getInputs()[1];
  if (!isStaticVectorMemRef(a.getType()) || !isStaticVectorMemRef(b.getType()))
    return false;

  auto aType = cast<MemRefType>(a.getType());
  auto bType = cast<MemRefType>(b.getType());
  if (aType.getDimSize(0) != bType.getDimSize(0))
    return false;

  auto outType = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
  return outType && outType.getRank() == 0;
}

// Replace a single `linalg.dot` with a call to `@sodaVectorDot(A, B, len,
// out)`, mirroring a Bambu memory-master hardware IP that streams both
// operands from memory and writes the reduced result back to `out`.
static void replaceVectorDot(linalg::DotOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  Value a = op.getInputs()[0];
  Value b = op.getInputs()[1];
  Value out = op.getOutputs()[0];
  auto aType = cast<MemRefType>(a.getType());

  builder.setInsertionPoint(op);
  Value len = builder.create<arith::ConstantOp>(
      loc, builder.getIndexType(), builder.getIndexAttr(aType.getDimSize(0)));
  createFuncCall(builder, loc, kVectorDot, TypeRange{},
                 ValueRange{a, b, len, out}, EmitCInterface::Off);
  op.erase();
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
    getOperation()->walk([&](func::FuncOp funcOp) {
      instrumentForOpsWithHWCounter(funcOp);
      if (reportAtEnd)
        instrumentFuncWithHWCounterReport(funcOp);
    });
  }
};

class SODAPInstrChangeLocation
    : public impl::InstrChangeLocationBase<SODAPInstrChangeLocation> {
public:
  using impl::InstrChangeLocationBase<
      SODAPInstrChangeLocation>::InstrChangeLocationBase;
  void runOnOperation() final {
    getOperation()->walk([](func::FuncOp funcOp) {
      instrumentForOpsWithChangeLocation(funcOp);
      instrumentFuncWithChangeLocationReport(funcOp);
    });
  }
};

class SODAPInstrDynamicOpCounts
    : public impl::InstrDynamicOpCountsBase<SODAPInstrDynamicOpCounts> {
public:
  using impl::InstrDynamicOpCountsBase<
      SODAPInstrDynamicOpCounts>::InstrDynamicOpCountsBase;
  void runOnOperation() final {
    getOperation()->walk([](func::FuncOp funcOp) {
      instrumentForOpsWithDynamicOpCounts(funcOp);
    });
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

class SODAPSwapOpToHW : public impl::SwapOpToHWBase<SODAPSwapOpToHW> {
public:
  using impl::SwapOpToHWBase<SODAPSwapOpToHW>::SwapOpToHWBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Collect matches first to avoid modifying the IR while walking it.
    SmallVector<linalg::DotOp> opsToReplace;
    module.walk([&](linalg::DotOp op) {
      if (isMatchableVectorDot(op))
        opsToReplace.push_back(op);
    });

    for (auto op : opsToReplace)
      replaceVectorDot(op, builder);
  }
};
} // namespace
} // namespace mlir::sodap
