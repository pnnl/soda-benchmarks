//===- InstrPasses.cpp - SODAP instrumentation passes -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "sodap/SODAPPasses.h"

/// Library for instrumentation functions
static constexpr const char *kAssertLessThen = "sodaInstrAssertLessThen";
static constexpr const char *kInstrHWCounters = "sodaInstrHWCounters";

using namespace mlir;

namespace mlir::sodap {
#define GEN_PASS_DEF_INSTRBOUNDS
#define GEN_PASS_DEF_INSTRHWCOUNTERS
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
} // namespace
} // namespace mlir::sodap
