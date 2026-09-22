//===- AnnotateTosaOps.cpp - Annotate TOSA ops with metrics --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sodap/AnalysisPasses.h"
#include "sodap/SODAPPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include <vector>

namespace mlir {
namespace sodap {

#define GEN_PASS_DEF_ANNOTATETOSAOPS
#include "sodap/SODAPPasses.h.inc"
} // namespace sodap
} // namespace mlir

using namespace mlir;
using namespace mlir::sodap;

namespace {

class AnnotateTosaOps
    : public sodap::impl::AnnotateTosaOpsBase<AnnotateTosaOps> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(getOperation());
    mlir::ModuleOp clonedModule = module.clone();

    // Store opInfo for each op in a list
    std::vector<sodap::linalg::reports::LinalgOpInfo> opInfos;

    // Collect opInfo for each TOSA op in the cloned module
    clonedModule.walk([&](Operation *op) {
      sodap::linalg::reports::LinalgOpInfo opInfo;
      if (op->getDialect() && op->getDialect()->getNamespace() == "tosa") {
        lowerSingleTosaOp(op->clone(), opInfo);
        opInfos.push_back(std::move(opInfo));
      }
    });

    // Walk over the original module and add attributes to TOSA ops
    size_t opInfoIdx = 0;
    module.walk([&](Operation *op) {
      if (op->getDialect() && op->getDialect()->getNamespace() == "tosa") {
        if (opInfoIdx < opInfos.size()) {
          auto &info = opInfos[opInfoIdx++];
          op->setAttr("numArithmeticOpsEstimative",
                      IntegerAttr::get(IntegerType::get(op->getContext(), 64),
                                       info.numArithmeticOpsEstimative));
          op->setAttr("numMemoryOpsEstimative",
                      IntegerAttr::get(IntegerType::get(op->getContext(), 64),
                                       info.numMemoryOpsEstimative));
          op->setAttr("numArithmeticOpsInKernel",
                      IntegerAttr::get(IntegerType::get(op->getContext(), 64),
                                       info.numArithmeticOpsInKernel));
          op->setAttr("numMemoryOpsInKernel",
                      IntegerAttr::get(IntegerType::get(op->getContext(), 64),
                                       info.numMemoryOpsInKernel));
        }
      }
    });
  }

private:
  void lowerSingleTosaOp(Operation *tosaOp,
                         sodap::linalg::reports::LinalgOpInfo &opInfo) {
    MLIRContext *ctx = tosaOp->getContext();
    // 1. Create a temporary module.
    OwningOpRef<ModuleOp> tmpModule = ModuleOp::create(tosaOp->getLoc());

    // 2. Build a dummy function whose name comes from the op.
    SmallVector<Type> argTypes;
    SmallVector<unsigned> operandIsConst;
    for (Value v : tosaOp->getOperands()) {
      if (isa_and_nonnull<tosa::ConstOp>(v.getDefiningOp())) {
        operandIsConst.push_back(1);
      } else {
        operandIsConst.push_back(0);
        argTypes.push_back(v.getType()); // full tensor type
      }
    }
    SmallVector<Type> resTypes(tosaOp->getResultTypes().begin(),
                               tosaOp->getResultTypes().end());

    auto funcType = FunctionType::get(ctx, argTypes, resTypes);
    std::string funcName = tosaOp->getName().getStringRef().str();
    OpBuilder builder(tmpModule->getBodyRegion());
    auto func =
        builder.create<func::FuncOp>(tosaOp->getLoc(), funcName, funcType);

    // 3. Add an entry block.
    Block *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // 4. Build the operand list for the cloned op:
    SmallVector<Value> newOperands;
    unsigned blockArgIdx = 0;
    for (unsigned i = 0; i < tosaOp->getNumOperands(); ++i) {
      Value oldOperand = tosaOp->getOperand(i);
      if (operandIsConst[i]) {
        // Clone the defining constant op into the entry block.
        Operation *constDef = oldOperand.getDefiningOp();
        Operation *newConst = builder.clone(*constDef);
        newOperands.push_back(newConst->getResult(0));
      } else {
        // Use block argument for external input.
        newOperands.push_back(entry->getArgument(blockArgIdx++));
      }
    }

    // 5. Clone the TOSA op with new operands.
    OperationState state(tosaOp->getLoc(), tosaOp->getName());
    state.addOperands(newOperands);
    state.addTypes(resTypes);
    state.addAttributes(tosaOp->getAttrs());
    Operation *cloned = builder.create(state);

    // 6. Return the results.
    builder.create<func::ReturnOp>(tosaOp->getLoc(), cloned->getResults());

    // 7. Run the TOSA lowering pipeline on the temporary module.
    PassManager pm(ctx);

    // Build an OpPassManager for func.func (nested pipeline):
    OpPassManager &funcPM = pm.nest<func::FuncOp>();

    // Combined bufferization pipeline
    std::string pipelineStr =
        // Tensor to Linalg conversion
        "convert-tensor-to-linalg,"
        "empty-tensor-to-alloc-tensor,"
        "eliminate-empty-tensors,"
        // Bufferization passes
        "one-shot-bufferize{"
        "function-boundary-type-conversion=identity-layout-map "
        "bufferize-function-boundaries "
        "allow-return-allocs-from-loops "
        "unknown-type-conversion=identity-layout-map"
        "},"
        "func-bufferize,"
        "buffer-deallocation-simplification,"
        "bufferization-lower-deallocations,"
        "buffer-results-to-out-params,"
        // Final cleanup
        "canonicalize,"
        "cse";

    // Now parse the inner pipeline for func.func:
    if (failed(parsePassPipeline(
            "tosa-to-arith{include-apply-rescale=true}, tosa-to-tensor, "
            "tosa-to-linalg-named, tosa-to-linalg, "
            "linalg-generalize-named-ops, canonicalize",
            funcPM)))
      return signalPassFailure();

    if (failed(parsePassPipeline(pipelineStr, pm)))
      return signalPassFailure();

    if (failed(pm.run(*tmpModule))) {
      llvm::errs() << "Lowering failed for op: " << *tosaOp << "\n";
      return;
    }

    // For each linalg.generic op in the lowered module, update opInfo.
    tmpModule->walk([&](mlir::linalg::GenericOp genericOp) {
      sodap::linalg::reports::LinalgOpInfo tempOpInfo;
      collectLinalgOperationInfo(tempOpInfo, genericOp);
      // Accumulate relevant info into opInfo.
      opInfo.numArithmeticOpsEstimative +=
          tempOpInfo.numArithmeticOpsEstimative;
      opInfo.numMemoryOpsEstimative += tempOpInfo.numMemoryOpsEstimative;
      opInfo.numArithmeticOpsInKernel += tempOpInfo.numArithmeticOpsInKernel;
      opInfo.numMemoryOpsInKernel += tempOpInfo.numMemoryOpsInKernel;
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::sodap::createAnnotateTosaOpsPass() {
  return std::make_unique<AnnotateTosaOps>();
}

