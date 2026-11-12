//===- TilesToLLVM.cpp - Each tile module to its LLVM IR file -------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_TILESTOLLVM
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

using namespace mlir;
using namespace mlir::sodap;
using namespace mlir::sodap::dataflow;

namespace {
struct TilesToLLVM : public dataflow::impl::TilesToLLVMBase<TilesToLLVM> {
  using TilesToLLVMBase::TilesToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
  }

  void runOnOperation() override {
    for (auto tile : getOperation().getOps<ModuleOp>()) {
      if (!tile->hasAttr("sodap.tile"))
        continue;
      if (failed(translate(tile)))
        return signalPassFailure();
    }
  }

  LogicalResult translate(ModuleOp tile) {
    StringRef name = *tile.getSymName();
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmModule =
        translateModuleToLLVMIR(tile, llvmContext);
    if (!llvmModule)
      return tile.emitError("cannot translate tile '") << name << "' to LLVM IR";
    std::string path = getTileFilePath(tile, "acc.ll");
    std::error_code error;
    llvm::raw_fd_ostream os(path, error);
    if (error)
      return tile.emitError("cannot write '") << path << "'";
    llvmModule->print(os, nullptr);

    auto kernel = tile.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (!kernel)
      return tile.emitError("expected an llvm.func named '") << name << "'";
    for (Operation &op : *tile.getBody())
      op.dropAllReferences();
    for (Operation &op : llvm::make_early_inc_range(*tile.getBody()))
      if (&op != kernel.getOperation())
        op.erase();
    kernel.getBody().getBlocks().clear();
    return success();
  }
};
} // namespace
