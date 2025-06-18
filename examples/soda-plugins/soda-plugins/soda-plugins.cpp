//===- soda-plugin.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

// #include "SODA/SODADialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "sodap/MyExtension.h"
#include "sodap/SODAPPasses.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace test {
void registerTestTransformDialectExtension(mlir::DialectRegistry &);
} // namespace test

/// Dialect plugin registration mechanism.
/// Observe that it also allows to register passes.
/// Necessary symbol to register the dialect plugin.
extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "SODA", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            // registry->insert<mlir::soda::SODADialect>();
            // mlir::soda::registerPasses();
            mlir::registerAllDialects(*registry);
            mlir::registerAllExtensions(*registry);
            test::registerTestTransformDialectExtension(*registry);
            registerMyExtension(*registry);
          }};
}

/// Pass plugin registration mechanism.
/// Necessary symbol to register the pass plugin.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "SODAPasses", LLVM_VERSION_STRING, []() {
            mlir::sodap::registerPasses();
            mlir::transform::registerInterpreterPass();
          }};
}
