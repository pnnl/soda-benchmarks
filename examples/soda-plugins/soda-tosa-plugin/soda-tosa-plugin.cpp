//===- soda-tosa-plugin.cpp - SODAP passes on TOSA --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass plugin of its own for the passes on TOSA: SODAPlugin binds its MLIR
// symbols to the tool loading it, and soda-opt, which loads SODAPlugin, has
// no TOSA dialect. Load this one in mlir-opt only:
//   mlir-opt --load-pass-plugin=SODATosaPlugin.so ...
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/Plugins/PassPlugin.h"
#include "sodap/Tosa/Passes.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "SODATosaPasses", LLVM_VERSION_STRING,
          []() { ::mlir::sodap::registerSODAPTosaPasses(); }};
}
