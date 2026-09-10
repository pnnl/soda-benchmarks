//===- Passes.h - DataflowToLLVM passes -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_CONVERSION_DATAFLOWTOLLVM_PASSES_H
#define SODAP_CONVERSION_DATAFLOWTOLLVM_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "sodap/Dialect/Dataflow/Dataflow.h"
#include "sodap/Config.h"

#include <memory>
#include <string>

namespace mlir {
namespace sodap {

#define GEN_PASS_DECL
#include "sodap/Conversion/DataflowToLLVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "sodap/Conversion/DataflowToLLVM/Passes.h.inc"

/// Register the `sodap-dataflow-to-llvm-pipeline`, which takes the flat
/// call-graph form `sodap-dataflow-nodes-to-func` produces down to the LLVM
/// dialect for Bambu.
void registerDataflowToLLVMPipeline();

} // namespace sodap
} // namespace mlir

#endif // SODAP_CONVERSION_DATAFLOWTOLLVM_PASSES_H
