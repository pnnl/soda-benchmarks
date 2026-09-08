//===- Passes.h - Dataflow dialect transforms -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_DIALECT_DATAFLOW_TRANSFORMS_PASSES_H
#define SODAP_DIALECT_DATAFLOW_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "sodap/Dialect/Dataflow/Dataflow.h"

#include <memory>

namespace mlir {
namespace sodap {
namespace dataflow {

#define GEN_PASS_DECL
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"

} // namespace dataflow
} // namespace sodap
} // namespace mlir

#endif // SODAP_DIALECT_DATAFLOW_TRANSFORMS_PASSES_H
