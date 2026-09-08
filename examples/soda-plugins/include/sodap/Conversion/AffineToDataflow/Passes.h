//===- Passes.h - AffineToDataflow passes -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_CONVERSION_AFFINETODATAFLOW_PASSES_H
#define SODAP_CONVERSION_AFFINETODATAFLOW_PASSES_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "sodap/Dialect/Dataflow/Dataflow.h"

#include <memory>

namespace mlir {
namespace sodap {

#define GEN_PASS_DECL
#include "sodap/Conversion/AffineToDataflow/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "sodap/Conversion/AffineToDataflow/Passes.h.inc"

/// Register the `sodap-affine-to-dataflow` pipeline, which chains the
/// conversion and the dataflow structuring passes.
void registerAffineToDataflowPipeline();

} // namespace sodap
} // namespace mlir

#endif // SODAP_CONVERSION_AFFINETODATAFLOW_PASSES_H
