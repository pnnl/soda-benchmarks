//===- Passes.h - SODAP passes on TOSA --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_TOSA_PASSES_H
#define SODAP_TOSA_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "sodap/Dialect/Dataflow/Dataflow.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sodap {

#define GEN_PASS_DECL
#include "sodap/Tosa/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "sodap/Tosa/Passes.h.inc"

} // namespace sodap
} // namespace mlir

#endif // SODAP_TOSA_PASSES_H
