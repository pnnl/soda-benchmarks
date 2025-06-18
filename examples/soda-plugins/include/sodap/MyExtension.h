//===-- MyExtension.h - Transform dialect tutorial --------------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the
// Chapter 2 of the Transform dialect tutorial.
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_MYEXTENSION_H
#define SODAP_MYEXTENSION_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

#define GET_OP_CLASSES
#include "sodap/MyExtension.h.inc"

// Transform ops have a specific namespace domain, for this reason we only add
// namespaces in the registration function.
namespace mlir {
namespace sodap {
// Registers our Transform dialect extension.
void registerMyExtension(::mlir::DialectRegistry &registry);
} // namespace sodap
} // namespace mlir

#endif