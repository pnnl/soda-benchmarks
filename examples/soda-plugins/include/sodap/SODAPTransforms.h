//===-- SODAPTransforms.h - SODAP transform dialect ops --------*- c++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SODAP transform dialect extension operations.
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_SODAPTRANSFORMS_H
#define SODAP_SODAPTRANSFORMS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

#define GET_OP_CLASSES
#include "sodap/SODAPTransforms.h.inc"

namespace mlir {
namespace sodap {
/// Registers the SODAP Transform dialect extension.
void registerSODAPTransforms(::mlir::DialectRegistry &registry);
} // namespace sodap
} // namespace mlir

#endif // SODAP_SODAPTRANSFORMS_H
