//===- BambuTypeNames.h - C spelling of MLIR scalars ------------*- C++ -*-===//
//
// Copyright (c) 2026 Tommaso Fellegara
//
//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The one place that spells an MLIR scalar type in C. Both halves of the Bambu
// backend need it and must agree: EmitBambuArchitecture writes the typename
// into the architecture XML, ConvertDataflowToLLVM instantiates ac_channel<T>
// with it. Two spellings of one port is a mismatch Bambu only reports much
// later, if at all.
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_CONVERSION_DATAFLOWTOLLVM_BAMBUTYPENAMES_H
#define SODAP_CONVERSION_DATAFLOWTOLLVM_BAMBUTYPENAMES_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Twine.h"

#include <string>

namespace mlir {
namespace sodap {

/// C spelling of `type`, or the empty string when Bambu's interface does not
/// support it: only float, double and the 8/16/32/64-bit integers are.
inline std::string bambuTypeName(Type type) {
  if (isa<Float32Type>(type))
    return "float";
  if (isa<Float64Type>(type))
    return "double";
  if (auto intType = dyn_cast<IntegerType>(type)) {
    auto width = intType.getWidth();
    if (width == 8 || width == 16 || width == 32 || width == 64)
      // Signless is what this flow produces, and it travels as signed.
      return ((intType.isUnsigned() ? "uint" : "int") + Twine(width) + "_t")
          .str();
  }
  return "";
}

} // namespace sodap
} // namespace mlir

#endif // SODAP_CONVERSION_DATAFLOWTOLLVM_BAMBUTYPENAMES_H
