//===- SODAPPasses.h - SODAP passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SODAP_SODAPPASSES_H
#define SODAP_SODAPPASSES_H

#include "mlir/Pass/Pass.h"
// #include "sodap/SODAPDialect.h"
// #include "sodap/SODAPOps.h"
#include <memory>

#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace sodap {
#define GEN_PASS_DECL
#include "sodap/SODAPPasses.h.inc"

/// Creates a pass to print op graphs.
std::unique_ptr<Pass> createPrintOpGraphPass(raw_ostream &os = llvm::errs());

#define GEN_PASS_REGISTRATION
#include "sodap/SODAPPasses.h.inc"
} // namespace sodap
} // namespace mlir

#endif
