//===- Interfaces.td - Dataflow dialect interfaces ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_DIALECT_DATAFLOW_INTERFACES_TD
#define SODAP_DIALECT_DATAFLOW_INTERFACES_TD

include "mlir/IR/OpBase.td"

def StageLikeInterface : OpInterface<"StageLikeInterface"> {
  let description = [{
    Implemented by operations that represent a dataflow stage, i.e. a unit of
    computation that runs concurrently with its sibling stages and communicates
    with them through streams.
  }];
  let cppNamespace = "::mlir::sodap::dataflow";

  let methods = [
    InterfaceMethod<"Return the body region of the stage",
      "::mlir::Region &", "getBody", (ins), "return $_op.getBody();">,
    InterfaceMethod<"Check whether the stage contains nested stages",
      "bool", "hasHierarchy", (ins), [{
        return $_op.walk([&](StageLikeInterface stage) {
          if (stage != $_op)
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        }).wasInterrupted();
      }]>,
    InterfaceMethod<"Return whether the value is a livein of the stage",
                    "bool", "isLivein", (ins "::mlir::Value":$value)>,
    InterfaceMethod<"Return the liveins of the stage",
                    "::llvm::SmallVector<::mlir::Value>", "getLiveins">,
    InterfaceMethod<"Return the internal users of a stage livein",
                    "::llvm::SmallVector<::mlir::Operation *>", "getLiveinUsers",
                    (ins "::mlir::Value":$livein)>,
  ];
}

#endif // SODAP_DIALECT_DATAFLOW_INTERFACES_TD
