//===- Dataflow.td - Dataflow dialect definition -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_DIALECT_DATAFLOW_TD
#define SODAP_DIALECT_DATAFLOW_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def DataflowDialect : Dialect {
  let name = "dataflow";
  let summary = "A dialect for representing HLS dataflow designs";
  let description = [{
    This dialect contains operations, types, and attributes for representing
    dataflow designs destined for HLS: explicit stream channels between
    concurrently executing stages, and the stage hierarchy itself
    (dispatch / task / node).
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::sodap::dataflow";
}

class DataflowType<string name> : TypeDef<DataflowDialect, name>;

class DataflowAttr<string name> : AttrDef<DataflowDialect, name>;

class DataflowOp<string mnemonic, list<Trait> traits = []> :
    Op<DataflowDialect, mnemonic, traits>;

include "sodap/Dialect/Dataflow/Attributes.td"
include "sodap/Dialect/Dataflow/Types.td"
include "sodap/Dialect/Dataflow/Ops.td"

#endif // SODAP_DIALECT_DATAFLOW_TD
