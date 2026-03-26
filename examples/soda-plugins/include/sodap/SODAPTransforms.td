//===-- SODAPTransforms.td - SODAP transform dialect ops ---*- tablegen -*-===//
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

#ifndef SODAP_TRANSFORMS
#define SODAP_TRANSFORMS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TagOpsOp : Op<Transform_Dialect, "sodap.tag_ops",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Tags payload operations with an incrementing unique ID";
  let description = [{
    For each payload operation associated with the handle, sets a `uid` string
    attribute of the form `<prefix>_<N>`, where `<prefix>` is provided as an
    attribute and `<N>` is an ever-increasing counter starting from 0.

    The counter increments in the order the payload operations appear in the
    handle's association list.

    Only reads the operand. Modifies the payload.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$target,
    StrAttr:$prefix);

  let results = (outs);

  let assemblyFormat = "$target `,` $prefix attr-dict `:` type($target)";
}

#endif // SODAP_TRANSFORMS
