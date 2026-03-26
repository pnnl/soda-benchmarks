//===-- SODAPTransforms.cpp - SODAP transform dialect ops ------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements SODAP transform dialect extension operations.
//
//===----------------------------------------------------------------------===//

#include "sodap/SODAPTransforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

#include <string>

class SODAPTransformsExtension
    : public ::mlir::transform::TransformDialectExtension<
          SODAPTransformsExtension> {
public:
  using Base::Base;
  void init();
};

void SODAPTransformsExtension::init() {
  registerTransformOps<
#define GET_OP_LIST
#include "sodap/SODAPTransforms.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "sodap/SODAPTransforms.cpp.inc"

::mlir::DiagnosedSilenceableFailure mlir::transform::TagOpsOp::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {

  auto payload = state.getPayloadOps(getTarget());
  unsigned counter = 0;
  for (Operation *payloadOp : payload) {
    std::string uid = getPrefix().str() + "_" + std::to_string(counter++);
    payloadOp->setAttr("uid",
                        StringAttr::get(payloadOp->getContext(), uid));
  }
  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::TagOpsOp::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

namespace mlir {
namespace sodap {
void registerSODAPTransforms(::mlir::DialectRegistry &registry) {
  registry.addExtensions<SODAPTransformsExtension>();
}
} // namespace sodap
} // namespace mlir
