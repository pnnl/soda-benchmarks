//===- Pipelines.cpp - AffineToDataflow pipeline --------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sodap/Conversion/AffineToDataflow/Passes.h"
#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void mlir::sodap::registerAffineToDataflowPipeline() {
  PassPipelineRegistration<>(
      "sodap-affine-to-dataflow",
      "Turn affine IR into a dataflow design: buffers between loop bands "
      "become stream channels, and the bands themselves become dataflow nodes "
      "inside a dispatch region. Anchor this on func.func.",
      [](OpPassManager &pm) {
        pm.addPass(createConvertMemRefsToStreams());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(createCreateDataflowTasks());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(dataflow::createTasksToNodes());
      });
}
