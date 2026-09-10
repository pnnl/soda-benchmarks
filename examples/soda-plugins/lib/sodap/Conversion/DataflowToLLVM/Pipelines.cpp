//===- Pipelines.cpp - DataflowToLLVM pipeline ----------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sodap/Conversion/DataflowToLLVM/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct DataflowToLLVMPipelineOptions
    : public PassPipelineOptions<DataflowToLLVMPipelineOptions> {
  // The only option with no default: see Passes.td.
  Option<std::string> topFunc{
      *this, "top-func",
      llvm::cl::desc("Top function of the design (required)")};
  Option<std::string> archFile{
      *this, "arch-file", llvm::cl::init("architecture.xml"),
      llvm::cl::desc("Where to write the Bambu --architecture-xml file")};
  Option<std::string> clangxx{
      *this, "clangxx", llvm::cl::init(SODAP_BAMBU_CLANGXX),
      llvm::cl::desc("C++ compiler used to build the specialization unit")};
  Option<std::string> includePandaPath{
      *this, "include-panda-path", llvm::cl::init(SODAP_BAMBU_INCLUDE),
      llvm::cl::desc("Directory holding ac_channel.h")};
};
} // namespace

/// Takes the flat call-graph form -- func.func per node, dataflow streams
/// across the calls -- down to the LLVM dialect for Bambu. Kept separate from
/// the front half so iterating on the backend does not mean rerunning it.
void mlir::sodap::registerDataflowToLLVMPipeline() {
  PassPipelineRegistration<DataflowToLLVMPipelineOptions>(
      "sodap-dataflow-to-llvm-pipeline",
      "Lower the dataflow call-graph form to the LLVM dialect for Bambu, "
      "emitting the --architecture-xml file on the way. Anchor this on "
      "builtin.module.",
      [](OpPassManager &pm, const DataflowToLLVMPipelineOptions &opts) {
        // Must run first: it needs the stream types to tell a FIFO from an
        // array, and the next pass erases them.
        pm.addPass(createEmitBambuArchitecture(
            {opts.archFile, opts.topFunc, opts.includePandaPath}));
        pm.addPass(
            createConvertDataflowToLLVM({opts.clangxx, opts.includePandaPath}));

        // Node-local buffers become stack arrays instead of malloc calls.
        // A func.func pass: --pass-pipeline does not nest it on its own.
        pm.addNestedPass<func::FuncOp>(
            bufferization::createPromoteBuffersToStackPass(1024 * 1024));

        pm.addPass(createLowerAffinePass());
        pm.addPass(createConvertSCFToCFPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        pm.addPass(createConvertMathToLLVMPass());
        pm.addPass(memref::createExpandStridedMetadataPass());
        // Strided metadata expansion may create affine expressions.
        pm.addPass(createLowerAffinePass());
        pm.addPass(createFinalizeMemRefToLLVMConversionPass());
        // Bare pointers: Bambu's array protocol wants a plain pointer per
        // array, not an exploded memref descriptor.
        ConvertFuncToLLVMPassOptions funcToLLVMOptions;
        funcToLLVMOptions.useBarePtrCallConv = true;
        pm.addPass(createConvertFuncToLLVMPass(funcToLLVMOptions));
        pm.addPass(createConvertIndexToLLVMPass());
        pm.addPass(createConvertControlFlowToLLVMPass());
        pm.addPass(createReconcileUnrealizedCastsPass());
        // Only now can the canonicalizer fold extractvalue(insertvalue) and
        // drop the memref descriptors func-to-llvm rebuilt at every entry.
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
      });
}
