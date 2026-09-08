//===- Passes.td - AffineToDataflow passes -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_CONVERSION_AFFINETODATAFLOW_PASSES_TD
#define SODAP_CONVERSION_AFFINETODATAFLOW_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertMemRefsToStreams :
    Pass<"sodap-convert-memrefs-to-streams", "::mlir::func::FuncOp"> {
  let summary = "Convert single-producer/single-consumer buffers to streams";
  let description = [{
    Replace each `memref.alloc` that carries values from exactly one producing
    `affine.store` to exactly one consuming `affine.load` with a
    `dataflow.stream` channel, rewriting the store into a
    `dataflow.stream_write` and the load into a `dataflow.stream_read`.

    A buffer is only converted when producer and consumer traverse it in the
    same order, so that a FIFO can stand in for the array. Buffers that fail
    any precondition -- more than one producer or consumer, producer and
    consumer in the same loop band, or mismatched access order -- are left
    untouched as ordinary allocations.
  }];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def CreateDataflowTasks :
    Pass<"sodap-create-dataflow-tasks", "::mlir::func::FuncOp"> {
  let summary = "Partition a function body into dataflow tasks";
  let description = [{
    Wrap the body of the function in a `dataflow.dispatch` region -- the
    analogue of an HLS `#pragma HLS DATAFLOW` scope -- and group the operations
    inside it into one `dataflow.task` per top-level `affine.for` loop band,
    together with the local buffers that band accesses.

    Stream declarations are hoisted to the top of the dispatch region so that
    the tasks they connect can refer to them.
  }];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

#endif // SODAP_CONVERSION_AFFINETODATAFLOW_PASSES_TD
