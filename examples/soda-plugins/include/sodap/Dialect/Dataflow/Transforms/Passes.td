//===- Passes.td - Dataflow dialect transforms -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_DIALECT_DATAFLOW_TRANSFORMS_PASSES_TD
#define SODAP_DIALECT_DATAFLOW_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TasksToNodes :
    Pass<"sodap-dataflow-tasks-to-nodes", "::mlir::func::FuncOp"> {
  let summary = "Lower dataflow tasks to nodes with explicit ports";
  let description = [{
    Replace every `dataflow.task` with a `dataflow.node` whose inputs, outputs,
    and parameters are explicit.

    Liveness analysis over the task body determines the ports: a livein of
    memref, stream, or tensor type becomes an output if the task writes it and
    an input otherwise, while every other livein becomes a parameter. Because a
    node is `IsolatedFromAbove`, each port arrives as a block argument, which is
    what lets a node later be emitted as a standalone HLS sub-function.
  }];
  let dependentDialects = ["::mlir::sodap::dataflow::DataflowDialect"];
}

def NodesToFunc :
    Pass<"sodap-dataflow-nodes-to-func", "::mlir::ModuleOp"> {
  let summary = "Lower dataflow nodes to functions and calls";
  let description = [{
    Outline each `dataflow.node` into its own `func.func` and replace it with a
    `func.call`, then unwrap the enclosing `dataflow.dispatch`.

    This flattens the dataflow hierarchy into the plain call-graph form that
    HLS backends consume, and is deliberately kept out of the default pipeline
    so that the structural dataflow operations survive when they are what the
    consumer wants to see.
  }];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::func::FuncDialect"
  ];
}

def InferTopInterface :
    Pass<"sodap-dataflow-infer-top-interface", "::mlir::ModuleOp"> {
  let summary = "Turn the external streams of the top into axis ports";
  let description = [{
    Works on the call-graph form `sodap-dataflow-nodes-to-func` produces. A
    stream argument of the top is part of the design's external interface
    when some stage actually reads or writes it; those are retyped from
    `!dataflow.stream<T, D>` to `!dataflow.stream<T, D, axis>`, in the top and
    in every function the stream is passed to, so the calls still type-check.

    A stream nobody touches is left a fifo, and so is every channel declared
    inside the design. A stream that is both read and written is rejected:
    an AXI4-Stream port carries data in one direction only. So is a function
    that receives an axis port at one call site and a fifo at another, since
    one signature cannot be both.

    Example:

    ```mlir
    func.func @top_node0(%req: !dataflow.stream<i64, 8>, ...)
    func.func @top(%req: !dataflow.stream<i64, 8>, ...) {
      call @top_node0(%req, ...) : ...
    }
    // after sodap-dataflow-infer-top-interface{top-func=top}, if top_node0
    // writes %req:
    func.func @top_node0(%req: !dataflow.stream<i64, 8, axis>, ...)
    func.func @top(%req: !dataflow.stream<i64, 8, axis>, ...)
    ```
  }];
  let options = [
    Option<"topFunc", "top-func", "std::string", /*default=*/"",
           "Top function of the design (required)">
  ];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::func::FuncDialect"
  ];
}

#endif // SODAP_DIALECT_DATAFLOW_TRANSFORMS_PASSES_TD
