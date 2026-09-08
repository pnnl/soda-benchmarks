//===- Ops.td - Dataflow dialect operations ----------------*- tablegen -*-===//
//
// Copyright 2020-2021 The ScaleHLS Authors
// Copyright 2024 Suhail Basalama
// Copyright 2026, Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Portions of this file were modified by Suhail Basalama in 2024 under the
// MIT License. The full MIT text can be found in the root NOTICE file.
//
// Ported and modified for the SODA dataflow dialect in 2026.
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_DIALECT_DATAFLOW_OPS_TD
#define SODAP_DIALECT_DATAFLOW_OPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "sodap/Dialect/Dataflow/Interfaces.td"

class StreamOf<list<Type> allowedTypes> : Type<And<[
    CPred<"::llvm::isa<::mlir::sodap::dataflow::StreamType>($_self)">,
    Concat<"[](::mlir::Type elementType) { return ",
      SubstLeaves<"$_self", "elementType", AnyTypeOf<allowedTypes>.predicate>,
      "; }(::llvm::cast<::mlir::sodap::dataflow::StreamType>($_self)"
      ".getElementType())">
  ]>,
  "stream of " # AnyTypeOf<allowedTypes>.summary # " values",
  "::mlir::sodap::dataflow::StreamType"
>;

def AnyStream : StreamOf<[AnyType]>;

def AnyMemrefOrStream : Type<Or<[AnyMemRef.predicate, AnyStream.predicate]>,
    "memref or stream values">;

//===----------------------------------------------------------------------===//
// Stream channel operations
//===----------------------------------------------------------------------===//

def StreamOp : DataflowOp<"stream", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Declare a stream channel";
  let description = [{
    Instantiate a stream channel connecting a producer stage to a consumer
    stage. This is the MLIR analogue of declaring an `hls::stream` in an HLS
    dataflow region.

    Example:

    ```mlir
    %c = dataflow.stream {depth = 1024 : i32} : <f32, 1024>
    ```
  }];

  let arguments = (ins I32Attr:$depth);
  let results = (outs AnyStream:$channel);
  let assemblyFormat = "attr-dict `:` type($channel)";

  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "mlir::Type":$channel),
        "build($_builder, $_state, channel, 1);">
  ];
}

def StreamReadOp : DataflowOp<"stream_read", []> {
  let summary = "Read a stream channel";
  let description = [{
    Read (pop) a value from a stream channel. The absence of a result indicates
    that the popped value is dropped. A channel may be read many times.

    Example:

    ```mlir
    %v = dataflow.stream_read %c : (!dataflow.stream<f32, 1024>) -> f32
    ```
  }];

  let arguments = (ins AnyStream:$channel);
  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = [{
    $channel attr-dict `:` functional-type($channel, $result)
  }];

  let hasVerifier = 1;
}

def StreamWriteOp : DataflowOp<"stream_write", []> {
  let summary = "Write a stream channel";
  let description = [{
    Write (push) a value to a stream channel.

    Example:

    ```mlir
    dataflow.stream_write %c, %v : <f32, 1024>, f32
    ```
  }];

  let arguments = (ins AnyStream:$channel, AnyType:$value);
  let assemblyFormat = [{
    $channel `,` $value attr-dict `:` type($channel) `,` type($value)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Stage hierarchy operations
//===----------------------------------------------------------------------===//

def DispatchOp : DataflowOp<"dispatch", [RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"YieldOp">,
      ParentOneOf<["func::FuncOp", "affine::AffineForOp"]>]> {
  let summary = "Represent a dataflow dispatch region";
  let description = [{
    Dispatch has a transparent region holding the stages to be dispatched. It
    is the analogue of an HLS `#pragma HLS DATAFLOW` region: every stage in the
    body runs concurrently, communicating through the streams declared in it.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword ( `:` type($results)^ )? $body";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Get the terminator yield op.
    YieldOp getYieldOp();
  }];
}

def TaskOp : DataflowOp<"task", [
      DeclareOpInterfaceMethods<StageLikeInterface>, RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"YieldOp">, HasParent<"DispatchOp">]> {
  let summary = "Represent a dataflow task";
  let description = [{
    Task has a transparent region holding the operations executed by one stage.
    It represents a task partition before the stage's inputs and outputs have
    been made explicit; `sodap-dataflow-tasks-to-nodes` lowers it to a node op.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword ( `:` type($results)^ )? $body";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Get the parent dispatch op.
    DispatchOp getDispatchOp();

    /// Get the terminator yield op.
    YieldOp getYieldOp();
  }];
}

def NodeOp : DataflowOp<"node", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      DeclareOpInterfaceMethods<StageLikeInterface>, IsolatedFromAbove,
      AffineScope, SingleBlock, NoTerminator, AttrSizedOperandSegments,
      HasParent<"DispatchOp">]> {
  let summary = "Represent a dataflow node";
  let description = [{
    Node is a dataflow stage with explicit inputs, outputs, and parameters. It
    is isolated from above, so every value it uses arrives through a block
    argument -- which is what makes each node independently emittable as an HLS
    sub-function.

    Example:

    ```mlir
    dataflow.node (%in) -> (%out) : (!dataflow.stream<f32, 8>) ->
                                    (!dataflow.stream<f32, 8>) {
      ...
    }
    ```
  }];

  let arguments = (ins Variadic<AnyMemrefOrStream>:$inputs,
    Variadic<AnyMemrefOrStream>:$outputs, Variadic<AnyType>:$params,
    I32ArrayAttr:$inputTaps, OptionalAttr<I32Attr>:$level);

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `(` $inputs `)` `->` `(` $outputs `)` ( `[` $params^ `]` )? attr-dict `:`
    functional-type($inputs, $outputs) ( `[` type($params)^ `]` )? $body
  }];

  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs,
        "mlir::ValueRange":$params, "llvm::ArrayRef<unsigned>":$inputTaps,
        "mlir::IntegerAttr":$level), [{
          auto newInputTaps = llvm::SmallVector<int32_t>(
            llvm::map_range(inputTaps, [](unsigned a) { return (int32_t)a; }));
          build($_builder, $_state, inputs, outputs, params,
                $_builder.getI32ArrayAttr(newInputTaps), level);
        }]>,

    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs,
        "mlir::ValueRange":$params, "llvm::ArrayRef<unsigned>":$inputTaps), [{
          build($_builder, $_state, inputs, outputs, params, inputTaps,
                mlir::IntegerAttr());
        }]>,

    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs,
        "llvm::ArrayRef<unsigned>":$inputTaps), [{
          build($_builder, $_state, inputs, outputs, mlir::ValueRange(),
                inputTaps);
        }]>,

    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs,
        "mlir::ValueRange":$params), [{
          build($_builder, $_state, inputs, outputs, params,
                llvm::SmallVector<unsigned>(inputs.size(), 0));
        }]>,

    OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$outputs),
        "build($_builder, $_state, inputs, outputs, mlir::ValueRange());">
  ];

  let extraClassDeclaration = [{
    /// Get the parent dispatch op.
    DispatchOp getDispatchOp();

    /// Get and set input taps.
    void setInputTap(unsigned idx, unsigned tap);
    unsigned getInputTap(unsigned idx);
    llvm::SmallVector<unsigned> getInputTapsAsInt();

    /// Get the number of inputs, outputs, and params.
    unsigned getNumInputs();
    unsigned getNumOutputs();
    unsigned getNumParams();

    /// Get the kind of an operand: input, output, or param.
    OperandKind getOperandKind(mlir::OpOperand &operand);
    OperandKind getOperandKind(unsigned operandIdx);

    /// Get the input, output, and param block arguments.
    llvm::iterator_range<mlir::Block::args_iterator> getInputArgs();
    llvm::iterator_range<mlir::Block::args_iterator> getOutputArgs();
    llvm::iterator_range<mlir::Block::args_iterator> getParamArgs();
  }];
}

def YieldOp : DataflowOp<"yield", [NoMemoryEffect, ReturnLike, Terminator,
      ParentOneOf<["DispatchOp", "TaskOp"]>]> {
  let summary = "Terminate and yield results of a dispatch or task op";

  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "$results attr-dict ( `:` type($results)^ )?";

  let builders = [OpBuilder<(ins), "build($_builder, $_state, mlir::ValueRange());">];
}

#endif // SODAP_DIALECT_DATAFLOW_OPS_TD
