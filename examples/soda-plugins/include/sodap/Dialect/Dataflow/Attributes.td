//===- Attributes.td - Dataflow dialect attributes ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_DIALECT_DATAFLOW_ATTRIBUTES_TD
#define SODAP_DIALECT_DATAFLOW_ATTRIBUTES_TD

def MM : I32EnumAttrCase<"MM", 0>;
def LITE : I32EnumAttrCase<"LITE", 1>;
def STREAM : I32EnumAttrCase<"STREAM", 2>;
def AxiKind : I32EnumAttr<"AxiKind", "Kind of AXI port", [MM, LITE, STREAM]> {
  let cppNamespace = "::mlir::sodap::dataflow";
}

def Resource : DataflowAttr<"Resource"> {
  let summary = "Resource utilization information";
  let description = [{
    Utilization of the on-chip resources that matter for HLS QoR: LUT, DSP,
    and BRAM.

    Example: `#dataflow.r<lut=100, dsp=4, bram=2>`
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "r";
  let parameters = (ins "int64_t":$lut, "int64_t":$dsp, "int64_t":$bram);
}

def Timing : DataflowAttr<"Timing"> {
  let summary = "Timing information";
  let description = [{
    Scheduling information for an operation: begin and end cycle, latency, and
    initiation interval.

    Example: `#dataflow.t<0 -> 8, 8, 1>`
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "t";
  let parameters = (ins "int64_t":$begin, "int64_t":$end, "int64_t":$latency,
                        "int64_t":$interval);
}

def LoopInfo : DataflowAttr<"LoopInfo"> {
  let summary = "Loop information";
  let description = [{
    Loop information used by estimators and optimizers: flattened trip count,
    iteration latency, and minimum initiation interval.

    Example: `#dataflow.l<flattenTripCount=1024, iterLatency=3, minII=1>`
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "l";
  let parameters = (ins "int64_t":$flattenTripCount, "int64_t":$iterLatency,
                        "int64_t":$minII);
}

def LoopDirective : DataflowAttr<"LoopDirective"> {
  let summary = "Aggregation of loop directives";
  let description = [{
    All loop-level HLS directives: pipelining, target initiation interval,
    dataflow, and flattening.

    Example:
    `#dataflow.pipeline_ii<pipeline=true, targetII=1, dataflow=false, flatten=false>`
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "pipeline_ii";
  let parameters = (ins "bool":$pipeline, "int64_t":$targetII, "bool":$dataflow,
                        "bool":$flatten);
}

def FuncDirective : DataflowAttr<"FuncDirective"> {
  let summary = "Aggregation of function directives";
  let description = [{
    All function-level HLS directives: pipelining, target interval, and
    dataflow.

    Example: `#dataflow.fd<pipeline=false, targetInterval=1, dataflow=true>`
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "fd";
  let parameters = (ins "bool":$pipeline, "int64_t":$targetInterval,
                        "bool":$dataflow);
}

#endif // SODAP_DIALECT_DATAFLOW_ATTRIBUTES_TD
