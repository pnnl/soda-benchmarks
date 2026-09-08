//===- Dataflow.h - Dataflow dialect ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_DIALECT_DATAFLOW_DATAFLOW_H
#define SODAP_DIALECT_DATAFLOW_DATAFLOW_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "sodap/Dialect/Dataflow/DataflowDialect.h.inc"

#include "sodap/Dialect/Dataflow/DataflowEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "sodap/Dialect/Dataflow/DataflowTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "sodap/Dialect/Dataflow/DataflowAttributes.h.inc"

#include "sodap/Dialect/Dataflow/DataflowInterfaces.h.inc"

namespace mlir {
namespace sodap {
namespace dataflow {

/// The role an operand plays for a dataflow node.
enum class OperandKind { INPUT, OUTPUT, PARAM };

/// Names of the discardable attributes attached by the helpers below.
constexpr llvm::StringLiteral kLoopDirectiveAttrName = "loop_directive";
constexpr llvm::StringLiteral kLoopInfoAttrName = "loop_info";
constexpr llvm::StringLiteral kFuncDirectiveAttrName = "func_directive";
constexpr llvm::StringLiteral kTimingAttrName = "timing";
constexpr llvm::StringLiteral kResourceAttrName = "resource";

/// Timing attribute utils.
TimingAttr getTiming(Operation *op);
void setTiming(Operation *op, TimingAttr timing);
void setTiming(Operation *op, int64_t begin, int64_t end, int64_t latency,
               int64_t interval);

/// Resource attribute utils.
ResourceAttr getResource(Operation *op);
void setResource(Operation *op, ResourceAttr resource);
void setResource(Operation *op, int64_t lut, int64_t dsp, int64_t bram);

/// Loop information attribute utils.
LoopInfoAttr getLoopInfo(Operation *op);
void setLoopInfo(Operation *op, LoopInfoAttr loopInfo);
void setLoopInfo(Operation *op, int64_t flattenTripCount, int64_t iterLatency,
                 int64_t minII);

/// Loop directive attribute utils.
LoopDirectiveAttr getLoopDirective(Operation *op);
void setLoopDirective(Operation *op, LoopDirectiveAttr loopDirective);
void setLoopDirective(Operation *op, bool pipeline, int64_t targetII,
                      bool dataflow, bool flatten);

/// Function directive attribute utils.
FuncDirectiveAttr getFuncDirective(Operation *op);
void setFuncDirective(Operation *op, FuncDirectiveAttr funcDirective);
void setFuncDirective(Operation *op, bool pipeline, int64_t targetInterval,
                      bool dataflow);

/// Unit attribute utils for loop and function classification.
bool hasParallelAttr(Operation *op);
void setParallelAttr(Operation *op);
bool hasReductionAttr(Operation *op);
void setReductionAttr(Operation *op);
bool hasTopFuncAttr(Operation *op);
void setTopFuncAttr(Operation *op);
bool hasRuntimeAttr(Operation *op);
void setRuntimeAttr(Operation *op);

} // namespace dataflow
} // namespace sodap
} // namespace mlir

#define GET_OP_CLASSES
#include "sodap/Dialect/Dataflow/Dataflow.h.inc"

#endif // SODAP_DIALECT_DATAFLOW_DATAFLOW_H
