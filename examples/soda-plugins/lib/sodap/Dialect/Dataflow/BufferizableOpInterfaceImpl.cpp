//===- BufferizableOpInterfaceImpl.cpp - Bufferize dataflow ops -----------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One-shot-bufferize through dispatch, task and yield, the way upstream does
// scf.execute_region and its scf.yield (LLVM 19.1.5): a result is the buffer
// of the value its op yields. Which buffer that is, and where it is allocated,
// is decided by the ops inside: an edge allocated in the dispatch
// (`bufferization.alloc_tensor`) stays there.
//
// The analysis assumes the ops run in program order, which tasks don't. That
// is sound as long as no task writes a buffer another task reads or writes,
// which the task verifier enforces on tensors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "sodap/Dialect/Dataflow/Dataflow.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::sodap::dataflow;

namespace {
/// dispatch and task: no tensor operands; each result is equivalent to the
/// value yielded for it.
template <typename OpTy>
struct RegionOpInterface
    : public BufferizableOpInterface::ExternalModel<RegionOpInterface<OpTy>,
                                                    OpTy> {
  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    return true;
  }

  AliasingOpOperandList
  getAliasingOpOperands(Operation *op, Value value,
                        const AnalysisState &state) const {
    unsigned resultNum = cast<OpResult>(value).getResultNumber();
    return {{&cast<OpTy>(op).getYieldOp()->getOpOperand(resultNum),
             BufferRelation::Equivalent}};
  }

  // The yield is bufferized first (ops go in post-order), so it already
  // yields memrefs: they are the new result types.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto oldOp = cast<OpTy>(op);
    auto newOp = rewriter.create<OpTy>(
        op->getLoc(), TypeRange(oldOp.getYieldOp()->getOperandTypes()));
    newOp->setAttrs(op->getAttrs()); // The role: compute, load, store.
    newOp.getBody().takeBody(oldOp.getBody());
    replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());
    return success();
  }
};

struct YieldOpInterface
    : public BufferizableOpInterface::ExternalModel<YieldOpInterface,
                                                    YieldOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getParentOp()->getResult(opOperand.getOperandNumber()),
             BufferRelation::Equivalent}};
  }

  // An out-of-place yield would allocate and copy inside the region.
  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    SmallVector<Value> results;
    for (Value value : op->getOperands()) {
      if (!isa<TensorType>(value.getType())) {
        results.push_back(value);
        continue;
      }
      FailureOr<Value> buffer = getBuffer(rewriter, value, options);
      if (failed(buffer))
        return failure();
      results.push_back(*buffer);
    }
    replaceOpWithNewBufferizedOp<YieldOp>(rewriter, op, results);
    return success();
  }
};
/// launch: the kernel reads its tensor operands, and each result is a new
/// buffer, allocated before the launch and passed in `outs`.
struct LaunchOpInterface
    : public BufferizableOpInterface::ExternalModel<LaunchOpInterface,
                                                    LaunchOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  bool bufferizesToAllocation(Operation *op, Value value) const {
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto launch = cast<LaunchOp>(op);
    SmallVector<Value> inputs, outs;
    for (Value value : launch.getInputs()) {
      if (!isa<TensorType>(value.getType())) {
        inputs.push_back(value);
        continue;
      }
      FailureOr<Value> buffer = getBuffer(rewriter, value, options);
      if (failed(buffer))
        return failure();
      inputs.push_back(*buffer);
    }
    for (Value result : launch.getResults()) {
      auto type = dyn_cast<RankedTensorType>(result.getType());
      if (!type || !type.hasStaticShape())
        return op->emitError("expected results of static shapes, got ")
               << result.getType();
      FailureOr<Value> alloc = options.createAlloc(
          rewriter, op->getLoc(),
          MemRefType::get(type.getShape(), type.getElementType()), {});
      if (failed(alloc))
        return failure();
      outs.push_back(*alloc);
    }
    auto newOp = rewriter.create<LaunchOp>(op->getLoc(), TypeRange(),
                                           launch.getKernel(), inputs, outs);
    newOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    replaceOpWithBufferizedValues(rewriter, op, outs);
    return success();
  }
};
} // namespace

void mlir::sodap::dataflow::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, DataflowDialect *dialect) {
    DispatchOp::attachInterface<RegionOpInterface<DispatchOp>>(*ctx);
    TaskOp::attachInterface<RegionOpInterface<TaskOp>>(*ctx);
    YieldOp::attachInterface<YieldOpInterface>(*ctx);
    LaunchOp::attachInterface<LaunchOpInterface>(*ctx);
  });
}
