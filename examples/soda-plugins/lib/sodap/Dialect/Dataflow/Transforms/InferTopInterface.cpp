//===- InferTopInterface.cpp - External streams of the top to axis --------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Finds which stream arguments of the top are the design's external
// interface and makes them axis ports.
//
// A top stream argument is followed through the calls it is passed to, down
// to the stream ops that use it. If some stage reads it, the outside world
// writes it; if some stage writes it, the outside world reads it. Either way
// it is a port, and its type -- in the top and in every function on the way --
// becomes !dataflow.stream<T, D, axis>. Retyping the whole chain is what keeps
// the calls valid: a call operand and the callee argument must agree.
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_INFERTOPINTERFACE
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

#define DEBUG_TYPE "sodap-dataflow-infer-top-interface"

using namespace mlir;
using namespace mlir::sodap;
using namespace mlir::sodap::dataflow;

namespace {

/// How the stages use one stream argument of the top.
struct StreamUse {
  /// First read (or peek) and first write found, for diagnostics.
  Operation *read = nullptr;
  Operation *write = nullptr;
  /// Every block argument the stream reaches, the top's own included.
  llvm::SetVector<BlockArgument> args;
};

/// Follows `root` through calls down to the stream ops that use it.
static LogicalResult traceStream(BlockArgument root, SymbolTable &symbols,
                                 StreamUse &use) {
  SmallVector<BlockArgument> worklist{root};
  use.args.insert(root);

  while (!worklist.empty()) {
    BlockArgument arg = worklist.pop_back_val();
    for (OpOperand &operand : arg.getUses()) {
      Operation *user = operand.getOwner();

      if (isa<StreamReadOp, StreamPeekOp>(user)) {
        if (!use.read)
          use.read = user;
        continue;
      }
      if (auto write = dyn_cast<StreamWriteOp>(user);
          write && operand.get() == write.getChannel()) {
        if (!use.write)
          use.write = user;
        continue;
      }
      if (auto call = dyn_cast<func::CallOp>(user)) {
        auto callee = symbols.lookup<func::FuncOp>(call.getCallee());
        if (!callee || callee.isExternal()) {
          return call.emitError()
                 << "passes a stream of the top to '" << call.getCallee()
                 << "', which has no body to tell how it is used";
        }
        auto calleeArg = callee.getArgument(operand.getOperandNumber());
        // insert() is false on a function seen before: recursion, or the
        // same stream passed twice along one path.
        if (use.args.insert(calleeArg))
          worklist.push_back(calleeArg);
        continue;
      }

      return user->emitError()
             << "uses a stream of the top in a way the interface inference "
                "does not understand";
    }
  }
  return success();
}

struct InferTopInterface
    : public dataflow::impl::InferTopInterfaceBase<InferTopInterface> {
  using InferTopInterfaceBase::InferTopInterfaceBase;

  void runOnOperation() override;
};

void InferTopInterface::runOnOperation() {
  auto module = getOperation();

  // Same story as EmitBambuArchitecture: a pass option cannot be required,
  // so a missing one shows up here as empty.
  if (topFunc.empty()) {
    module.emitError() << "option 'top-func' is required";
    return signalPassFailure();
  }
  SymbolTable symbols(module);
  auto top = symbols.lookup<func::FuncOp>(topFunc);
  if (!top || top.isExternal()) {
    module.emitError() << "top function '" << topFunc << "' not found";
    return signalPassFailure();
  }

  // Every argument to retype, the top's and those of the functions its
  // ports are passed to.
  llvm::SetVector<BlockArgument> ports;
  for (auto arg : top.getArguments()) {
    if (!isa<StreamType>(arg.getType()))
      continue;

    StreamUse use;
    if (failed(traceStream(arg, symbols, use)))
      return signalPassFailure();

    if (use.read && use.write) {
      auto diag = top.emitError()
                  << "stream argument " << arg.getArgNumber()
                  << " is both read and written, but an axis port carries "
                     "data in one direction only";
      diag.attachNote(use.read->getLoc()) << "read here";
      diag.attachNote(use.write->getLoc()) << "written here";
      return signalPassFailure();
    }
    if (!use.read && !use.write) {
      LLVM_DEBUG(llvm::dbgs() << "argument " << arg.getArgNumber()
                              << ": unused, left as is\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "argument " << arg.getArgNumber() << ": "
               << (use.read ? "read" : "written") << " by the design, "
               << "an axis port reaching " << use.args.size()
               << " argument(s)\n");
    ports.insert(use.args.begin(), use.args.end());
  }

  // Retype the ports, then the signatures that hold them.
  llvm::SetVector<func::FuncOp> retyped;
  for (auto arg : ports) {
    arg.setType(
        cast<StreamType>(arg.getType()).withInterface(StreamInterface::AXIS));
    retyped.insert(cast<func::FuncOp>(arg.getOwner()->getParentOp()));
  }
  for (auto func : retyped) {
    func.setType(FunctionType::get(&getContext(),
                                   func.front().getArgumentTypes(),
                                   func.getFunctionType().getResults()));
  }

  // A function whose argument became a port has to receive a port at every
  // call. One that also gets a fifo somewhere -- a stage shared between an
  // external stream and an internal channel -- cannot have both signatures.
  auto result = module.walk([&](func::CallOp call) {
    auto callee = symbols.lookup<func::FuncOp>(call.getCallee());
    if (!callee || !retyped.contains(callee))
      return WalkResult::advance();
    for (auto [index, types] : llvm::enumerate(llvm::zip_equal(
             call.getOperandTypes(), callee.getArgumentTypes()))) {
      auto [operandType, argType] = types;
      if (operandType == argType)
        continue;
      call.emitError() << "passes " << operandType << " as argument " << index
                       << " of '" << callee.getName()
                       << "', which is an axis port at another call; give "
                          "each use its own function";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    signalPassFailure();
}

} // namespace
