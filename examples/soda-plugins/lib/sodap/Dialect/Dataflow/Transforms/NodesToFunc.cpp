//===- NodesToFunc.cpp - Outline dataflow nodes into functions ------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Outline each node into its own function and unwrap the enclosing dispatch,
// flattening the dataflow hierarchy into the call-graph form an HLS backend
// consumes.
//
// This is a plain ordered walk rather than a greedy rewrite: the greedy driver
// visits operations bottom-up, which would make the generated node names
// depend on traversal order instead of program order.
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_NODESTOFUNC
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

#define DEBUG_TYPE "sodap-dataflow-nodes-to-func"

using namespace mlir;
using namespace mlir::sodap;
using namespace mlir::sodap::dataflow;

namespace mlir {
namespace sodap {
namespace dataflow {
namespace {

/// Outline `node` into a standalone function placed just before `parent`, and
/// replace the node with a call to it.
void outlineNode(NodeOp node, func::FuncOp parent, unsigned index,
                 OpBuilder &builder) {
  builder.setInsertionPoint(parent);
  auto name = (parent.getName() + "_node" + Twine(index)).str();
  auto subFunc = builder.create<func::FuncOp>(
      node.getLoc(), name,
      builder.getFunctionType(node.getOperandTypes(), TypeRange()));

  // A node is IsolatedFromAbove and its ports are already block arguments, so
  // its body can move wholesale into the new function.
  subFunc.getBody().takeBody(node.getBodyRegion());
  builder.setInsertionPointToEnd(&subFunc.front());
  builder.create<func::ReturnOp>(node.getLoc());

  builder.setInsertionPoint(node);
  builder.create<func::CallOp>(node.getLoc(), subFunc, node.getOperands());
  node.erase();
}

/// Splice the contents of `dispatch` into its parent block and remove it.
void unwrapDispatch(DispatchOp dispatch) {
  auto yield = dispatch.getYieldOp();
  dispatch.replaceAllUsesWith(yield.getOperands());

  auto &dispatchOps = dispatch.getBody().front().getOperations();
  auto &parentOps = dispatch->getBlock()->getOperations();
  parentOps.splice(dispatch->getIterator(), dispatchOps, dispatchOps.begin(),
                   std::prev(dispatchOps.end()));
  dispatch.erase();
}

struct NodesToFunc : public impl::NodesToFuncBase<NodesToFunc> {
  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module.getContext());

    // Collect first: outlining inserts new functions into the module, which
    // would otherwise perturb an in-flight walk.
    SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>());

    for (auto func : funcs) {
      if (func.isExternal())
        continue;

      SmallVector<DispatchOp> dispatches;
      func.walk([&](DispatchOp dispatch) { dispatches.push_back(dispatch); });

      for (auto dispatch : dispatches) {
        unsigned index = 0;
        SmallVector<NodeOp> nodes(dispatch.getOps<NodeOp>());
        for (auto node : nodes)
          outlineNode(node, func, index++, builder);
        unwrapDispatch(dispatch);
      }
    }
  }
};

} // namespace
} // namespace dataflow
} // namespace sodap
} // namespace mlir
