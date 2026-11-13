//===- BindDims.cpp - Dimensions from index arguments ---------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_BINDDIMS
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

using namespace mlir;

namespace {
struct BindDims : public sodap::dataflow::impl::BindDimsBase<BindDims> {
  void runOnOperation() final {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;
    auto bind = [&](Operation *op, Value source, OpFoldResult index) {
      auto arg = dyn_cast<BlockArgument>(source);
      std::optional<int64_t> d = getConstantIntValue(index);
      if (!arg || arg.getOwner() != &func.front() || !d)
        return;
      auto dims = func.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(),
                                                           "sodap.dims");
      if (!dims || *d >= dims.size())
        return;
      op->getResult(0).replaceAllUsesWith(func.getArgument(dims[*d]));
      op->erase();
    };
    func.walk([&](Operation *op) {
      if (auto dim = dyn_cast<tensor::DimOp>(op))
        bind(op, dim.getSource(), dim.getIndex());
      else if (auto dim = dyn_cast<memref::DimOp>(op))
        bind(op, dim.getSource(), dim.getIndex());
    });
  }
};
} // namespace
