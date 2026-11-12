//===- TosaTransposes.cpp - Move and cancel TOSA transposes -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// sodap-tosa-sink-transposes and sodap-tosa-hoist-transposes: a model from
// PyTorch in TOSA's layouts (NHWC), with no transposes inside its blocks.
// sodap-tosa-raise-transposes, sodap-tosa-fold-constant-transposes and
// sodap-tosa-outline-kernel: one kernel split into a tile and a host that
// transposes its arguments and results.
//
//===----------------------------------------------------------------------===//

#include "sodap/Tosa/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"

#include <optional>

namespace mlir::sodap {
#define GEN_PASS_DEF_TOSASINKTRANSPOSES
#define GEN_PASS_DEF_TOSAHOISTTRANSPOSES
#define GEN_PASS_DEF_TOSARAISETRANSPOSES
#define GEN_PASS_DEF_TOSAFOLDCONSTANTTRANSPOSES
#define GEN_PASS_DEF_TOSAOUTLINEKERNEL
#include "sodap/Tosa/Passes.h.inc"

namespace {
/// The constant permutation of `transpose`, if any.
std::optional<SmallVector<int64_t>> permsOf(tosa::TransposeOp transpose) {
  SmallVector<int64_t> perms;
  if (failed(transpose.getConstantPerms(perms)))
    return std::nullopt;
  return perms;
}

/// `type` with its dims permuted by `perms`, as tosa.transpose permutes them.
RankedTensorType permuted(RankedTensorType type, ArrayRef<int64_t> perms) {
  SmallVector<int64_t> shape;
  for (int64_t d : perms)
    shape.push_back(type.getDimSize(d));
  return RankedTensorType::get(shape, type.getElementType());
}

SmallVector<int64_t> inverse(ArrayRef<int64_t> perms) {
  SmallVector<int64_t> result(perms.size());
  for (auto [i, d] : llvm::enumerate(perms))
    result[d] = i;
  return result;
}

/// `value` transposed by `perms`, a new tosa.const holding them.
Value transpose(OpBuilder &builder, Location loc, Value value,
                ArrayRef<int64_t> perms) {
  auto permsType = RankedTensorType::get({static_cast<int64_t>(perms.size())},
                                         builder.getI32Type());
  SmallVector<int32_t> values(perms.begin(), perms.end());
  Value permsValue = builder.create<tosa::ConstOp>(
      loc, permsType, DenseIntElementsAttr::get(permsType, values));
  return builder.create<tosa::TransposeOp>(
      loc, permuted(cast<RankedTensorType>(value.getType()), perms), value,
      permsValue);
}

/// A transpose of a non-constant value: one the pass moves.
tosa::TransposeOp movable(Value value) {
  auto transpose = value.getDefiningOp<tosa::TransposeOp>();
  if (!transpose || matchPattern(transpose.getInput1(), m_Constant()))
    return nullptr;
  return transpose;
}

/// elementwise(transpose(x), ...) -> transpose(elementwise(x, ...)).
struct SinkTranspose : RewritePattern {
  SinkTranspose(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa_and_nonnull<tosa::TosaDialect>(op->getDialect()) ||
        !op->hasTrait<OpTrait::ResultsBroadcastableShape>() ||
        op->getNumResults() != 1)
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return failure();

    // The permutation, the same for every transposed operand.
    std::optional<SmallVector<int64_t>> perms;
    Value permsValue;
    for (Value operand : op->getOperands()) {
      tosa::TransposeOp transpose = movable(operand);
      if (!transpose)
        continue;
      std::optional<SmallVector<int64_t>> these = permsOf(transpose);
      if (!these || (perms && *perms != *these))
        return failure();
      perms = these;
      permsValue = transpose.getPerms();
    }
    if (!perms)
      return failure();
    // The others: constants of the same rank, transposed back.
    SmallVector<int64_t> back = inverse(*perms);
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      if (tosa::TransposeOp transpose = movable(operand)) {
        operands.push_back(transpose.getInput1());
        continue;
      }
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (!type || type.getRank() != resultType.getRank() ||
          !matchPattern(operand, m_Constant()))
        return failure();
      operands.push_back(transpose(rewriter, op->getLoc(), operand, back));
    }

    Operation *sunk = rewriter.clone(*op);
    sunk->setOperands(operands);
    sunk->getResult(0).setType(permuted(resultType, back));
    rewriter.replaceOpWithNewOp<tosa::TransposeOp>(
        op, resultType, sunk->getResult(0), permsValue);
    return success();
  }
};

struct TosaSinkTransposes : impl::TosaSinkTransposesBase<TosaSinkTransposes> {
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<SinkTranspose>(&getContext());
    tosa::TransposeOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct TosaHoistTransposes
    : impl::TosaHoistTransposesBase<TosaHoistTransposes> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    for (auto func : llvm::to_vector(module.getOps<func::FuncOp>())) {
      if (!func.isPrivate() || func.isExternal())
        continue;
      std::optional<SymbolTable::UseRange> uses =
          SymbolTable::getSymbolUses(func, module);
      if (!uses || llvm::any_of(*uses, [](const SymbolTable::SymbolUse &use) {
            return !isa<func::CallOp>(use.getUser());
          }))
        continue;

      // Arguments read by transposes with one permutation. Other readers get
      // the inverse transpose, in the function: a residual add then reads two
      // values transposed alike, which sodap-tosa-sink-transposes moves on.
      // ponytail: hoists on any transposed read; weigh the reads if a block
      // gets more transposes than it loses.
      SmallVector<std::optional<SmallVector<int64_t>>> argPerms;
      for (BlockArgument arg : func.getArguments()) {
        std::optional<SmallVector<int64_t>> perms;
        bool hoist = true;
        for (Operation *user : arg.getUsers()) {
          auto transpose = dyn_cast<tosa::TransposeOp>(user);
          if (!transpose || transpose.getInput1() != arg)
            continue;
          std::optional<SmallVector<int64_t>> these = permsOf(transpose);
          if (!these || (perms && *perms != *these))
            hoist = false;
          else
            perms = these;
        }
        argPerms.push_back(hoist ? perms : std::nullopt);
      }
      // Results returned straight from a transpose.
      auto ret = cast<func::ReturnOp>(func.getBody().back().getTerminator());
      SmallVector<std::optional<SmallVector<int64_t>>> resultPerms;
      for (Value result : ret.getOperands()) {
        auto transpose = result.getDefiningOp<tosa::TransposeOp>();
        resultPerms.push_back(transpose ? permsOf(transpose) : std::nullopt);
      }
      if (llvm::none_of(argPerms, [](auto &p) { return p.has_value(); }) &&
          llvm::none_of(resultPerms, [](auto &p) { return p.has_value(); }))
        continue;

      // The function.
      OpBuilder builder(func.getContext());
      for (auto [arg, perms] : llvm::zip_equal(func.getArguments(), argPerms)) {
        if (!perms)
          continue;
        Type original = arg.getType();
        arg.setType(permuted(cast<RankedTensorType>(original), *perms));
        builder.setInsertionPointToStart(&func.front());
        SmallVector<OpOperand *> argUses;
        for (OpOperand &use : arg.getUses())
          argUses.push_back(&use);
        Value back;
        for (OpOperand *use : argUses) {
          auto read = dyn_cast<tosa::TransposeOp>(use->getOwner());
          if (read && read.getInput1() == arg) {
            read.getResult().replaceAllUsesWith(arg);
            read.erase();
            continue;
          }
          if (!back)
            back = transpose(builder, func.getLoc(), arg, inverse(*perms));
          use->set(back);
        }
      }
      for (auto [operand, perms] :
           llvm::zip_equal(ret->getOpOperands(), resultPerms))
        if (perms)
          operand.set(
              operand.get().getDefiningOp<tosa::TransposeOp>().getInput1());
      func.setType(builder.getFunctionType(func.front().getArgumentTypes(),
                                           ret.getOperandTypes()));

      // The callers.
      for (const SymbolTable::SymbolUse &use : *uses) {
        auto call = cast<func::CallOp>(use.getUser());
        builder.setInsertionPoint(call);
        SmallVector<Value> operands;
        for (auto [operand, perms] :
             llvm::zip_equal(call.getOperands(), argPerms))
          operands.push_back(
              perms ? transpose(builder, call.getLoc(), operand, *perms)
                    : operand);
        auto hoisted =
            builder.create<func::CallOp>(call.getLoc(), func, operands);
        for (auto [old, result, perms] : llvm::zip_equal(
                 call.getResults(), hoisted.getResults(), resultPerms))
          old.replaceAllUsesWith(
              perms ? transpose(builder, call.getLoc(), result, *perms)
                    : result);
        call.erase();
      }
    }
  }
};

/// transpose(op(x, c...)) -> op(transpose(x), transpose(c)...), `op`
/// elementwise, read only by the transpose, with at most one operand that is
/// not a tosa.const.
struct RaiseTranspose : OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp transpose,
                                PatternRewriter &rewriter) const override {
    Operation *op = transpose.getInput1().getDefiningOp();
    if (!op || !isa<tosa::TosaDialect>(op->getDialect()) ||
        !op->hasTrait<OpTrait::ResultsBroadcastableShape>() ||
        op->getNumResults() != 1 || !op->hasOneUse())
      return failure();
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    if (llvm::count_if(op->getOperands(), [](Value v) {
          return !v.getDefiningOp<tosa::ConstOp>();
        }) > 1)
      return failure();
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (!type || type.getRank() != resultType.getRank())
        return failure();
      operands.push_back(operand);
    }
    std::optional<SmallVector<int64_t>> perms = permsOf(transpose);
    if (!perms)
      return failure();

    for (Value &operand : operands)
      operand = rewriter.create<tosa::TransposeOp>(
          transpose.getLoc(),
          permuted(cast<RankedTensorType>(operand.getType()), *perms), operand,
          transpose.getPerms());
    Operation *raised = rewriter.clone(*op);
    raised->setOperands(operands);
    raised->getResult(0).setType(transpose.getType());
    rewriter.replaceOp(transpose, raised->getResults());
    rewriter.eraseOp(op);
    return success();
  }
};

/// The transposes in `root`, in order.
SmallVector<Operation *> transposesIn(Operation *root) {
  SmallVector<Operation *> transposes;
  root->walk([&](tosa::TransposeOp t) { transposes.push_back(t); });
  return transposes;
}

struct TosaRaiseTransposes
    : impl::TosaRaiseTransposesBase<TosaRaiseTransposes> {
  void runOnOperation() final {
    SmallVector<Operation *> transposes = transposesIn(getOperation());
    if (transposes.empty())
      return;
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseTranspose>(&getContext());
    tosa::TransposeOp::getCanonicalizationPatterns(patterns, &getContext());
    // Only the transposes and what the patterns create: the rest of the
    // function keeps its form.
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyOpPatternsAndFold(transposes, std::move(patterns), config)))
      signalPassFailure();
  }
};

struct TosaFoldConstantTransposes
    : impl::TosaFoldConstantTransposesBase<TosaFoldConstantTransposes> {
  void runOnOperation() final {
    SmallVector<Operation *> transposes;
    SmallVector<Operation *> constants;
    WalkResult walk = getOperation()->walk([&](tosa::TransposeOp t) {
      auto constant = t.getInput1().getDefiningOp<tosa::ConstOp>();
      if (!constant)
        return WalkResult::advance();
      if (!constant->hasOneUse()) {
        t.emitError("transpose of a constant that has other uses: folding "
                    "it would keep the data in both layouts");
        return WalkResult::interrupt();
      }
      transposes.push_back(t);
      constants.push_back(constant);
      return WalkResult::advance();
    });
    if (walk.wasInterrupted())
      return signalPassFailure();
    if (transposes.empty())
      return;
    RewritePatternSet patterns(&getContext());
    tosa::populateTosaFoldConstantTransposePatterns(&getContext(), patterns);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;
    bool allErased = false;
    if (failed(applyOpPatternsAndFold(transposes, std::move(patterns), config,
                                      /*changed=*/nullptr, &allErased)) ||
        !allErased) {
      getOperation()->emitError("could not fold every transpose of a constant");
      return signalPassFailure();
    }
    for (Operation *constant : constants)
      if (constant->use_empty())
        constant->erase();
  }
};

struct TosaOutlineKernel : impl::TosaOutlineKernelBase<TosaOutlineKernel> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    SmallVector<func::FuncOp> publics;
    for (auto f : module.getOps<func::FuncOp>())
      if (f.isPublic() && !f.isExternal())
        publics.push_back(f);
    if (publics.size() != 1) {
      module.emitError("expected one public function with a body, got ")
          << publics.size();
      return signalPassFailure();
    }
    func::FuncOp kernel = publics.front();
    if (!kernel.getBody().hasOneBlock()) {
      kernel.emitError("expected a function with one block");
      return signalPassFailure();
    }
    Block &body = kernel.front();
    auto ret = cast<func::ReturnOp>(body.getTerminator());

    // The host's transposes: of an argument, or of a returned value.
    DenseMap<BlockArgument, SmallVector<tosa::TransposeOp>> inputs;
    SmallVector<tosa::TransposeOp> outputs;
    for (Operation *op : transposesIn(kernel)) {
      auto t = cast<tosa::TransposeOp>(op);
      auto arg = dyn_cast<BlockArgument>(t.getInput1());
      if (arg && arg.getOwner() == &body) {
        inputs[arg].push_back(t);
        continue;
      }
      if (t->hasOneUse() && *t->user_begin() == ret.getOperation() &&
          t->getBlock() == &body) {
        outputs.push_back(t);
        continue;
      }
      t.emitError("transpose between two ops: the accelerator must not "
                  "transpose, only the host transposes the arguments and "
                  "results of @")
          << kernel.getSymName();
      return signalPassFailure();
    }

    // The kernel's arguments: each argument if the computation reads it,
    // then its transposes.
    SmallVector<Value> tileInputs;
    for (BlockArgument arg : body.getArguments()) {
      auto it = inputs.find(arg);
      size_t transposed = it == inputs.end() ? 0 : it->second.size();
      if (std::distance(arg.use_begin(), arg.use_end()) > (long)transposed)
        tileInputs.push_back(arg);
      if (transposed)
        for (tosa::TransposeOp t : it->second)
          tileInputs.push_back(t);
    }

    // The host: the transposes of its arguments, the launch, the transposes
    // of its results.
    OpBuilder builder(ctx);
    builder.setInsertionPointAfter(kernel);
    std::string name = "forward_kernel";
    auto host = builder.create<func::FuncOp>(kernel.getLoc(),
                                             kernel.getSymName(),
                                             kernel.getFunctionType());
    Block *hostBody = host.addEntryBlock();
    builder.setInsertionPointToStart(hostBody);
    IRMapping map;
    map.map(body.getArguments(), hostBody->getArguments());
    auto copy = [&](tosa::TransposeOp t, Value input) {
      Operation *perms = t.getPerms().getDefiningOp();
      Value p = perms ? builder.clone(*perms)->getResult(0) : Value(t.getPerms());
      return builder.create<tosa::TransposeOp>(t.getLoc(), t.getType(), input,
                                               p);
    };
    SmallVector<Value> launchInputs;
    for (Value v : tileInputs) {
      if (auto t = v.getDefiningOp<tosa::TransposeOp>())
        launchInputs.push_back(copy(t, map.lookup(t.getInput1())));
      else
        launchInputs.push_back(map.lookup(v));
    }
    SmallVector<Value> tileResults;
    for (Value v : ret.getOperands()) {
      auto t = v.getDefiningOp<tosa::TransposeOp>();
      tileResults.push_back(t && llvm::is_contained(outputs, t) ? t.getInput1()
                                                                : v);
    }
    auto kernelRef = SymbolRefAttr::get(ctx, name,
                                        {FlatSymbolRefAttr::get(ctx, name)});
    auto launch = builder.create<dataflow::LaunchOp>(
        kernel.getLoc(), ValueRange(tileResults).getTypes(), kernelRef,
        launchInputs, ValueRange());
    SmallVector<Value> hostResults;
    for (auto [v, r] : llvm::zip_equal(ret.getOperands(), launch.getResults())) {
      auto t = v.getDefiningOp<tosa::TransposeOp>();
      hostResults.push_back(t && llvm::is_contained(outputs, t) ? Value(copy(t, r))
                                                                : r);
    }
    builder.create<func::ReturnOp>(kernel.getLoc(), hostResults);

    // The tile: the kernel without the host's transposes.
    ret->setOperands(tileResults);
    SetVector<Operation *> perms;
    for (tosa::TransposeOp t : outputs) {
      if (Operation *p = t.getPerms().getDefiningOp())
        perms.insert(p);
      t.erase();
    }
    auto *tileBody = new Block();
    for (Value v : tileInputs)
      v.replaceAllUsesWith(tileBody->addArgument(v.getType(), v.getLoc()));
    for (auto &[arg, ts] : inputs)
      for (tosa::TransposeOp t : ts) {
        if (Operation *p = t.getPerms().getDefiningOp())
          perms.insert(p);
        t.erase();
      }
    tileBody->getOperations().splice(tileBody->end(), body.getOperations());
    kernel.getBody().push_back(tileBody);
    body.erase();
    for (Operation *p : perms)
      if (p->use_empty())
        p->erase();
    kernel.setSymName(name);
    kernel.setType(FunctionType::get(ctx, tileBody->getArgumentTypes(),
                                     ret.getOperandTypes()));

    builder.setInsertionPoint(kernel);
    auto tile = builder.create<ModuleOp>(kernel.getLoc(), name);
    tile->setAttr("sodap.tile", builder.getI64IntegerAttr(0));
    tile->setAttr("sodap.tile_dir", builder.getStringAttr("tiles/" + name));
    kernel->moveBefore(tile.getBody(), tile.getBody()->end());
  }
};
} // namespace

} // namespace mlir::sodap
