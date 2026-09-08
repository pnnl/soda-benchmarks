//===- Dataflow.cpp - Dataflow dialect implementation ---------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Dataflow.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::sodap;
using namespace mlir::sodap::dataflow;

#include "sodap/Dialect/Dataflow/DataflowInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Dataflow dialect
//===----------------------------------------------------------------------===//

void DataflowDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "sodap/Dialect/Dataflow/DataflowTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "sodap/Dialect/Dataflow/DataflowAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "sodap/Dialect/Dataflow/Dataflow.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// StreamOp, StreamReadOp, and StreamWriteOp
//===----------------------------------------------------------------------===//

LogicalResult StreamOp::verify() {
  if (getDepth() != llvm::cast<StreamType>(getChannel().getType()).getDepth())
    return emitOpError("stream channel depth is not aligned");
  return success();
}

void StreamOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       llvm::cast<OpResult>(getChannel()),
                       SideEffects::DefaultResource::get());
}

LogicalResult StreamReadOp::verify() {
  if (getResult())
    if (llvm::cast<StreamType>(getChannel().getType()).getElementType() !=
        getResult().getType())
      return emitOpError("result type doesn't align with channel type");
  return success();
}

LogicalResult StreamWriteOp::verify() {
  if (llvm::cast<StreamType>(getChannel().getType()).getElementType() !=
      getValue().getType())
    return emitOpError("value type doesn't align with channel type");
  return success();
}

//===----------------------------------------------------------------------===//
// Shared DispatchOp/TaskOp canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
/// Drop results of a dispatch or task op that nothing uses.
template <typename OpType>
struct SimplifyDispatchOrTaskOutputs : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto yield = op.getYieldOp();
    bool hasUnusedPort = false;

    // Identify output values that are used.
    SmallVector<Value, 4> usedOutputs;
    SmallVector<Value, 4> usedResults;
    for (auto result : op.getResults())
      if (result.use_empty()) {
        hasUnusedPort = true;
      } else {
        usedOutputs.push_back(yield.getOperand(result.getResultNumber()));
        usedResults.push_back(result);
      }

    if (!hasUnusedPort)
      return failure();

    // Construct a new op with only the used outputs.
    rewriter.setInsertionPoint(yield);
    rewriter.replaceOpWithNewOp<YieldOp>(yield, usedOutputs);

    rewriter.setInsertionPoint(op);
    auto newOp = rewriter.create<OpType>(op.getLoc(), ValueRange(usedOutputs));
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    for (auto t : llvm::zip(usedResults, newOp.getResults()))
      std::get<0>(t).replaceAllUsesWith(std::get<1>(t));

    rewriter.eraseOp(op);
    return success();
  }
};

/// Inline a dispatch or task op into its parent when `condition` holds.
///
/// `condition` is a plain function pointer rather than a `llvm::function_ref`
/// on purpose: a `function_ref` member would dangle, since the lambda passed
/// at pattern-construction time is a temporary that dies before the pattern
/// ever runs.
template <typename OpType>
struct InlineDispatchOrTask : public OpRewritePattern<OpType> {
  InlineDispatchOrTask(MLIRContext *context, bool (*condition)(OpType))
      : OpRewritePattern<OpType>(context), condition(condition) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!condition(op))
      return failure();

    auto &ops = op.getBody().front().getOperations();
    auto &parentOps = op->getBlock()->getOperations();
    parentOps.splice(op->getIterator(), ops, ops.begin(), std::prev(ops.end()));
    rewriter.replaceOp(op, op.getYieldOp()->getOperands());
    return success();
  }

private:
  bool (*condition)(OpType);
};
} // namespace

//===----------------------------------------------------------------------===//
// DispatchOp
//===----------------------------------------------------------------------===//

void DispatchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<SimplifyDispatchOrTaskOutputs<DispatchOp>>(context);
  results.add<InlineDispatchOrTask<DispatchOp>>(context, [](DispatchOp op) {
    return op.getOps<TaskOp>().empty() || llvm::hasSingleElement(op.getOps());
  });
}

LogicalResult DispatchOp::verify() {
  if (getResultTypes() != getYieldOp().getOperandTypes())
    return emitOpError("yield type doesn't align with result type");
  return success();
}

YieldOp DispatchOp::getYieldOp() {
  return cast<YieldOp>(getBody().front().getTerminator());
}

//===----------------------------------------------------------------------===//
// TaskOp
//===----------------------------------------------------------------------===//

void TaskOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<SimplifyDispatchOrTaskOutputs<TaskOp>>(context);
  results.add<InlineDispatchOrTask<TaskOp>>(
      context, [](TaskOp op) { return llvm::hasSingleElement(op.getOps()); });
}

LogicalResult TaskOp::verify() {
  if (getResultTypes() != getYieldOp().getOperandTypes())
    return emitOpError("yield type doesn't align with result type");
  return success();
}

DispatchOp TaskOp::getDispatchOp() {
  return (*this)->getParentOfType<DispatchOp>();
}

YieldOp TaskOp::getYieldOp() {
  return cast<YieldOp>(getBody().front().getTerminator());
}

bool TaskOp::isLivein(Value value) {
  auto liveins = Liveness(*this).getLiveIn(&(*this).getBody().front());
  return liveins.count(value);
}

SmallVector<Value> TaskOp::getLiveins() {
  auto liveins = Liveness(*this).getLiveIn(&(*this).getBody().front());
  return {liveins.begin(), liveins.end()};
}

SmallVector<Operation *> TaskOp::getLiveinUsers(Value livein) {
  assert(isLivein(livein) && "invalid livein");
  auto users = llvm::make_filter_range(livein.getUsers(), [&](Operation *user) {
    return (*this)->isAncestor(user);
  });
  return {users.begin(), users.end()};
}

//===----------------------------------------------------------------------===//
// NodeOp
//===----------------------------------------------------------------------===//

void NodeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto &operand : getInputsMutable())
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
  for (auto &operand : getOutputsMutable()) {
    effects.emplace_back(MemoryEffects::Read::get(), &operand,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), &operand,
                         SideEffects::DefaultResource::get());
  }
}

DispatchOp NodeOp::getDispatchOp() {
  return (*this)->getParentOfType<DispatchOp>();
}

void NodeOp::setInputTap(unsigned idx, unsigned tap) {
  SmallVector<int32_t> newInputTaps(llvm::map_range(
      getInputTapsAsInt(), [](unsigned a) { return (int32_t)a; }));
  newInputTaps[idx] = tap;
  Builder builder(getContext());
  setInputTapsAttr(builder.getI32ArrayAttr(newInputTaps));
}

unsigned NodeOp::getInputTap(unsigned idx) {
  return llvm::cast<IntegerAttr>(getInputTaps()[idx]).getInt();
}

SmallVector<unsigned> NodeOp::getInputTapsAsInt() {
  auto array = llvm::map_range(getInputTaps(), [](Attribute attr) {
    return llvm::cast<IntegerAttr>(attr).getInt();
  });
  return {array.begin(), array.end()};
}

unsigned NodeOp::getNumInputs() {
  return getODSOperandIndexAndLength(0).second;
}
unsigned NodeOp::getNumOutputs() {
  return getODSOperandIndexAndLength(1).second;
}
unsigned NodeOp::getNumParams() {
  return getODSOperandIndexAndLength(2).second;
}

OperandKind NodeOp::getOperandKind(OpOperand &operand) {
  assert(operand.getOwner() == *this && "invalid operand");
  return getOperandKind(operand.getOperandNumber());
}

OperandKind NodeOp::getOperandKind(unsigned operandIdx) {
  if (operandIdx >= getODSOperandIndexAndLength(2).first)
    return OperandKind::PARAM;
  if (operandIdx >= getODSOperandIndexAndLength(1).first)
    return OperandKind::OUTPUT;
  return OperandKind::INPUT;
}

llvm::iterator_range<Block::args_iterator> NodeOp::getInputArgs() {
  auto range = getODSOperandIndexAndLength(0);
  return {std::next(getBody().args_begin(), range.first),
          std::next(getBody().args_begin(), range.first + range.second)};
}
llvm::iterator_range<Block::args_iterator> NodeOp::getOutputArgs() {
  auto range = getODSOperandIndexAndLength(1);
  return {std::next(getBody().args_begin(), range.first),
          std::next(getBody().args_begin(), range.first + range.second)};
}
llvm::iterator_range<Block::args_iterator> NodeOp::getParamArgs() {
  auto range = getODSOperandIndexAndLength(2);
  return {std::next(getBody().args_begin(), range.first),
          std::next(getBody().args_begin(), range.first + range.second)};
}

bool NodeOp::isLivein(Value value) {
  return llvm::isa<BlockArgument>(value) &&
         value.getParentRegion() == &(*this).getBody();
}

SmallVector<Value> NodeOp::getLiveins() {
  auto args = (*this).getBody().getArguments();
  return {args.begin(), args.end()};
}

SmallVector<Operation *> NodeOp::getLiveinUsers(Value livein) {
  assert(isLivein(livein) && "invalid livein");
  auto users = livein.getUsers();
  return {users.begin(), users.end()};
}

//===----------------------------------------------------------------------===//
// Attribute helpers
//===----------------------------------------------------------------------===//

TimingAttr dataflow::getTiming(Operation *op) {
  return op->getAttrOfType<TimingAttr>(kTimingAttrName);
}
void dataflow::setTiming(Operation *op, TimingAttr timing) {
  op->setAttr(kTimingAttrName, timing);
}
void dataflow::setTiming(Operation *op, int64_t begin, int64_t end,
                         int64_t latency, int64_t interval) {
  setTiming(op,
            TimingAttr::get(op->getContext(), begin, end, latency, interval));
}

ResourceAttr dataflow::getResource(Operation *op) {
  return op->getAttrOfType<ResourceAttr>(kResourceAttrName);
}
void dataflow::setResource(Operation *op, ResourceAttr resource) {
  op->setAttr(kResourceAttrName, resource);
}
void dataflow::setResource(Operation *op, int64_t lut, int64_t dsp,
                           int64_t bram) {
  setResource(op, ResourceAttr::get(op->getContext(), lut, dsp, bram));
}

LoopInfoAttr dataflow::getLoopInfo(Operation *op) {
  return op->getAttrOfType<LoopInfoAttr>(kLoopInfoAttrName);
}
void dataflow::setLoopInfo(Operation *op, LoopInfoAttr loopInfo) {
  op->setAttr(kLoopInfoAttrName, loopInfo);
}
void dataflow::setLoopInfo(Operation *op, int64_t flattenTripCount,
                           int64_t iterLatency, int64_t minII) {
  setLoopInfo(op, LoopInfoAttr::get(op->getContext(), flattenTripCount,
                                    iterLatency, minII));
}

LoopDirectiveAttr dataflow::getLoopDirective(Operation *op) {
  return op->getAttrOfType<LoopDirectiveAttr>(kLoopDirectiveAttrName);
}
void dataflow::setLoopDirective(Operation *op,
                                LoopDirectiveAttr loopDirective) {
  op->setAttr(kLoopDirectiveAttrName, loopDirective);
}
void dataflow::setLoopDirective(Operation *op, bool pipeline, int64_t targetII,
                                bool dataflowAttr, bool flatten) {
  setLoopDirective(op, LoopDirectiveAttr::get(op->getContext(), pipeline,
                                              targetII, dataflowAttr, flatten));
}

FuncDirectiveAttr dataflow::getFuncDirective(Operation *op) {
  return op->getAttrOfType<FuncDirectiveAttr>(kFuncDirectiveAttrName);
}
void dataflow::setFuncDirective(Operation *op,
                                FuncDirectiveAttr funcDirective) {
  op->setAttr(kFuncDirectiveAttrName, funcDirective);
}
void dataflow::setFuncDirective(Operation *op, bool pipeline,
                                int64_t targetInterval, bool dataflowAttr) {
  setFuncDirective(op, FuncDirectiveAttr::get(op->getContext(), pipeline,
                                              targetInterval, dataflowAttr));
}

bool dataflow::hasParallelAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("parallel");
}
void dataflow::setParallelAttr(Operation *op) {
  op->setAttr("parallel", UnitAttr::get(op->getContext()));
}
bool dataflow::hasReductionAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("reduction");
}
void dataflow::setReductionAttr(Operation *op) {
  op->setAttr("reduction", UnitAttr::get(op->getContext()));
}
bool dataflow::hasTopFuncAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("top_func");
}
void dataflow::setTopFuncAttr(Operation *op) {
  op->setAttr("top_func", UnitAttr::get(op->getContext()));
}
bool dataflow::hasRuntimeAttr(Operation *op) {
  return op->hasAttrOfType<UnitAttr>("runtime");
}
void dataflow::setRuntimeAttr(Operation *op) {
  op->setAttr("runtime", UnitAttr::get(op->getContext()));
}

//===----------------------------------------------------------------------===//
// Attribute custom assembly formats
//===----------------------------------------------------------------------===//

namespace {
/// Parse a `true` / `false` keyword into `value`.
///
/// The printers below emit these same keywords rather than a raw bool, so that
/// the directive attributes round-trip. Anything that is neither keyword is
/// rejected instead of being silently read as false.
ParseResult parseBoolKeyword(AsmParser &p, bool &value) {
  auto loc = p.getCurrentLocation();
  StringRef keyword;
  if (succeeded(p.parseOptionalKeyword(&keyword))) {
    if (keyword == "true") {
      value = true;
      return success();
    }
    if (keyword == "false") {
      value = false;
      return success();
    }
  }
  return p.emitError(loc, "expected 'true' or 'false'");
}

/// Parse `<keyword>` and check it against the expected spelling.
ParseResult parseExpectedKeyword(AsmParser &p, StringRef expected) {
  StringRef keyword;
  auto loc = p.getCurrentLocation();
  if (p.parseKeyword(&keyword))
    return failure();
  if (keyword != expected)
    return p.emitError(loc, "expected '") << expected << "'";
  return success();
}
} // namespace

//===----------------------------------------------------------------------===//
// ResourceAttr
//===----------------------------------------------------------------------===//

Attribute ResourceAttr::parse(AsmParser &p, Type type) {
  int64_t lut, dsp, bram;
  if (p.parseLess() || parseExpectedKeyword(p, "lut") || p.parseEqual() ||
      p.parseInteger(lut) || p.parseComma() || parseExpectedKeyword(p, "dsp") ||
      p.parseEqual() || p.parseInteger(dsp) || p.parseComma() ||
      parseExpectedKeyword(p, "bram") || p.parseEqual() ||
      p.parseInteger(bram) || p.parseGreater())
    return Attribute();

  return ResourceAttr::get(p.getContext(), lut, dsp, bram);
}

void ResourceAttr::print(AsmPrinter &p) const {
  p << "<lut=" << getLut() << ", dsp=" << getDsp() << ", bram=" << getBram()
    << ">";
}

//===----------------------------------------------------------------------===//
// TimingAttr
//===----------------------------------------------------------------------===//

Attribute TimingAttr::parse(AsmParser &p, Type type) {
  int64_t begin, end, latency, interval;
  if (p.parseLess() || p.parseInteger(begin) || p.parseArrow() ||
      p.parseInteger(end) || p.parseComma() || p.parseInteger(latency) ||
      p.parseComma() || p.parseInteger(interval) || p.parseGreater())
    return Attribute();

  return TimingAttr::get(p.getContext(), begin, end, latency, interval);
}

void TimingAttr::print(AsmPrinter &p) const {
  p << "<" << getBegin() << " -> " << getEnd() << ", " << getLatency() << ", "
    << getInterval() << ">";
}

//===----------------------------------------------------------------------===//
// LoopInfoAttr
//===----------------------------------------------------------------------===//

Attribute LoopInfoAttr::parse(AsmParser &p, Type type) {
  int64_t flattenTripCount, iterLatency, minII;
  if (p.parseLess() || parseExpectedKeyword(p, "flattenTripCount") ||
      p.parseEqual() || p.parseInteger(flattenTripCount) || p.parseComma() ||
      parseExpectedKeyword(p, "iterLatency") || p.parseEqual() ||
      p.parseInteger(iterLatency) || p.parseComma() ||
      parseExpectedKeyword(p, "minII") || p.parseEqual() ||
      p.parseInteger(minII) || p.parseGreater())
    return Attribute();

  return LoopInfoAttr::get(p.getContext(), flattenTripCount, iterLatency,
                           minII);
}

void LoopInfoAttr::print(AsmPrinter &p) const {
  p << "<flattenTripCount=" << getFlattenTripCount()
    << ", iterLatency=" << getIterLatency() << ", minII=" << getMinII() << ">";
}

//===----------------------------------------------------------------------===//
// LoopDirectiveAttr
//===----------------------------------------------------------------------===//

Attribute LoopDirectiveAttr::parse(AsmParser &p, Type type) {
  bool pipeline, dataflowAttr, flatten;
  int64_t targetII;
  if (p.parseLess() || parseExpectedKeyword(p, "pipeline") || p.parseEqual() ||
      parseBoolKeyword(p, pipeline) || p.parseComma() ||
      parseExpectedKeyword(p, "targetII") || p.parseEqual() ||
      p.parseInteger(targetII) || p.parseComma() ||
      parseExpectedKeyword(p, "dataflow") || p.parseEqual() ||
      parseBoolKeyword(p, dataflowAttr) || p.parseComma() ||
      parseExpectedKeyword(p, "flatten") || p.parseEqual() ||
      parseBoolKeyword(p, flatten) || p.parseGreater())
    return Attribute();

  return LoopDirectiveAttr::get(p.getContext(), pipeline, targetII,
                                dataflowAttr, flatten);
}

void LoopDirectiveAttr::print(AsmPrinter &p) const {
  p << "<pipeline=" << (getPipeline() ? "true" : "false")
    << ", targetII=" << getTargetII()
    << ", dataflow=" << (getDataflow() ? "true" : "false")
    << ", flatten=" << (getFlatten() ? "true" : "false") << ">";
}

//===----------------------------------------------------------------------===//
// FuncDirectiveAttr
//===----------------------------------------------------------------------===//

Attribute FuncDirectiveAttr::parse(AsmParser &p, Type type) {
  bool pipeline, dataflowAttr;
  int64_t targetInterval;
  if (p.parseLess() || parseExpectedKeyword(p, "pipeline") || p.parseEqual() ||
      parseBoolKeyword(p, pipeline) || p.parseComma() ||
      parseExpectedKeyword(p, "targetInterval") || p.parseEqual() ||
      p.parseInteger(targetInterval) || p.parseComma() ||
      parseExpectedKeyword(p, "dataflow") || p.parseEqual() ||
      parseBoolKeyword(p, dataflowAttr) || p.parseGreater())
    return Attribute();

  return FuncDirectiveAttr::get(p.getContext(), pipeline, targetInterval,
                                dataflowAttr);
}

void FuncDirectiveAttr::print(AsmPrinter &p) const {
  p << "<pipeline=" << (getPipeline() ? "true" : "false")
    << ", targetInterval=" << getTargetInterval()
    << ", dataflow=" << (getDataflow() ? "true" : "false") << ">";
}

//===----------------------------------------------------------------------===//
// TableGen'd definitions
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/DataflowDialect.cpp.inc"

#include "sodap/Dialect/Dataflow/DataflowEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "sodap/Dialect/Dataflow/DataflowTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "sodap/Dialect/Dataflow/DataflowAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "sodap/Dialect/Dataflow/Dataflow.cpp.inc"
