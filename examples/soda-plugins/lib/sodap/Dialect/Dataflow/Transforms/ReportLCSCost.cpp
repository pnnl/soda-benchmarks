// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/Dialect/Dataflow/Transforms/Passes.h"
#include "Prefix.h"
#include "sodap/Dialect/Dataflow/Dataflow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FileSystem.h"
#include <limits>

namespace mlir::sodap::dataflow {
#define GEN_PASS_DEF_REPORTLCSCOST
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::sodap::dataflow;
namespace {
struct ReportLCSCost : mlir::sodap::dataflow::impl::ReportLCSCostBase<ReportLCSCost> {
  using ReportLCSCostBase::ReportLCSCostBase;
  bool invalid = false;
  int64_t multiply(int64_t a, int64_t b) {
    if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max()/b)) {
      invalid = true;
      return 0;
    }
    return a*b;
  }
  void add(int64_t &a, int64_t b) {
    if (b < 0 || a > std::numeric_limits<int64_t>::max()-b) invalid = true;
    else a += b;
  }
  /// The index arguments' values in the invocation analysed
  /// (sodap.invocations of a sodap-join-multiple-tiles tile), if any.
  SmallVector<int64_t> inv;
  std::optional<int64_t> valueOf(Value v) {
    if (std::optional<int64_t> c = getConstantIntValue(v))
      return c;
    auto arg = dyn_cast<BlockArgument>(v);
    auto func = arg ? dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp()) : nullptr;
    if (!func || !arg.getType().isIndex())
      return std::nullopt;
    size_t position = llvm::count_if(func.getArguments().take_front(arg.getArgNumber()),
                                     [](BlockArgument a) { return a.getType().isIndex(); });
    if (position >= inv.size())
      return std::nullopt;
    return inv[position];
  }
  int64_t repetitions(Operation *op) {
    int64_t n = 1;
    for (Operation *p=op->getParentOp(); p && !isa<func::FuncOp>(p); p=p->getParentOp()) {
      if (auto loop=dyn_cast<scf::ForOp>(p)) {
        auto lb=valueOf(loop.getLowerBound()), ub=valueOf(loop.getUpperBound()),
             step=valueOf(loop.getStep());
        if (!lb || !ub || !step || *lb<0 || *ub<*lb || *step<=0) { invalid=true; return 0; }
        n=multiply(n,llvm::divideCeil(*ub-*lb,*step));
      } else if (!isa<TaskOp,DispatchOp>(p)) {
        op->emitError("cost analysis requires static scf.for/task nesting");
        invalid=true;
        return 0;
      }
    }
    return n;
  }
  int64_t elements(ShapedType type) {
    int64_t n=1;
    if (!type.hasStaticShape()) { invalid=true; return 0; }
    for (int64_t d:type.getShape()) n=multiply(n,d);
    return n;
  }
  // CopyToDMA transfers only function arguments (or their subviews). Copies
  // between local buffers are not traffic on the external DMA interface.
  bool isExternal(Value value) {
    while (auto view = value.getDefiningOp<memref::SubViewOp>())
      value = view.getSource();
    auto arg = dyn_cast<BlockArgument>(value);
    return arg && isa<func::FuncOp>(arg.getOwner()->getParentOp());
  }
  void runOnOperation() override {
    if (requestCycles<0 || wordCycles<0 || (wordStride!=1 && wordStride!=2) || budget<0) {
      getOperation().emitError("expected nonnegative costs/budget and DMA stride 1 or 2");
      return signalPassFailure();
    }
    int64_t bytes=0,requests=0,words=0,iterations=0;
    llvm::json::Array copies,compute;
    // A tile with invocations: the work summed over them, the storage once.
    SmallVector<SmallVector<int64_t>> invocations{{}};
    if (auto list = getOperation()->getAttrOfType<ArrayAttr>("sodap.invocations")) {
      invocations.clear();
      for (Attribute each : list)
        invocations.push_back(llvm::to_vector(cast<DenseI64ArrayAttr>(each).asArrayRef()));
    }
    for (auto [i, each] : llvm::enumerate(invocations)) {
    inv = each;
    bool first = i == 0;
    getOperation().walk([&](Operation *op) {
      if (!op->getParentOfType<DispatchOp>()) return;
      if (isa<memref::AllocOp,memref::AllocaOp>(op)) {
        if (!first) return;
        auto t=cast<MemRefType>(op->getResult(0).getType());
        if (!t.getElementType().isIntOrFloat()) { invalid=true; return; }
        add(bytes,multiply(elements(t),llvm::divideCeil(t.getElementTypeBitWidth(),8)));
      } else if (auto copy=dyn_cast<memref::CopyOp>(op)) {
        if (!isExternal(copy.getSource()) && !isExternal(copy.getTarget()))
          return;
        auto t=copy.getSource().getType();
        if (!t.getElementType().isF32()) { invalid=true; return; }
        int64_t n=repetitions(op),rows=1;
        for (int64_t d:t.getShape().drop_back(t.getRank()?1:0)) rows=multiply(rows,d);
        int64_t r=multiply(n,rows),w=multiply(multiply(n,elements(t)),wordStride);
        add(requests,r); add(words,w);
        copies.push_back(llvm::json::Object{{"invocations",n},{"requests",r},{"words",w}});
      } else if (auto linalg=dyn_cast<linalg::LinalgOp>(op)) {
        int64_t n=repetitions(op);
        for (int64_t d:linalg.getStaticLoopRanges()) n=multiply(n,d);
        add(iterations,n);
        compute.push_back(llvm::json::Object{{"op",op->getName().getStringRef()},
                                            {"iterations",n},{"reduction",linalg.getNumReductionLoops()>0}});
      }
    });
    }
    int64_t cycles=multiply(requests,requestCycles);
    add(cycles,multiply(words,wordCycles));
    if (invalid) {
      getOperation().emitError("unsupported dynamic/control-dependent schedule or overflow in static cost analysis");
      return signalPassFailure();
    }
    llvm::json::Object result{{"dma_cycles",cycles},{"dma_requests",requests},{"dma_words",words},
        {"plm_bytes",bytes},{"compute_iterations",iterations},{"dma_word_stride",int64_t(wordStride)},
        {"copies",std::move(copies)},{"compute",std::move(compute)},
        {"scope","Static scheduled work and logical allocation; not measured latency or FPGA resource use"}};
    std::error_code ec;
    // Nested in a tile module: its own report and budget, in its directory.
    llvm::raw_fd_ostream os(getTileFilePath(getOperation(),file),ec);
    if (ec) { getOperation().emitError("cannot write cost report: ")<<ec.message(); return signalPassFailure(); }
    os<<llvm::formatv("{0:2}\n",llvm::json::Value(std::move(result)));
    if (budget && bytes>budget) {
      getOperation().emitError("scheduled local storage ")<<bytes<<" exceeds budget "<<budget;
      signalPassFailure();
    }
  }
};
}
