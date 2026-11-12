// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Isolate invocations from an untiled, bufferized straight-line matmul graph.
// The pass has no tile/schedule options. Its output is shared by all hardware
// implementations of ReusableGemmABI.h.
#include "sodap/SODAPPasses.h"
#include "sodap/ExecutionEngine/ReusableGemmABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SetVector.h"
#include <limits>

namespace mlir::sodap {
#define GEN_PASS_DEF_REUSABLEGEMMTOESP
#include "sodap/SODAPPasses.h.inc"
namespace {
Value rootOf(Value value) {
  while (true) {
    if (auto op = value.getDefiningOp<memref::ExpandShapeOp>())
      value = op.getSrc();
    else if (auto op = value.getDefiningOp<memref::CollapseShapeOp>())
      value = op.getSrc();
    else if (auto op = value.getDefiningOp<memref::CastOp>())
      value = op.getSource();
    else
      return value;
  }
}

bool supportedBuffer(Type type) {
  auto mem = dyn_cast<MemRefType>(type);
  if (!mem || !mem.hasStaticShape() || !mem.getElementType().isF32() ||
      !mem.getLayout().isIdentity()) return false;
  uint64_t elements = 1;
  for (int64_t extent : mem.getShape()) {
    if (extent <= 0 || uint64_t(extent) > (1u << 27) / elements) return false;
    elements *= extent;
  }
  return true;
}

struct Invocation {
  Value a, b, c;
  int64_t m, n, k;
};

class ReusableGemmToESP : public impl::ReusableGemmToESPBase<ReusableGemmToESP> {
public:
  using ReusableGemmToESPBase::ReusableGemmToESPBase;
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto function = module.lookupSymbol<func::FuncOp>(kernel);
    if (!function || function.isExternal() ||
        !llvm::hasSingleElement(function.getBody()) || function.getNumResults()) {
      module.emitError("reusable GEMM requires a single-block function with output arguments; canonicalize first");
      return signalPassFailure();
    }
    if (instance < 0) {
      function.emitError("instance must be nonnegative");
      return signalPassFailure();
    }
    for (Value arg : function.getArguments())
      if (!supportedBuffer(arg.getType())) {
        function.emitError("initial GEMM isolation requires static contiguous f32 arguments");
        return signalPassFailure();
      }

    SmallVector<Invocation> calls;
    SmallVector<std::pair<Value, Value>> outputs;
    llvm::SetVector<Value> buffers;
    llvm::SmallDenseSet<Value> filled, produced, readArguments, returned;
    bool sawCopy = false;
    // Preflight the complete graph before rewriting any operation.
    for (Operation &op : function.front()) {
      if (isa<arith::ConstantOp, func::ReturnOp>(op))
        continue;
      if (isa<memref::AllocOp, memref::ExpandShapeOp,
              memref::CollapseShapeOp, memref::CastOp>(op)) {
        if (op.getNumResults() != 1 || !supportedBuffer(op.getResult(0).getType())) {
          op.emitError("expected a static contiguous f32 buffer or full reshape");
          return signalPassFailure();
        }
        continue;
      }
      if (isa<memref::DeallocOp>(op))
        continue; // Arena lifetime covers the entire graph.
      if (auto fill = dyn_cast<linalg::FillOp>(op)) {
        Value root = rootOf(fill.getOutputs()[0]);
        auto scalar = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
        auto value = scalar ? dyn_cast<FloatAttr>(scalar.getValue()) : FloatAttr();
        if (!value || !value.getValue().isPosZero() ||
            produced.contains(root) || filled.contains(root)) {
          op.emitError("reusable GEMM supports one zero initialization per output");
          return signalPassFailure();
        }
        filled.insert(root);
        continue;
      }
      if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
        auto matmul = cast<linalg::LinalgOp>(&op);
        Value av = matmul.getDpsInputs()[0], bv = matmul.getDpsInputs()[1];
        Value cv = matmul.getDpsInits()[0];
        auto a = cast<MemRefType>(av.getType());
        auto b = cast<MemRefType>(bv.getType());
        auto c = cast<MemRefType>(cv.getType());
        unsigned rank = isa<linalg::BatchMatmulOp>(op) ? 3 : 2;
        if (sawCopy || a.getRank() != rank || b.getRank() != rank ||
            c.getRank() != rank || !supportedBuffer(a) || !supportedBuffer(b) ||
            !supportedBuffer(c) || (rank == 3 &&
                (a.getDimSize(0) != 1 || b.getDimSize(0) != 1 || c.getDimSize(0) != 1))) {
          op.emitError("expected matmul or batch-one matmul before terminal output copies");
          return signalPassFailure();
        }
        Invocation call{rootOf(av), rootOf(bv), rootOf(cv),
                        a.getDimSize(rank - 2), b.getDimSize(rank - 1),
                        a.getDimSize(rank - 1)};
        if (b.getDimSize(rank - 2) != call.k ||
            c.getDimSize(rank - 2) != call.m || c.getDimSize(rank - 1) != call.n ||
            call.a == call.c || call.b == call.c || !filled.erase(call.c) ||
            produced.contains(call.c)) {
          op.emitError("expected compatible GEMM shapes and a distinct zero-initialized output");
          return signalPassFailure();
        }
        for (Value input : {call.a, call.b}) {
          if (!produced.contains(input)) {
            if (!isa<BlockArgument>(input) || filled.contains(input)) {
              op.emitError("GEMM input must be a function argument or a completed GEMM result");
              return signalPassFailure();
            }
            readArguments.insert(input);
          }
          buffers.insert(input);
        }
        buffers.insert(call.c);
        produced.insert(call.c);
        calls.push_back(call);
        continue;
      }
      if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        sawCopy = true;
        Value source = rootOf(copy.getSource()), dest = rootOf(copy.getTarget());
        if (!produced.contains(source) || !isa<BlockArgument>(dest) ||
            readArguments.contains(dest) || produced.contains(dest) ||
            !returned.insert(dest).second ||
            cast<MemRefType>(source.getType()).getNumElements() !=
                cast<MemRefType>(dest.getType()).getNumElements()) {
          op.emitError("only terminal copies of GEMM results to distinct output arguments are supported");
          return signalPassFailure();
        }
        outputs.push_back({source, dest});
        continue;
      }
      op.emitError("unsupported operation in reusable GEMM graph");
      return signalPassFailure();
    }
    if (calls.empty() || !filled.empty()) {
      function.emitError("expected GEMM invocations and no unused zero fills");
      return signalPassFailure();
    }
    for (Value arg : function.getArguments())
      if (produced.contains(arg))
        outputs.push_back({arg, arg});

    DenseMap<Value, int64_t> offset;
    int64_t bytes = GEMM_HEADER_BYTES;
    for (Value buffer : buffers) {
      int64_t elements = cast<MemRefType>(buffer.getType()).getNumElements();
      if (elements > ((1 << 30) - bytes) / GEMM_SLOT_BYTES) {
        function.emitError("GEMM arena exceeds the one-GiB interface limit");
        return signalPassFailure();
      }
      offset[buffer] = bytes;
      bytes += elements * GEMM_SLOT_BYTES;
    }

    OpBuilder builder(module.getContext());
    Location loc = function.getLoc();
    auto i64 = builder.getI64Type();
    auto f32 = builder.getF32Type();
    auto unranked = UnrankedMemRefType::get(f32, 0);
    auto declare = [&](StringRef name, TypeRange inputs, TypeRange results) {
      OpBuilder::InsertionGuard guard(builder);
      if (module.lookupSymbol(name))
        return;
      builder.setInsertionPointToStart(module.getBody());
      builder.create<func::FuncOp>(loc, name,
          builder.getFunctionType(inputs, results)).setPrivate();
    };
    declare("esp_gemm_open", {i64, i64}, {i64});
    declare("esp_gemm_close", {i64}, {});
    declare("esp_gemm_copy_in", {i64, i64, unranked}, {});
    declare("esp_gemm_copy_out", {i64, i64, unranked}, {});
    SmallVector<Type> invokeTypes(10, i64);
    invokeTypes.append({f32, f32});
    declare("esp_gemm_invoke", invokeTypes, {});

    // Retain host arguments, but no host allocation for intermediate results.
    function.getBody().dropAllReferences();
    function.front().clear();
    builder.setInsertionPointToStart(&function.front());
    auto constant = [&](int64_t n) -> Value {
      return builder.create<arith::ConstantIntOp>(loc, n, 64);
    };
    Value context = builder.create<func::CallOp>(loc, "esp_gemm_open", TypeRange{i64},
        ValueRange{constant(bytes), constant(instance)}).getResult(0);
    for (Value arg : function.getArguments()) {
      if (!readArguments.contains(arg)) continue;
      Value cast = builder.create<memref::CastOp>(loc, unranked, arg);
      builder.create<func::CallOp>(loc, "esp_gemm_copy_in", TypeRange{},
          ValueRange{context, constant(offset[arg]), cast});
    }
    Value one = builder.create<arith::ConstantFloatOp>(loc, APFloat(1.0f), f32);
    Value zero = builder.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), f32);
    for (auto [index, call] : llvm::enumerate(calls)) {
      auto invoke = builder.create<func::CallOp>(loc, "esp_gemm_invoke", TypeRange{},
          ValueRange{context, constant(offset[call.a]), constant(offset[call.b]),
            constant(offset[call.c]), constant(call.m), constant(call.n), constant(call.k),
            constant(call.k * GEMM_SLOT_BYTES), constant(call.n * GEMM_SLOT_BYTES),
            constant(call.n * GEMM_SLOT_BYTES), one, zero});
      invoke->setAttr("sodap.operator", builder.getStringAttr("gemm_f32_v1"));
      invoke->setAttr("sodap.invocation", builder.getI64IntegerAttr(index));
      invoke->setAttr("sodap.instance", builder.getI64IntegerAttr(instance));
    }
    for (auto [source, destination] : outputs) {
      Value cast = builder.create<memref::CastOp>(loc, unranked, destination);
      builder.create<func::CallOp>(loc, "esp_gemm_copy_out", TypeRange{},
          ValueRange{context, constant(offset[source]), cast});
    }
    builder.create<func::CallOp>(loc, "esp_gemm_close", TypeRange{}, ValueRange{context});
    builder.create<func::ReturnOp>(loc);
    function->setAttr("sodap.gemm_abi", builder.getI32IntegerAttr(GEMM_VERSION));
    function->setAttr("sodap.gemm_invocations", builder.getI64IntegerAttr(calls.size()));
  }
};
} // namespace
} // namespace mlir::sodap
