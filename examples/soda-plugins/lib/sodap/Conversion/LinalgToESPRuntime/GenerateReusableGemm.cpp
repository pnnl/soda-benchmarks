// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// A parameterized GEMM implementation. Only the three local tile extents are
// specialization parameters; all matrix dimensions and addresses come from
// the descriptor read at every launch. This baseline deliberately uses one
// sequential DMA/compute process so the invocation ABI is independent of a
// future load/compute/store schedule.
#include "sodap/SODAPPasses.h"
#include "sodap/Dialect/Dataflow/Dataflow.h"
#include "sodap/ExecutionEngine/ReusableGemmABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::sodap {
#define GEN_PASS_DEF_GENERATEREUSABLEGEMM
#include "sodap/SODAPPasses.h.inc"
namespace {
using namespace dataflow;

class KernelBuilder {
public:
  KernelBuilder(OpBuilder &builder, Location location, ValueRange ports)
      : b(builder), loc(location), rc(ports[0]), rd(ports[1]),
        wc(ports[2]), wd(ports[3]) {}

  Value idx(int64_t n) { return b.create<arith::ConstantIndexOp>(loc, n); }
  Value add(Value a, Value c) { return b.create<arith::AddIOp>(loc, a, c); }
  Value mul(Value a, Value c) { return b.create<arith::MulIOp>(loc, a, c); }
  Value bound(Value end, Value start, int64_t tile) {
    return b.create<arith::MinUIOp>(loc,
        b.create<arith::SubIOp>(loc, end, start), idx(tile));
  }
  void loop(Value lo, Value hi, Value step,
            llvm::function_ref<void(Value)> body) {
    auto op = b.create<scf::ForOp>(loc, lo, hi, step);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(op.getBody());
    body(op.getInductionVar());
  }
  void request(Value stream, Value address, Value count) {
    Value len = b.create<arith::IndexCastOp>(loc, b.getI64Type(), count);
    Value addr = b.create<arith::IndexCastOp>(loc, b.getI64Type(), address);
    Value shift = b.create<arith::ConstantIntOp>(loc, 32, 64);
    Value packet = b.create<arith::OrIOp>(loc,
        b.create<arith::ShLIOp>(loc, len, shift), addr);
    b.create<StreamWriteOp>(loc, stream, packet);
  }
  Value readWord() {
    Value word = b.create<StreamReadOp>(loc, b.getI32Type(), rd).getResult();
    b.create<StreamReadOp>(loc, b.getI32Type(), rd); // DMA64 padding.
    return word;
  }
  Value load(Value buffer, Value i, Value j) {
    return b.create<memref::LoadOp>(loc, buffer, ValueRange{i, j});
  }
  void store(Value value, Value buffer, Value i, Value j) {
    b.create<memref::StoreOp>(loc, value, buffer, ValueRange{i, j});
  }
  Value allocate(int64_t rows, int64_t cols) {
    return b.create<memref::AllocaOp>(loc,
        MemRefType::get({rows, cols}, b.getF32Type()));
  }
  void readTile(Value base, Value stride, Value row, Value col,
                Value rows, Value cols, Value buffer) {
    loop(idx(0), rows, idx(1), [&](Value i) {
      Value address = add(base, add(mul(add(row, i), stride),
                                    mul(col, idx(GEMM_WORD_STRIDE))));
      request(rc, address, mul(cols, idx(GEMM_WORD_STRIDE)));
      loop(idx(0), cols, idx(1), [&](Value j) {
        store(b.create<arith::BitcastOp>(loc, b.getF32Type(), readWord()),
              buffer, i, j);
      });
    });
  }

  void build(int64_t tm, int64_t tn, int64_t tk) {
    Value a = allocate(tm, tk), rhs = allocate(tk, tn);
    Value accum = allocate(tm, tn), oldC = allocate(tm, tn);
    Value config = b.create<memref::AllocaOp>(loc,
        MemRefType::get({GEMM_FIELD_COUNT}, b.getI32Type()));
    request(rc, idx(0), idx(GEMM_FIELD_COUNT * GEMM_WORD_STRIDE));
    // A single ordered stream reader latches the complete descriptor before
    // any field is used. Keep this loop independent of the matrix schedule.
    loop(idx(0), idx(GEMM_FIELD_COUNT), idx(1), [&](Value f) {
      b.create<memref::StoreOp>(loc, readWord(), config, ValueRange{f});
    });
    SmallVector<Value> fields;
    for (unsigned f = 0; f < GEMM_FIELD_COUNT; ++f)
      fields.push_back(b.create<memref::LoadOp>(loc, config, ValueRange{idx(f)}));
    auto dimension = [&](unsigned f) -> Value {
      return b.create<arith::IndexCastUIOp>(loc, b.getIndexType(), fields[f]);
    };
    Value m = dimension(GEMM_FIELD_M), n = dimension(GEMM_FIELD_N);
    Value k = dimension(GEMM_FIELD_K);
    Value baseA = dimension(GEMM_FIELD_A), baseB = dimension(GEMM_FIELD_B);
    Value baseC = dimension(GEMM_FIELD_C);
    Value lda = dimension(GEMM_FIELD_LDA), ldb = dimension(GEMM_FIELD_LDB);
    Value ldc = dimension(GEMM_FIELD_LDC);
    Value alpha = b.create<arith::BitcastOp>(loc, b.getF32Type(), fields[GEMM_FIELD_ALPHA]);
    Value beta = b.create<arith::BitcastOp>(loc, b.getF32Type(), fields[GEMM_FIELD_BETA]);
    Value zero = b.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), b.getF32Type());
    Value hasC = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, beta, zero);
    Value magic = b.create<arith::ConstantIntOp>(loc, GEMM_MAGIC, 32);
    Value version = b.create<arith::ConstantIntOp>(loc, GEMM_VERSION, 32);
    Value valid = b.create<arith::AndIOp>(loc,
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, fields[GEMM_FIELD_MAGIC], magic),
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, fields[GEMM_FIELD_VERSION], version));
    auto configured = b.create<scf::IfOp>(loc, valid, false);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&configured.getThenRegion().front());
      loop(idx(0), m, idx(tm), [&](Value ii) {
        Value rows = bound(m, ii, tm);
        loop(idx(0), n, idx(tn), [&](Value jj) {
          Value cols = bound(n, jj, tn);
          loop(idx(0), rows, idx(1), [&](Value i) {
            loop(idx(0), cols, idx(1), [&](Value j) { store(zero, accum, i, j); });
          });
          loop(idx(0), k, idx(tk), [&](Value kk) {
            Value depth = bound(k, kk, tk);
            readTile(baseA, lda, ii, kk, rows, depth, a);
            readTile(baseB, ldb, kk, jj, depth, cols, rhs);
            loop(idx(0), rows, idx(1), [&](Value i) {
              loop(idx(0), cols, idx(1), [&](Value j) {
                loop(idx(0), depth, idx(1), [&](Value q) {
                  Value product = b.create<arith::MulFOp>(loc, load(a, i, q), load(rhs, q, j));
                  store(b.create<arith::AddFOp>(loc, load(accum, i, j), product), accum, i, j);
                });
              });
            });
          });
          auto readC = b.create<scf::IfOp>(loc, hasC, false);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(&readC.getThenRegion().front());
            readTile(baseC, ldc, ii, jj, rows, cols, oldC);
          }
          loop(idx(0), rows, idx(1), [&](Value i) {
            Value address = add(baseC, add(mul(add(ii, i), ldc),
                                          mul(jj, idx(GEMM_WORD_STRIDE))));
            request(wc, address, mul(cols, idx(GEMM_WORD_STRIDE)));
            loop(idx(0), cols, idx(1), [&](Value j) {
              Value scaled = b.create<arith::MulFOp>(loc, alpha, load(accum, i, j));
              auto result = b.create<scf::IfOp>(loc, TypeRange{b.getF32Type()}, hasC, true);
              {
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(&result.getThenRegion().front());
                Value prior = b.create<arith::MulFOp>(loc, beta, load(oldC, i, j));
                Value sum = b.create<arith::AddFOp>(loc, scaled, prior);
                b.create<scf::YieldOp>(loc, sum);
                b.setInsertionPointToStart(&result.getElseRegion().front());
                b.create<scf::YieldOp>(loc, scaled);
              }
              Value bits = b.create<arith::BitcastOp>(loc, b.getI32Type(), result.getResult(0));
              b.create<StreamWriteOp>(loc, wd, bits);
              b.create<StreamWriteOp>(loc, wd, b.create<arith::ConstantIntOp>(loc, 0, 32));
            });
          });
        });
      });
    }
  }
private:
  OpBuilder &b;
  Location loc;
  Value rc, rd, wc, wd;
};

class GenerateReusableGemm : public impl::GenerateReusableGemmBase<GenerateReusableGemm> {
public:
  using GenerateReusableGemmBase::GenerateReusableGemmBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (tileM < 1 || tileN < 1 || tileK < 1 ||
        tileM > 128 || tileN > 128 || tileK > 128) {
      module.emitError("local tile extents must be in [1,128]");
      return signalPassFailure();
    }
    if (!module.getBody()->empty()) {
      module.emitError("generate reusable hardware in a separate empty module");
      return signalPassFailure();
    }
    OpBuilder b(module.getContext());
    b.setInsertionPointToEnd(module.getBody());
    auto ctrl = StreamType::get(module.getContext(), b.getI64Type(), 8, StreamInterface::AXIS);
    auto data = StreamType::get(module.getContext(), b.getI32Type(), 16, StreamInterface::AXIS);
    auto type = b.getFunctionType({ctrl, data, ctrl, data}, {});
    auto engine = b.create<func::FuncOp>(module.getLoc(), "gemm_engine", type);
    b.setInsertionPointToStart(engine.addEntryBlock());
    KernelBuilder(b, module.getLoc(), engine.getArguments()).build(tileM, tileN, tileK);
    b.create<func::ReturnOp>(module.getLoc());
    b.setInsertionPointToEnd(module.getBody());
    auto top = b.create<func::FuncOp>(module.getLoc(), topName, type);
    b.setInsertionPointToStart(top.addEntryBlock());
    b.create<func::CallOp>(module.getLoc(), engine, top.getArguments());
    b.create<func::ReturnOp>(module.getLoc());
    module->setAttr("sodap.gemm_abi", b.getI32IntegerAttr(GEMM_VERSION));
    module->setAttr("sodap.gemm_tiles", b.getDenseI64ArrayAttr({tileM, tileN, tileK}));
  }
};
} // namespace
} // namespace mlir::sodap
