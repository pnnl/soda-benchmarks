// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Parameterized GEMM implementations. Local tile extents and the sequential
// or LCS schedule specialize hardware; matrix dimensions and addresses come
// from the descriptor read at every launch. Both schedules share the same ABI.
#include "sodap/SODAPPasses.h"
#include "sodap/Dialect/Dataflow/Dataflow.h"
#include "sodap/Dialect/Dataflow/Transforms/Passes.h"
#include "sodap/ExecutionEngine/ReusableGemmABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"

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
protected:
  OpBuilder &b;
  Location loc;
  Value rc, rd, wc, wd;
};

// The three tasks have identical descriptor-controlled tile traversals. Only
// local storage extents are static. PingPong supplies the buffer ownership
// protocol; TasksToNodes/NodesToFunc retain Tommaso's common Bambu lowering.
class PipelineBuilder : public KernelBuilder {
public:
  using KernelBuilder::KernelBuilder;
  SmallVector<Value> fields;
  Value m, n, k, baseA, baseB, baseC, lda, ldb, ldc, alpha, beta, hasC, zero;

  void tile(llvm::function_ref<void()> body) {
    auto region = b.create<scf::ExecuteRegionOp>(loc, TypeRange{});
    region->setAttr("sodap.dataflow_tile", b.getUnitAttr());
    region.getRegion().push_back(new Block());
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&region.getRegion().front());
    body();
    b.create<scf::YieldOp>(loc);
  }
  void condition(Value test, llvm::function_ref<void()> yes,
                 llvm::function_ref<void()> no = {}) {
    auto op = b.create<scf::IfOp>(loc, test, bool(no));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&op.getThenRegion().front());
    yes();
    if (no) {
      b.setInsertionPointToStart(&op.getElseRegion().front());
      no();
    }
  }
  Value config(Value input, Value computeConfig = {}, Value storeConfig = {}) {
    auto buffer = b.create<memref::AllocaOp>(loc,
        MemRefType::get({GEMM_FIELD_COUNT}, b.getI32Type()));
    if (!input)
      request(rc, idx(0), idx(GEMM_FIELD_COUNT * GEMM_WORD_STRIDE));
    loop(idx(0), idx(GEMM_FIELD_COUNT), idx(1), [&](Value f) {
      Value value = input ? b.create<StreamReadOp>(loc, b.getI32Type(), input).getResult()
                          : readWord();
      b.create<memref::StoreOp>(loc, value, buffer, ValueRange{f});
      if (!input) {
        b.create<StreamWriteOp>(loc, computeConfig, value);
        b.create<StreamWriteOp>(loc, storeConfig, value);
      }
    });
    fields.clear();
    for (unsigned f = 0; f < GEMM_FIELD_COUNT; ++f)
      fields.push_back(b.create<memref::LoadOp>(loc, buffer, ValueRange{idx(f)}));
    auto dim = [&](unsigned f) -> Value {
      return b.create<arith::IndexCastUIOp>(loc, b.getIndexType(), fields[f]);
    };
    m=dim(GEMM_FIELD_M); n=dim(GEMM_FIELD_N); k=dim(GEMM_FIELD_K);
    baseA=dim(GEMM_FIELD_A); baseB=dim(GEMM_FIELD_B); baseC=dim(GEMM_FIELD_C);
    lda=dim(GEMM_FIELD_LDA); ldb=dim(GEMM_FIELD_LDB); ldc=dim(GEMM_FIELD_LDC);
    alpha=b.create<arith::BitcastOp>(loc,b.getF32Type(),fields[GEMM_FIELD_ALPHA]);
    beta=b.create<arith::BitcastOp>(loc,b.getF32Type(),fields[GEMM_FIELD_BETA]);
    zero=b.create<arith::ConstantFloatOp>(loc,APFloat(0.0f),b.getF32Type());
    hasC=b.create<arith::CmpFOp>(loc,arith::CmpFPredicate::UNE,beta,zero);
    return b.create<arith::AndIOp>(loc,
        b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::eq,fields[GEMM_FIELD_MAGIC],
            b.create<arith::ConstantIntOp>(loc,GEMM_MAGIC,32)),
        b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::eq,fields[GEMM_FIELD_VERSION],
            b.create<arith::ConstantIntOp>(loc,GEMM_VERSION,32)));
  }
  void outputs(int64_t tm, int64_t tn,
               llvm::function_ref<void(Value,Value,Value,Value)> body) {
    loop(idx(0),m,idx(tm),[&](Value ii) {
      Value rows=bound(m,ii,tm);
      loop(idx(0),n,idx(tn),[&](Value jj) { body(ii,jj,rows,bound(n,jj,tn)); });
    });
  }
  void fill(Value buffer, Value rows, Value cols) {
    loop(idx(0),rows,idx(1),[&](Value i) {
      loop(idx(0),cols,idx(1),[&](Value j) { store(zero,buffer,i,j); });
    });
  }
  void loadStage(int64_t tm,int64_t tn,int64_t tk,Value a,Value rhs,Value oldC,
                 Value computeConfig,Value storeConfig) {
    Value valid=config({},computeConfig,storeConfig);
    condition(valid,[&] {
      outputs(tm,tn,[&](Value ii,Value jj,Value rows,Value cols) {
        // Every output tile publishes one old-C token, even when beta=0.
        // The zero case performs no C DMA reads and preserves that ABI rule.
        tile([&] {
          condition(hasC,[&] { readTile(baseC,ldc,ii,jj,rows,cols,oldC); },
                        [&] { fill(oldC,rows,cols); });
        });
        loop(idx(0),k,idx(tk),[&](Value kk) {
          Value depth=bound(k,kk,tk);
          tile([&] {
            readTile(baseA,lda,ii,kk,rows,depth,a);
            readTile(baseB,ldb,kk,jj,depth,cols,rhs);
          });
        });
      });
    });
  }
  void computeStage(int64_t tm,int64_t tn,int64_t tk,Value a,Value rhs,
                    Value oldC,Value result,Value configStream) {
    Value valid=config(configStream);
    Value accum=allocate(tm,tn);
    condition(valid,[&] {
      outputs(tm,tn,[&](Value ii,Value jj,Value rows,Value cols) {
        fill(accum,rows,cols);
        loop(idx(0),k,idx(tk),[&](Value kk) {
          Value depth=bound(k,kk,tk);
          tile([&] {
            loop(idx(0),rows,idx(1),[&](Value i) {
              loop(idx(0),cols,idx(1),[&](Value j) {
                loop(idx(0),depth,idx(1),[&](Value q) {
                  Value product=b.create<arith::MulFOp>(loc,load(a,i,q),load(rhs,q,j));
                  store(b.create<arith::AddFOp>(loc,load(accum,i,j),product),accum,i,j);
                });
              });
            });
          });
        });
        tile([&] {
          loop(idx(0),rows,idx(1),[&](Value i) {
            loop(idx(0),cols,idx(1),[&](Value j) {
              Value scaled=b.create<arith::MulFOp>(loc,alpha,load(accum,i,j));
              condition(hasC,[&] {
                Value prior=b.create<arith::MulFOp>(loc,beta,load(oldC,i,j));
                store(b.create<arith::AddFOp>(loc,scaled,prior),result,i,j);
              },[&] { store(scaled,result,i,j); });
            });
          });
        });
      });
    });
  }
  void storeStage(int64_t tm,int64_t tn,Value result,Value configStream) {
    Value valid=config(configStream);
    condition(valid,[&] {
      outputs(tm,tn,[&](Value ii,Value jj,Value rows,Value cols) {
        tile([&] {
          loop(idx(0),rows,idx(1),[&](Value i) {
            Value address=add(baseC,add(mul(add(ii,i),ldc),mul(jj,idx(GEMM_WORD_STRIDE))));
            request(wc,address,mul(cols,idx(GEMM_WORD_STRIDE)));
            loop(idx(0),cols,idx(1),[&](Value j) {
              Value bits=b.create<arith::BitcastOp>(loc,b.getI32Type(),load(result,i,j));
              b.create<StreamWriteOp>(loc,wd,bits);
              b.create<StreamWriteOp>(loc,wd,b.create<arith::ConstantIntOp>(loc,0,32));
            });
          });
        });
      });
    });
  }
};

class GenerateReusableGemm : public impl::GenerateReusableGemmBase<GenerateReusableGemm> {
public:
  using GenerateReusableGemmBase::GenerateReusableGemmBase;
  void runOnOperation() final {
    auto module = getOperation();
    if (schedule != "sequential" && schedule != "lcs") {
      module.emitError("GEMM schedule must be sequential or lcs");
      return signalPassFailure();
    }
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
    if (schedule == "lcs") {
      auto top = b.create<func::FuncOp>(module.getLoc(), topName, type);
      b.setInsertionPointToStart(top.addEntryBlock());
      auto dispatch = b.create<DispatchOp>(module.getLoc(), TypeRange{});
      dispatch.getBody().push_back(new Block());
      b.setInsertionPointToStart(&dispatch.getBody().front());
      auto alloc = [&](int64_t rows,int64_t cols) -> Value {
        return b.create<memref::AllocOp>(module.getLoc(),MemRefType::get({rows,cols},b.getF32Type()));
      };
      Value a=alloc(tileM,tileK), rhs=alloc(tileK,tileN);
      Value oldC=alloc(tileM,tileN), result=alloc(tileM,tileN);
      auto cfgType=StreamType::get(module.getContext(),b.getI32Type(),GEMM_FIELD_COUNT);
      Value computeConfig=b.create<StreamOp>(module.getLoc(),cfgType);
      Value storeConfig=b.create<StreamOp>(module.getLoc(),cfgType);
      auto task = [&](llvm::function_ref<void(PipelineBuilder &)> body) {
        auto op=b.create<TaskOp>(module.getLoc(),TypeRange{});
        op.getBody().push_back(new Block());
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&op.getBody().front());
        PipelineBuilder builder(b,module.getLoc(),top.getArguments());
        body(builder);
        b.create<dataflow::YieldOp>(module.getLoc());
      };
      task([&](PipelineBuilder &p) { p.loadStage(tileM,tileN,tileK,a,rhs,oldC,computeConfig,storeConfig); });
      task([&](PipelineBuilder &p) { p.computeStage(tileM,tileN,tileK,a,rhs,oldC,result,computeConfig); });
      task([&](PipelineBuilder &p) { p.storeStage(tileM,tileN,result,storeConfig); });
      b.create<dataflow::YieldOp>(module.getLoc());
      b.setInsertionPointAfter(dispatch);
      b.create<func::ReturnOp>(module.getLoc());
      // Consume the tile scopes before canonicalization can inline them.
      OpPassManager passes("builtin.module");
      passes.addNestedPass<func::FuncOp>(dataflow::createPingPong());
      passes.addNestedPass<func::FuncOp>(dataflow::createTasksToNodes());
      passes.addPass(dataflow::createNodesToFunc());
      if (failed(runPipeline(passes,module)))
        return signalPassFailure();
      module->setAttr("sodap.gemm_abi",b.getI32IntegerAttr(GEMM_VERSION));
      module->setAttr("sodap.gemm_tiles",b.getDenseI64ArrayAttr({tileM,tileN,tileK}));
      module->setAttr("sodap.gemm_schedule",b.getStringAttr(schedule));
      return;
    }
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
