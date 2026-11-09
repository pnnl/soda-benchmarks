// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/SODAPPasses.h"
#include "sodap/Dialect/Dataflow/Dataflow.h"
#include "sodap/ExecutionEngine/ReusableOperatorABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cfloat>

namespace mlir::sodap {
#define GEN_PASS_DEF_GENERATEREUSABLEOPERATOR
#include "sodap/SODAPPasses.h.inc"
namespace {
using namespace dataflow;
struct Generator {
  OpBuilder &b;
  Location loc;
  Value rc, rd, wc, wd;
  SmallVector<Value> cfg;
  Value z, one, fzero;
  Generator(OpBuilder &builder, Location location, ValueRange ports)
      : b(builder), loc(location), rc(ports[0]), rd(ports[1]), wc(ports[2]), wd(ports[3]) {
    z=idx(0); one=idx(1);
    fzero=b.create<arith::ConstantFloatOp>(loc,APFloat(0.0f),b.getF32Type());
  }
  Value idx(int64_t n) { return b.create<arith::ConstantIndexOp>(loc,n); }
  Value add(Value a,Value c) { return b.create<arith::AddIOp>(loc,a,c); }
  Value sub(Value a,Value c) { return b.create<arith::SubIOp>(loc,a,c); }
  Value mul(Value a,Value c) { return b.create<arith::MulIOp>(loc,a,c); }
  Value cmp(arith::CmpIPredicate p,Value a,Value c) { return b.create<arith::CmpIOp>(loc,p,a,c); }
  Value both(Value a,Value c) { return b.create<arith::AndIOp>(loc,a,c); }
  Value bound(Value end,Value start,int64_t amount) {
    return b.create<arith::MinUIOp>(loc,sub(end,start),idx(amount));
  }
  void loop(Value end,llvm::function_ref<void(Value)> body) { loop(z,end,one,body); }
  void loop(Value start,Value end,Value step,llvm::function_ref<void(Value)> body) {
    auto op=b.create<scf::ForOp>(loc,start,end,step);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(op.getBody()); body(op.getInductionVar());
  }
  void condition(Value test,llvm::function_ref<void()> yes,
                 llvm::function_ref<void()> no = {}) {
    auto op=b.create<scf::IfOp>(loc,test,bool(no));
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&op.getThenRegion().front()); yes();
    if(no) { b.setInsertionPointToStart(&op.getElseRegion().front()); no(); }
  }
  Value alloc(ArrayRef<int64_t> dims,Type type={}) {
    return b.create<memref::AllocaOp>(loc,MemRefType::get(dims,type?type:b.getF32Type()));
  }
  Value load(Value mem,ValueRange indices) { return b.create<memref::LoadOp>(loc,mem,indices); }
  void store(Value value,Value mem,ValueRange indices) { b.create<memref::StoreOp>(loc,value,mem,indices); }
  void request(Value port,Value address,Value count) {
    Value a=b.create<arith::IndexCastOp>(loc,b.getI64Type(),address);
    Value n=b.create<arith::IndexCastOp>(loc,b.getI64Type(),count);
    Value shift=b.create<arith::ConstantIntOp>(loc,32,64);
    Value packed=b.create<arith::OrIOp>(loc,b.create<arith::ShLIOp>(loc,n,shift),a);
    b.create<StreamWriteOp>(loc,port,packed);
  }
  Value readWord() {
    Value value=b.create<StreamReadOp>(loc,b.getI32Type(),rd).getResult();
    b.create<StreamReadOp>(loc,b.getI32Type(),rd);
    return value;
  }
  Value readFloat() { return b.create<arith::BitcastOp>(loc,b.getF32Type(),readWord()); }
  void writeFloat(Value value) {
    b.create<StreamWriteOp>(loc,wd,b.create<arith::BitcastOp>(loc,b.getI32Type(),value));
    b.create<StreamWriteOp>(loc,wd,b.create<arith::ConstantIntOp>(loc,0,32));
  }
  Value address(Value base,Value element) { return add(base,mul(element,idx(2))); }
  Value field(unsigned f) { return b.create<arith::IndexCastUIOp>(loc,b.getIndexType(),cfg[f]); }
  Value readConfig(uint32_t magic) {
    Value config=alloc({OP_HEADER_FIELDS},b.getI32Type());
    request(rc,z,idx(OP_HEADER_FIELDS*2));
    loop(idx(OP_HEADER_FIELDS),[&](Value i){store(readWord(),config,{i});});
    for(unsigned i=0;i<OP_HEADER_FIELDS;++i) cfg.push_back(load(config,{idx(i)}));
    Value m=b.create<arith::ConstantIntOp>(loc,magic,32);
    Value v=b.create<arith::ConstantIntOp>(loc,OP_VERSION,32);
    return both(cmp(arith::CmpIPredicate::eq,cfg[0],m),cmp(arith::CmpIPredicate::eq,cfg[1],v));
  }
  Value relu(Value value,Value enabled) {
    Value lo=b.create<arith::CmpFOp>(loc,arith::CmpFPredicate::OLT,value,fzero);
    Value upper=b.create<arith::ConstantFloatOp>(loc,APFloat(FLT_MAX),b.getF32Type());
    Value hi=b.create<arith::CmpFOp>(loc,arith::CmpFPredicate::OGT,value,upper);
    Value clipped=b.create<arith::SelectOp>(loc,lo,fzero,value);
    clipped=b.create<arith::SelectOp>(loc,hi,upper,clipped);
    return b.create<arith::SelectOp>(loc,enabled,clipped,value);
  }

  void addition(int64_t block) {
    Value a=alloc({block}),rhs=alloc({block});
    Value valid=readConfig(OP_ADD_MAGIC);
    condition(valid,[&]{
      Value n=field(ADD_COUNT),baseA=field(ADD_A),baseB=field(ADD_B),baseC=field(ADD_C);
      Value activation=cmp(arith::CmpIPredicate::ne,field(ADD_RELU),z);
      loop(z,n,idx(block),[&](Value start){
        Value count=bound(n,start,block);
        request(rc,address(baseA,start),mul(count,idx(2)));
        loop(count,[&](Value i){store(readFloat(),a,{i});});
        request(rc,address(baseB,start),mul(count,idx(2)));
        loop(count,[&](Value i){store(readFloat(),rhs,{i});});
        request(wc,address(baseC,start),mul(count,idx(2)));
        loop(count,[&](Value i){
          Value sum=b.create<arith::AddFOp>(loc,load(a,{i}),load(rhs,{i}));
          writeFloat(relu(sum,activation));
        });
      });
    });
  }

  void convolution(int64_t pixels,int64_t lanes,int64_t channels) {
    Value a=alloc({pixels,channels}),result=alloc({pixels,lanes}),bias=alloc({lanes});
    SmallVector<Value> weights,accum;
    // Separate banks per output lane allow parallel MACs. These resources are
    // chosen here, independently of the invocation contract and graph lowering.
    for(int64_t lane=0;lane<lanes;++lane) {
      weights.push_back(alloc({channels})); accum.push_back(alloc({pixels}));
    }
    Value valid=readConfig(OP_CONV_MAGIC);
    condition(valid,[&]{
      Value n=field(CONV_N),ih=field(CONV_IH),iw=field(CONV_IW),ic=field(CONV_IC);
      Value oc=field(CONV_OC),kh=field(CONV_KH),kw=field(CONV_KW);
      Value oh=field(CONV_OH),ow=field(CONV_OW),sh=field(CONV_SH),sw=field(CONV_SW);
      Value pt=field(CONV_PT),pl=field(CONV_PL),dh=field(CONV_DH),dw=field(CONV_DW);
      Value baseA=field(CONV_A),baseW=field(CONV_W),baseBias=field(CONV_BIAS),baseC=field(CONV_C);
      Value activation=cmp(arith::CmpIPredicate::ne,field(CONV_RELU),z);
      Value plane=mul(oh,ow);
      loop(n,[&](Value batch){
        loop(z,plane,idx(pixels),[&](Value start){
          Value rows=bound(plane,start,pixels);
          loop(z,oc,idx(lanes),[&](Value outChannel){
            Value columns=bound(oc,outChannel,lanes);
            loop(rows,[&](Value p){
              for(int64_t lane=0;lane<lanes;++lane) store(fzero,accum[lane],{p});
            });
            loop(kh,[&](Value ky){loop(kw,[&](Value kx){
              loop(z,ic,idx(channels),[&](Value inChannel){
                Value depth=bound(ic,inChannel,channels);
                loop(rows,[&](Value p){
                  Value linear=add(start,p);
                  Value y=b.create<arith::DivUIOp>(loc,linear,ow);
                  Value x=b.create<arith::RemUIOp>(loc,linear,ow);
                  Value sy=sub(add(mul(y,sh),mul(ky,dh)),pt);
                  Value sx=sub(add(mul(x,sw),mul(kx,dw)),pl);
                  Value inside=both(both(cmp(arith::CmpIPredicate::sge,sy,z),cmp(arith::CmpIPredicate::slt,sy,ih)),
                                    both(cmp(arith::CmpIPredicate::sge,sx,z),cmp(arith::CmpIPredicate::slt,sx,iw)));
                  condition(inside,[&]{
                    Value index=add(mul(add(mul(add(mul(batch,ih),sy),iw),sx),ic),inChannel);
                    request(rc,address(baseA,index),mul(depth,idx(2)));
                    loop(depth,[&](Value q){store(readFloat(),a,{p,q});});
                  },[&]{loop(depth,[&](Value q){store(fzero,a,{p,q});});});
                });
                for(int64_t lane=0;lane<lanes;++lane) {
                  condition(cmp(arith::CmpIPredicate::ult,idx(lane),columns),[&]{
                    Value index=add(mul(add(mul(add(mul(add(outChannel,idx(lane)),kh),ky),kw),kx),ic),inChannel);
                    request(rc,address(baseW,index),mul(depth,idx(2)));
                    loop(depth,[&](Value q){store(readFloat(),weights[lane],{q});});
                  },[&]{loop(depth,[&](Value q){store(fzero,weights[lane],{q});});});
                }
                loop(rows,[&](Value p){loop(depth,[&](Value q){
                  Value input=load(a,{p,q});
                  for(int64_t lane=0;lane<lanes;++lane) {
                    Value product=b.create<arith::MulFOp>(loc,input,load(weights[lane],{q}));
                    Value sum=b.create<arith::AddFOp>(loc,load(accum[lane],{p}),product);
                    store(sum,accum[lane],{p});
                  }
                });});
              });
            });});
            request(rc,address(baseBias,outChannel),mul(columns,idx(2)));
            loop(columns,[&](Value j){store(readFloat(),bias,{j});});
            // Repack banked accumulators for one ordered output-stream writer.
            loop(rows,[&](Value p){
              for(int64_t lane=0;lane<lanes;++lane) store(load(accum[lane],{p}),result,{p,idx(lane)});
            });
            loop(rows,[&](Value p){
              Value output=add(mul(add(mul(batch,plane),add(start,p)),oc),outChannel);
              request(wc,address(baseC,output),mul(columns,idx(2)));
              loop(columns,[&](Value j){
                Value value=b.create<arith::AddFOp>(loc,load(result,{p,j}),load(bias,{j}));
                writeFloat(relu(value,activation));
              });
            });
          });
        });
      });
    });
  }
};

class GenerateReusableOperator : public impl::GenerateReusableOperatorBase<GenerateReusableOperator> {
public:
  using GenerateReusableOperatorBase::GenerateReusableOperatorBase;
  void runOnOperation() final {
    auto module=getOperation();
    if((kind!="conv" && kind!="add") || pixels<1 || pixels>128 || lanes<1 || lanes>32 ||
       channels<1 || channels>128 || blockElements<1 || blockElements>4096 || !module.getBody()->empty()) {
      module.emitError("expected empty module, kind conv/add, and bounded positive implementation parameters");
      return signalPassFailure();
    }
    OpBuilder b(module.getContext());
    b.setInsertionPointToEnd(module.getBody());
    auto ctrl=StreamType::get(module.getContext(),b.getI64Type(),8,StreamInterface::AXIS);
    auto data=StreamType::get(module.getContext(),b.getI32Type(),16,StreamInterface::AXIS);
    auto type=b.getFunctionType({ctrl,data,ctrl,data},{});
    auto engine=b.create<func::FuncOp>(module.getLoc(),kind+"_engine",type);
    b.setInsertionPointToStart(engine.addEntryBlock());
    Generator generator(b,module.getLoc(),engine.getArguments());
    if(kind=="conv") generator.convolution(pixels,lanes,channels);
    else generator.addition(blockElements);
    b.create<func::ReturnOp>(module.getLoc());
    b.setInsertionPointToEnd(module.getBody());
    auto top=b.create<func::FuncOp>(module.getLoc(),kind+"_kernel",type);
    b.setInsertionPointToStart(top.addEntryBlock());
    b.create<func::CallOp>(module.getLoc(),engine,top.getArguments());
    b.create<func::ReturnOp>(module.getLoc());
    module->setAttr("sodap.operator_abi",b.getI32IntegerAttr(OP_VERSION));
  }
};
} // namespace
} // namespace mlir::sodap
