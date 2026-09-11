//===- ConvertDataflowToLLVM.cpp - Dataflow streams to Bambu ac_channel ---===//
//
// Copyright (c) 2026 Tommaso Fellegara
//
//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers the dataflow stream ops to the ac_channel ABI that Bambu recognizes.
//
// Nothing about that ABI is hard-coded here. The pass writes a small C++
// translation unit that instantiates ac_channel<T> for every element type the
// module uses, compiles it with Clang, and reads the answers back out of the
// generated IR:
//
//   - an anchor probe holds the channel in a local, whose alloca carries the
//     real llvm::StructType of ac_channel<T>;
//   - one probe per operation gives its FunctionType, plus the mangled callee
//     for the constructor and the destructor. Reads and writes call the probe
//     itself, so the mangled seam is only ever named inside the Clang module.
//
// Clang is the authority for C++ layout and mangling; this pass only consumes
// the result. The alternative -- spelling out Itanium mangling and assuming a
// one-byte channel object -- silently rotted the moment ac_channel.h dropped
// its nested `fifo` class and started sending plain payloads as themselves.
//
// The read/write wrappers are always_inline and the consumer has to run
// `opt -passes=always-inline` after the link: Bambu does not inline them
// itself, and an ac_channel access it sees as an ordinary call becomes a
// module to synthesise rather than a FIFO port.
//
// The generated IR is left in the working directory as
// ac_channel_specializations.ll: llvm-link it into the translated kernel to
// get a single module that carries the constructor bodies and the read/write
// wrappers, while leaving the seams the wrappers call undefined, which is
// exactly what Bambu's InterfaceInfer replaces with FIFO ports.
//
//===----------------------------------------------------------------------===//

#include "sodap/Conversion/DataflowToLLVM/BambuTypeNames.h"
#include "sodap/Conversion/DataflowToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace sodap {
#define GEN_PASS_DEF_CONVERTDATAFLOWTOLLVM
#include "sodap/Conversion/DataflowToLLVM/Passes.h.inc"
} // namespace sodap
} // namespace mlir

using namespace mlir;
using namespace mlir::sodap;
using namespace mlir::sodap::dataflow;

#define DEBUG_TYPE "sodap-convert-dataflow-to-llvm"

namespace {

/// The specialization unit and the IR Clang makes of it, both dropped in the
/// working directory. Not options: the consumer has to llvm-link the .ll
/// anyway, so a knob for where it lands would only duplicate `cd`.
static constexpr StringRef tuFileName = "ac_channel_specializations.cpp";
static constexpr StringRef irFileName = "ac_channel_specializations.ll";

/// Suffix of the extern "C" probe names for `type`. Has to be a valid C
/// identifier fragment and stable across runs.
static std::string stableId(Type type) {
  std::string id;
  llvm::raw_string_ostream os(id);
  os << type;
  return os.str();
}

/// Everything the lowering needs to know about one ac_channel<T>, all of it
/// read back from the module Clang produced.
struct ChannelInfo {
  Type objectType;

  std::string ctorSymbol;
  LLVM::LLVMFunctionType ctorType;
  std::string dtorSymbol;
  LLVM::LLVMFunctionType dtorType;
  std::string readSymbol;
  LLVM::LLVMFunctionType readType;
  std::string writeSymbol;
  LLVM::LLVMFunctionType writeType;

  unsigned alignment;
};

struct ConvertDataflowToLLVM
    : public sodap::impl::ConvertDataflowToLLVMBase<ConvertDataflowToLLVM> {
  using ConvertDataflowToLLVMBase::ConvertDataflowToLLVMBase;

  void runOnOperation() override;

private:
  /// Collects the distinct stream element types and rejects what this
  /// lowering does not model. Failure means a diagnostic was already emitted.
  LogicalResult collectSpecializations(SmallVectorImpl<Type> &elementTypes);

  /// Writes the specialization TU, compiles it into `irFileName`, and fills
  /// `channels` from the resulting IR.
  LogicalResult buildSpecializations(ArrayRef<Type> elementTypes);

  /// Declares `name` at the top of the module on first use.
  LLVM::LLVMFuncOp getOrInsertFunc(StringRef name, LLVM::LLVMFunctionType type);

  ModuleOp module;
  DenseMap<Type, ChannelInfo> channels;
};

//===----------------------------------------------------------------------===//
// Collecting what the module needs
//===----------------------------------------------------------------------===//

LogicalResult ConvertDataflowToLLVM::collectSpecializations(
    SmallVectorImpl<Type> &elementTypes) {
  SetVector<Type> seen;
  auto result = module.walk([&](Operation *op) {
    if (isa<DispatchOp, TaskOp, NodeOp>(op)) {
      op->emitError() << "'" << op->getName()
                      << "' must be lowered to func.func "
                         "(sodap-dataflow-nodes-to-func) before this pass";
      return WalkResult::interrupt();
    }

    Type streamType;
    if (auto streamOp = dyn_cast<StreamOp>(op))
      streamType = streamOp.getChannel().getType();
    else if (auto readOp = dyn_cast<StreamReadOp>(op))
      streamType = readOp.getChannel().getType();
    else if (auto writeOp = dyn_cast<StreamWriteOp>(op))
      streamType = writeOp.getChannel().getType();
    else
      return WalkResult::advance();

    auto elementType = cast<StreamType>(streamType).getElementType();
    if (bambuTypeName(elementType).empty()) {
      op->emitError() << "no ac_channel element type for " << elementType;
      return WalkResult::interrupt();
    }
    seen.insert(elementType);
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  elementTypes.assign(seen.begin(), seen.end());
  return success();
}

//===----------------------------------------------------------------------===//
// Asking Clang
//===----------------------------------------------------------------------===//

LogicalResult
ConvertDataflowToLLVM::buildSpecializations(ArrayRef<Type> elementTypes) {
  // 1. The translation unit. An anchor probe per T whose local gives the
  // object layout as an alloca -- a global would work too, but drags a
  // static initializer and an @llvm.global_ctors entry all the way into the
  // module handed to Bambu -- and one probe per operation, which is the
  // symbol the lowering calls: the mangled entry point is only ever named
  // inside this module. The read/write probes are always_inline so that a
  // later `opt -passes=always-inline` leaves Bambu the bare seam.
  // ac_channel_sim.h is deliberately not included: the read/write seams have
  // to stay undefined, which is what Bambu matches on.
  {
    std::error_code ec;
    llvm::raw_fd_ostream os(tuFileName, ec);
    if (ec)
      return module.emitError("cannot write ") << tuFileName;

    os << "// Generated by " DEBUG_TYPE ". Clang is the authority for the\n"
          "// ac_channel<T> layout and for the mangled entry points.\n"
          "#include <ac_channel.h>\n"
          "#include <cstdint>\n"
          "#include <new>\n\n";
    for (auto elementType : elementTypes) {
      auto id = stableId(elementType);
      auto bambuTypeNameStr = bambuTypeName(elementType);
      os << "using C_" << id << " = ac_channel<" << bambuTypeNameStr << ">;\n"
         << "extern \"C\" void __df_anchor_" << id << "() { C_" << id
         << " c; (void)c; }\n"
         << "extern \"C\" __attribute__((always_inline)) void __df_ctor_" << id
         << "(C_" << id << " *p) { new (p) C_" << id << "(); }\n"
         << "extern \"C\" __attribute__((always_inline)) void __df_dtor_" << id
         << "(C_" << id << " *p) { p->~C_" << id << "(); }\n"
         << "extern \"C\" __attribute__((always_inline)) " << bambuTypeNameStr
         << " __df_read_" << id << "(C_" << id << " *p) { return p->read(); }\n"
         << "extern \"C\" __attribute__((always_inline)) void __df_write_" << id
         << "(C_" << id << " *p, " << bambuTypeNameStr
         << " v) { p->write(v); }\n\n";
    }
  }

  // 2. Compile it. -O0 -fno-inline keeps the ctor/dtor probes readable; the
  // always_inline on read()/write() still fires, which is what exposes the
  // seam. -D__BAMBU__ keeps ac_int.h off its host branch (<iostream>,
  // <execinfo.h> and an ios_base_library_init module asm). -m32 is not a
  // knob: it has to match the ABI of the Bambu gold model, and a 64-bit
  // ac_channel layout would be read back here and then be wrong.
  auto clang = llvm::sys::findProgramByName(clangxx);
  if (!clang)
    return module.emitError("cannot find the C++ compiler '") << clangxx << "'";

  {
    std::string includeFlag = "-I" + includePandaPath;
    StringRef argv[] = {
        clangxx,       "-m32", includeFlag,  "-std=c++17", "-O0", "-fno-inline",
        "-D__BAMBU__", "-S",   "-emit-llvm", tuFileName,   "-o",  irFileName};

    std::string error;
    if (llvm::sys::ExecuteAndWait(*clang, argv, /*Env=*/std::nullopt,
                                  /*Redirects=*/{}, /*SecondsToWait=*/0,
                                  /*MemoryLimit=*/0, &error) != 0) {
      module.emitError("compiling the ac_channel specialization unit failed");
      if (!error.empty())
        module.emitRemark() << error;
      module.emitRemark() << "kept " << tuFileName << " for inspection";
      return failure();
    }
  }

  // 3. Read the answers back.
  // The context only has to outlive the translation below: every type we
  // keep is an MLIR one by the time this returns.
  llvm::LLVMContext llvmContext;
  llvm::SMDiagnostic diag;
  auto specModule = llvm::parseIRFile(irFileName, diag, llvmContext);
  if (!specModule)
    return module.emitError("cannot parse ") << irFileName;

  auto *context = &getContext();
  LLVM::TypeFromLLVMIRTranslator typeTranslator(*context);

  for (auto elementType : elementTypes) {
    auto id = stableId(elementType);
    ChannelInfo info;

    auto *anchor = specModule->getFunction("__df_anchor_" + id);
    llvm::AllocaInst *slot = nullptr;
    if (anchor && !anchor->isDeclaration())
      for (auto &inst : anchor->getEntryBlock())
        if ((slot = llvm::dyn_cast<llvm::AllocaInst>(&inst)))
          break;
    if (!slot)
      return module.emitError("no channel alloca in __df_anchor_")
             << id << " in " << irFileName;
    info.objectType = typeTranslator.translateType(slot->getAllocatedType());
    // Clang's own alignment for the local, not the one the data layout
    // derives: it models the tail padding of ac_channel<T> by packing the
    // struct, and getABITypeAlign answers 1 for anything packed.
    info.alignment = slot->getAlign().value();

    struct {
      StringRef probe;
      std::string *symbol;
      LLVM::LLVMFunctionType *type;
    } wanted[] = {
        {"__df_ctor_", &info.ctorSymbol, &info.ctorType},
        {"__df_dtor_", &info.dtorSymbol, &info.dtorType},
        {"__df_read_", &info.readSymbol, &info.readType},
        {"__df_write_", &info.writeSymbol, &info.writeType},
    };

    for (auto &entry : wanted) {
      auto probeName = (entry.probe + id).str();
      auto *probe = specModule->getFunction(probeName);
      if (!probe || probe->isDeclaration())
        return module.emitError("no body for ")
               << probeName << " in " << irFileName;

      *entry.symbol = probe->getName().str();
      *entry.type = cast<LLVM::LLVMFunctionType>(
          typeTranslator.translateType(probe->getFunctionType()));
    }

    channels.try_emplace(elementType, std::move(info));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Lowering
//===----------------------------------------------------------------------===//

LLVM::LLVMFuncOp
ConvertDataflowToLLVM::getOrInsertFunc(StringRef name,
                                       LLVM::LLVMFunctionType type) {
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return func;

  OpBuilder builder(&getContext());
  builder.setInsertionPointToStart(module.getBody());
  return builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
}

void ConvertDataflowToLLVM::runOnOperation() {
  module = getOperation();
  auto *context = &getContext();
  auto ptrType = LLVM::LLVMPointerType::get(context);

  // Both options default to the configured Bambu install, so this only
  // catches someone passing an empty one -- worth catching, because
  // findProgramByName asserts on an empty name rather than failing. MLIR
  // does not strip quotes from a pass option, so `clangxx=""` arrives as the
  // two-character string "" and is caught here too.
  if (clangxx.empty() || clangxx == "\"\"" || includePandaPath.empty() ||
      includePandaPath == "\"\"") {
    module.emitError() << "clangxx and include-panda-path cannot be empty";
    return signalPassFailure();
  }

  // Bail out before touching anything if the design uses constructs this
  // lowering does not model, rather than emitting silently wrong IR.
  SmallVector<Type> elementTypes;
  if (failed(collectSpecializations(elementTypes)))
    return signalPassFailure();
  if (!elementTypes.empty()) {
    if (failed(buildSpecializations(elementTypes)))
      return signalPassFailure();
  }

  SmallVector<Operation *> toErase;

  // Reads and writes go first: lowering the creation replaces the channel
  // value with the alloca, and after that the operand no longer carries the
  // element type. The calls built here take a value that is still a
  // !dataflow.stream; the creation below rewires them, and for a channel
  // that arrives as a function argument the retyping at the end does.
  module.walk([&](Operation *op) {
    OpBuilder builder(op);
    auto loc = op->getLoc();

    if (auto readOp = dyn_cast<StreamReadOp>(op)) {
      auto elementType =
          cast<StreamType>(readOp.getChannel().getType()).getElementType();
      auto &info = channels[elementType];
      auto call = builder.create<LLVM::CallOp>(
          loc, getOrInsertFunc(info.readSymbol, info.readType),
          ValueRange{readOp.getChannel()});
      // The result is optional: an absent one means the popped value is
      // dropped, but the call still has to happen.
      if (auto result = readOp.getResult()) {
        Value payload = call.getResult();
        // A plain type travels as itself, so this is normally a no-op; it
        // stays for the element types whose payload is narrower than their
        // object representation.
        if (payload.getType() != elementType)
          payload = builder.create<LLVM::BitcastOp>(loc, elementType, payload);
        result.replaceAllUsesWith(payload);
      }
      toErase.push_back(op);
      return;
    }

    if (auto writeOp = dyn_cast<StreamWriteOp>(op)) {
      auto elementType =
          cast<StreamType>(writeOp.getChannel().getType()).getElementType();
      auto &info = channels[elementType];
      auto func = getOrInsertFunc(info.writeSymbol, info.writeType);
      Value payload = writeOp.getValue();
      auto payloadType = func.getFunctionType().getParams().back();
      if (payload.getType() != payloadType)
        payload = builder.create<LLVM::BitcastOp>(loc, payloadType, payload);
      // The i1 "written" result is dropped: the FIFO is sized so the write
      // cannot fail, which is the same assumption the HLS C++ backend makes.
      builder.create<LLVM::CallOp>(loc, func,
                                   ValueRange{writeOp.getChannel(), payload});
      toErase.push_back(op);
      return;
    }
  });

  module.walk([&](StreamOp streamOp) {
    OpBuilder builder(streamOp);
    auto loc = streamOp.getLoc();
    auto &info = channels[cast<StreamType>(streamOp.getChannel().getType())
                              .getElementType()];

    auto one = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(context, 64), builder.getI64IntegerAttr(1));
    auto alloca = builder.create<LLVM::AllocaOp>(loc, ptrType, info.objectType,
                                                 one, info.alignment);
    builder.create<LLVM::CallOp>(
        loc, getOrInsertFunc(info.ctorSymbol, info.ctorType),
        ValueRange{alloca.getResult()});
    streamOp.getChannel().replaceAllUsesWith(alloca.getResult());

    // ponytail: the channel is destroyed at every return of the function
    // that created it, which covers the FIFOs the dataflow flow emits
    // (declared in the top, passed to the nodes). A FIFO that outlives its
    // function needs real ownership tracking.
    auto dtor = getOrInsertFunc(info.dtorSymbol, info.dtorType);
    streamOp->getParentOfType<func::FuncOp>().walk(
        [&](func::ReturnOp returnOp) {
          OpBuilder atReturn(returnOp);
          atReturn.create<LLVM::CallOp>(loc, dtor,
                                        ValueRange{alloca.getResult()});
        });

    toErase.push_back(streamOp);
  });

  for (auto *op : toErase)
    op->erase();

  // Retype the stream arguments and mark every pointer argument noalias:
  // Bambu serializes accesses to potentially-aliasing buffers, and the kernel
  // arguments are distinct buffers under this flow's assumptions.
  auto unitAttr = UnitAttr::get(context);
  auto noAlias = LLVM::LLVMDialect::getNoAliasAttrName();
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;

    SmallVector<Type> argTypes(func.getArgumentTypes());
    for (auto [index, arg] : llvm::enumerate(func.getArguments())) {
      if (isa<StreamType>(arg.getType())) {
        arg.setType(ptrType);
        argTypes[index] = ptrType;
      }
      if (isa<MemRefType>(arg.getType()) || arg.getType() == ptrType)
        func.setArgAttr(index, noAlias, unitAttr);
    }
    func.setType(FunctionType::get(context, argTypes,
                                   func.getFunctionType().getResults()));
  }
}

} // namespace
