//===- EmitBambuArchitecture.cpp - Bambu architecture XML -----------------===//
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
// Emits the XML that Bambu takes through --architecture-xml. It has to run
// while the dataflow stream types are still around: once the streams are
// pointers there is no way to tell a FIFO from an array.
//
// Two parameter protocols. Streams carry the same ac_channel<T> that
// ConvertDataflowToLLVM lowers them to, so the typename here and the template
// Clang instantiates there describe one type:
//
//   <bundle depth="32" mode="fifo" name="fifo_0"/>
//   <parameter bundle="fifo_0" index="2" port="fifo_0" size_in_bytes="4"
//              typename="ac_channel&lt;float&gt;&amp;" .../>
//
// Memrefs. The pointer is flat -- the lowering emits a linearized
// getelementptr -- but array_dims still carries the shape, one extent per
// dimension, while elem_count is the flat length:
//
//   <bundle mode="array" name="arg0"/>
//   <parameter bundle="arg0" index="0" port="arg0" typename="float*"
//              original_typename="float (*)[200]" array_dims="240,200"
//              elem_count="48000" size_in_bytes="192000"/>
//
// typename is the pointer the IR carries; original_typename is the parameter
// as it would have been declared, which for a C array is a pointer to its
// first row.
//
// size_in_bytes is the size of the data behind the parameter: the whole array
// for an array, one payload for a stream.
//
//===----------------------------------------------------------------------===//

#include "sodap/Conversion/DataflowToLLVM/BambuTypeNames.h"
#include "sodap/Conversion/DataflowToLLVM/Passes.h"

#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

namespace mlir {
namespace sodap {
#define GEN_PASS_DEF_EMITBAMBUARCHITECTURE
#include "sodap/Conversion/DataflowToLLVM/Passes.h.inc"
} // namespace sodap
} // namespace mlir

using namespace mlir;
using namespace mlir::sodap;
using namespace mlir::sodap::dataflow;

#define DEBUG_TYPE "sodap-emit-bambu-architecture"

namespace {

static std::string xmlEscape(StringRef in) {
  std::string out;
  for (char c : in) {
    switch (c) {
    case '&':
      out += "&amp;";
      break;
    case '<':
      out += "&lt;";
      break;
    case '>':
      out += "&gt;";
      break;
    case '"':
      out += "&quot;";
      break;
    default:
      out += c;
    }
  }
  return out;
}

/// The scalar every parameter bottoms out in: the payload of a stream, the
/// element of an array, or the argument itself.
static Type scalarTypeOf(Type type) {
  if (auto streamType = dyn_cast<StreamType>(type))
    return streamType.getElementType();
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return memrefType.getElementType();
  return type;
}

/// One <parameter> plus the <bundle> it belongs to.
struct ParamInfo {
  unsigned index;
  /// Name Bambu sees for this argument in the LLVM IR. Its AST plugin, which
  /// is what recovers the source-level names, does not run on LLVM IR input,
  /// so Bambu falls back to positional P<index> names and the XML has to
  /// match: anything else is rejected with "Not matched parameter name".
  std::string port;
  /// Name of the connection this parameter is wired to. Bambu names the RTL
  /// ports of the top after `port` but names the testbench interface modules
  /// after `bundle`, so on the top the two have to be the same string or the
  /// two sides fail to connect with "Port <bundle>_TDATA not found in DUT".
  /// Nothing forces them apart on a module, so they are kept equal there too.
  std::string bundle;
  /// "axis" on a stream of the top, "fifo" on an internal one, "array" for a
  /// memref, "default" for a scalar.
  std::string bundleMode;
  /// FIFO depth, straight from !dataflow.stream<T, N>. Emitted on the bundle
  /// of a stream and on nothing else.
  unsigned depth = 0;
  std::string typeName;
  /// The parameter as declared in C. Same as `typeName` except for an array
  /// of rank two or more, whose declaration decays to a pointer to its first
  /// row: "int32_t (*)[40]" for an int32_t[40][40].
  std::string originalTypeName;
  std::string includes;
  unsigned sizeInBytes = 0;
  /// Comma-separated shape of an array parameter, "8,6" for a memref<8x6xf32>.
  /// Empty for everything else, which is what marks a parameter as an array.
  std::string arrayDims;
  int64_t elemCount = 0;
};

struct EmitBambuArchitecture
    : public sodap::impl::EmitBambuArchitectureBase<EmitBambuArchitecture> {
  using EmitBambuArchitectureBase::EmitBambuArchitectureBase;

  void runOnOperation() override;

private:
  /// Describes every parameter of `func`.
  SmallVector<ParamInfo> describeParams(func::FuncOp func);

  std::string getBundleMode(BlockArgument arg);
  unsigned getDepth(BlockArgument arg);
  std::string getTypeName(BlockArgument arg);
  std::string getOriginalTypeName(BlockArgument arg);
  std::string getIncludes(BlockArgument arg);
  unsigned byteWidth(Type type);
  unsigned getSizeInBytes(BlockArgument arg);
  std::string getArrayDims(BlockArgument arg);
  int64_t getElemCount(BlockArgument arg);

  void emitFunction(llvm::raw_ostream &os, func::FuncOp func, bool isTop,
                    ArrayRef<ParamInfo> params);

  /// includePandaPath + "/ac_channel.h", built once in runOnOperation.
  std::string channelInclude;
};

std::string EmitBambuArchitecture::getBundleMode(BlockArgument arg)
{
  // TODO: the interface a parameter asks for is not in its type. The same
  // !dataflow.stream is an AXI-Stream port on the top, where Bambu wants
  // mode="axis", and a FIFO between two modules, where it wants mode="fifo",
  // so the guess below is wrong for every top-level stream. Read it from the
  // argument instead, once the attribute exists:
  //
  //   auto func = cast<func::FuncOp>(arg.getOwner()->getParentOp());
  //   auto index = arg.getArgNumber();
  //   if (auto port = func.getArgAttrOfType<PortAttr>(index, "dataflow.port"))
  //     return bambuModeName(port.getKind());   // STREAM -> "axis", MM -> "m_axi"
  //
  // and keep what follows as the fallback for an argument that carries none.
  Type type = arg.getType();
  return isa<StreamType>(type) ? "fifo"  :
         isa<MemRefType>(type) ? "array" :
         "default";
}

unsigned EmitBambuArchitecture::getDepth(BlockArgument arg)
{
  if(auto streamTy = dyn_cast<StreamType>(arg.getType()))
  {
    return streamTy.getDepth();
  }
  return 0;
}

std::string EmitBambuArchitecture::getTypeName(BlockArgument arg)
{
  Type type = arg.getType();
  auto scalarName = bambuTypeName(scalarTypeOf(type));
  if(isa<StreamType>(type))
  {
    // The very template ConvertDataflowToLLVM instantiates for this stream.
    return "ac_channel<" + scalarName + ">&";
  }
  if(isa<MemRefType>(type))
  {
    // The lowering linearizes the indexing, so what the IR carries is a
    // pointer to the first element whatever the rank.
    return scalarName + "*";
  }
  return scalarName;
}

std::string EmitBambuArchitecture::getOriginalTypeName(BlockArgument arg)
{
  // Only an array of rank two or more is declared differently from the
  // pointer the IR carries: "int32_t (*)[16]" for an int32_t[2][16].
  auto memrefTy = dyn_cast<MemRefType>(arg.getType());
  if(!memrefTy || memrefTy.getRank() < 2)
  {
    return getTypeName(arg);
  }

  std::string name = bambuTypeName(memrefTy.getElementType()) + " (*)";
  llvm::raw_string_ostream rows(name);
  for(auto extent : memrefTy.getShape().drop_front())
  {
    rows << "[" << extent << "]";
  }
  return name;
}

std::string EmitBambuArchitecture::getIncludes(BlockArgument arg)
{
  // Bambu has to see the ac_channel declaration to interface a stream; an
  // array or a scalar needs no header.
  return isa<StreamType>(arg.getType()) ? channelInclude : "";
}

unsigned EmitBambuArchitecture::byteWidth(Type type) {
  return (type.getIntOrFloatBitWidth() + 7) / 8;
}

unsigned EmitBambuArchitecture::getSizeInBytes(BlockArgument arg)
{
  // The data behind the parameter: the whole array for an array, one payload
  // for a stream or a scalar.
  auto scalarBytes = byteWidth(scalarTypeOf(arg.getType()));
  auto elemCount = getElemCount(arg);
  return elemCount ? scalarBytes * elemCount : scalarBytes;
}

std::string EmitBambuArchitecture::getArrayDims(BlockArgument arg)
{
  auto memrefTy = dyn_cast<MemRefType>(arg.getType());
  if(!memrefTy)
  {
    return "";
  }

  // One extent per dimension, "2,16" for a memref<2x16xi32>.
  std::string dims;
  llvm::raw_string_ostream os(dims);
  llvm::interleave(memrefTy.getShape(), os, ",");
  return dims;
}

int64_t EmitBambuArchitecture::getElemCount(BlockArgument arg)
{
  // Flat length, the product of the extents. Zero marks a parameter that is
  // not an array, and is what keeps elem_count out of its <parameter>.
  auto memrefTy = dyn_cast<MemRefType>(arg.getType());
  return memrefTy ? memrefTy.getNumElements() : 0;
}

SmallVector<ParamInfo> EmitBambuArchitecture::describeParams(func::FuncOp func) {
  SmallVector<ParamInfo> params;

  for (auto [index, arg] : llvm::enumerate(func.getArguments())) {
    // Rejected here rather than in the helpers, which would otherwise have to
    // answer for a type Bambu cannot interface at all.
    if (bambuTypeName(scalarTypeOf(arg.getType())).empty()) {
      func.emitError() << "argument " << index << ": no Bambu typename for "
                       << scalarTypeOf(arg.getType());
      signalPassFailure();
      continue;
    }
    if (auto memrefType = dyn_cast<MemRefType>(arg.getType());
        memrefType && !memrefType.hasStaticShape()) {
      func.emitError() << "argument " << index
                       << " has a dynamic shape, which Bambu's array "
                          "protocol cannot describe";
      signalPassFailure();
      continue;
    }

    ParamInfo info;

    // Bambu recovers source-level parameter names through its AST plugin,
    // which does not run on LLVM IR input: what it sees here are the
    // positional names, and the XML has to agree. Port and bundle are the
    // same string because the top would not connect to its testbench
    // otherwise, and nothing asks them to differ on a module.
    info.index = index;
    info.bundle = info.port = ("P" + Twine(index)).str();
    info.bundleMode = getBundleMode(arg);
    info.depth = getDepth(arg);
    info.typeName = getTypeName(arg);
    info.originalTypeName = getOriginalTypeName(arg);
    info.includes = getIncludes(arg);
    info.sizeInBytes = getSizeInBytes(arg);
    info.arrayDims = getArrayDims(arg);
    info.elemCount = getElemCount(arg);

    params.push_back(info);
  }

  return params;
}

void EmitBambuArchitecture::emitFunction(llvm::raw_ostream &os,
                                         func::FuncOp func, bool isTop,
                                         ArrayRef<ParamInfo> params) {
  auto name = func.getName();
  // The generated functions carry plain C names, so the symbol Bambu has to
  // match in the IR is the name itself -- no C++ mangling to reproduce.
  os << "  <function " << (isTop ? "dataflow_top" : "dataflow_module")
     << "=\"1\" inline=\"off\" name=\"" << name << "\" symbol=\"" << name
     << "\">\n";

  os << "    <bundles>\n";
  for (const auto &param : params) {
    os << "      <bundle";
    if (param.bundleMode == "fifo")
      os << " depth=\"" << param.depth << "\"";
    os << " mode=\"" << param.bundleMode << "\" name=\"" << param.bundle
       << "\"></bundle>\n";
  }
  os << "    </bundles>\n";

  os << "    <parameters>\n";
  for (const auto &param : params) {
    os << "      <parameter";
    if (!param.arrayDims.empty())
      os << " array_dims=\"" << param.arrayDims << "\"";
    os << " bundle=\"" << param.bundle << "\"";
    if (!param.arrayDims.empty())
      os << " elem_count=\"" << param.elemCount << "\"";
    if (!param.includes.empty())
      os << " includes=\"" << param.includes << "\"";
    os << " index=\"" << param.index << "\"";
    os << " original_typename=\"" << xmlEscape(param.originalTypeName) << "\"";
    os << " port=\"" << param.port << "\"";
    os << " size_in_bytes=\"" << param.sizeInBytes << "\"";
    os << " typename=\"" << xmlEscape(param.typeName) << "\"";
    os << "></parameter>\n";
  }
  os << "    </parameters>\n";
  os << "  </function>\n";
}

void EmitBambuArchitecture::runOnOperation() {
  auto module = getOperation();

  // top-func is the one option with no default, and a pass option cannot be
  // marked required: MLIR configures a pass by feeding the option string to
  // cl::ProvidePositionalOption, which never runs cl's own required check, so
  // llvm::cl::Required on one is inert and a missing option arrives here
  // empty.
  if (topFunc.empty()) {
    module.emitError() << "option 'top-func' is required";
    return signalPassFailure();
  }
  if (includePandaPath.empty()) {
    module.emitError() << "option 'include-panda-path' cannot be empty";
    return signalPassFailure();
  }
  SmallString<128> header(includePandaPath);
  llvm::sys::path::append(header, "ac_channel.h");
  channelInclude = header.str().str();

  auto top = module.lookupSymbol<func::FuncOp>(topFunc);
  if (!top) {
    module.emitError() << "top function '" << topFunc << "' not found";
    return signalPassFailure();
  }

  // write-to-terminal wins over `file`: no file is opened at all.
  std::unique_ptr<llvm::raw_fd_ostream> file;
  if (!writeToTerminal) {
    std::error_code error;
    file = std::make_unique<llvm::raw_fd_ostream>(xmlFileName, error);
    if (error) {
      module.emitError() << "cannot open '" << xmlFileName << "'";
      return signalPassFailure();
    }
  }
  llvm::raw_ostream &os = writeToTerminal ? llvm::outs() : *file;

  os << "<?xml version=\"1.0\"?>\n<module>\n";
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;

    bool isTop = func.getName() == topFunc;
    emitFunction(os, func, isTop, describeParams(func));
  }
  os << "</module>\n";
}

} // namespace
