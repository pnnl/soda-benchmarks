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

static unsigned byteWidth(Type type) {
  return (type.getIntOrFloatBitWidth() + 7) / 8;
}

/// One <parameter> plus the <bundle> it belongs to.
struct ParamInfo {
  unsigned index;
  /// Name Bambu sees for this argument in the LLVM IR. Its AST plugin, which
  /// is what recovers the source-level names, does not run on LLVM IR input,
  /// so Bambu falls back to positional P<index> names and the XML has to
  /// match: anything else is rejected with "Not matched parameter name".
  std::string port;
  /// Design-wide name of the connection. Bundle names are global, so two
  /// modules sharing a buffer must agree on it and two unrelated buffers must
  /// not collide -- which is why this is not the same string as `port`.
  std::string bundle;
  std::string bundleMode; // "fifo", "array", or "default" for scalars
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
  /// Describes every parameter of `func`. `ports` maps an argument index to
  /// the design-wide bundle name that argument was wired to; bundle names are
  /// global in Bambu, so two functions sharing a buffer must agree on it.
  SmallVector<ParamInfo>
  describeParams(func::FuncOp func,
                 const DenseMap<unsigned, std::string> &ports);

  void emitFunction(llvm::raw_ostream &os, func::FuncOp func, bool isTop,
                    ArrayRef<ParamInfo> params);

  /// includePandaPath + "/ac_channel.h", built once in runOnOperation.
  std::string channelInclude;
};

SmallVector<ParamInfo> EmitBambuArchitecture::describeParams(
    func::FuncOp func, const DenseMap<unsigned, std::string> &ports) {
  SmallVector<ParamInfo> params;

  for (auto [index, arg] : llvm::enumerate(func.getArguments())) {
    ParamInfo info;
    info.index = index;

    info.port = ("P" + Twine(index)).str();

    auto it = ports.find(index);
    if (it != ports.end())
      info.bundle = it->second;
    else
      // Values that never cross the top: scalar constants materialized in
      // the caller. Qualify the name so it cannot collide with another
      // module's.
      info.bundle = (func.getName() + "_" + info.port).str();

    // Every parameter bottoms out in a scalar: the payload of a stream, the
    // element of an array, or the argument itself.
    Type scalarType = arg.getType();
    if (auto streamType = dyn_cast<StreamType>(scalarType))
      scalarType = streamType.getElementType();
    else if (auto memrefType = dyn_cast<MemRefType>(scalarType))
      scalarType = memrefType.getElementType();

    auto scalarName = bambuTypeName(scalarType);
    if (scalarName.empty()) {
      func.emitError() << "argument " << index << ": no Bambu typename for "
                       << scalarType;
      signalPassFailure();
      continue;
    }
    info.sizeInBytes = byteWidth(scalarType);

    if (auto streamType = dyn_cast<StreamType>(arg.getType())) {
      info.bundleMode = "fifo";
      info.depth = streamType.getDepth();
      info.typeName = "ac_channel<" + scalarName + ">&";
      info.includes = channelInclude;
    } else if (auto memrefType = dyn_cast<MemRefType>(arg.getType())) {
      if (!memrefType.hasStaticShape()) {
        func.emitError() << "argument " << index
                         << " has a dynamic shape, which Bambu's array "
                            "protocol cannot describe";
        signalPassFailure();
        continue;
      }
      info.bundleMode = "array";
      info.typeName = scalarName + "*";
      info.elemCount = memrefType.getNumElements();
      info.sizeInBytes *= info.elemCount;

      auto shape = memrefType.getShape();
      llvm::raw_string_ostream dims(info.arrayDims);
      llvm::interleave(shape, dims, ",");

      if (shape.size() > 1) {
        info.originalTypeName = scalarName + " (*)";
        llvm::raw_string_ostream rows(info.originalTypeName);
        for (auto extent : shape.drop_front())
          rows << "[" << extent << "]";
      }
    } else {
      // Bambu wants a bundle for every parameter, scalars included: without
      // one it stops at "Missing parameter bundle name".
      info.bundleMode = "default";
      info.typeName = scalarName;
    }

    // Only a multidimensional array declares itself differently from the
    // pointer the IR carries.
    if (info.originalTypeName.empty())
      info.originalTypeName = info.typeName;

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

  // Bundle names live in one design-wide namespace: two modules connected by
  // the same buffer must name it identically, and two unrelated buffers must
  // not collide. So names are minted once, at the top, and propagated to the
  // callees through the call operands.
  DenseMap<Value, std::string> portNames;
  DenseMap<unsigned, std::string> topPorts;
  for (auto [index, arg] : llvm::enumerate(top.getArguments())) {
    auto name = ("arg" + Twine(index)).str();
    portNames[arg] = name;
    topPorts[index] = name;
  }
  // Internal FIFOs follow the "fifo_N" convention of the Vivado backend, so
  // the two backends agree on port names for the same design.
  unsigned fifoCounter = 0;
  top.walk([&](StreamOp streamOp) {
    portNames[streamOp.getChannel()] = ("fifo_" + Twine(fifoCounter++)).str();
  });

  DenseMap<StringRef, DenseMap<unsigned, std::string>> calleePorts;
  SmallVector<func::FuncOp> callees;
  top.walk([&](func::CallOp callOp) {
    auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (!callee)
      return;
    if (!llvm::is_contained(callees, callee))
      callees.push_back(callee);
    for (auto [index, operand] : llvm::enumerate(callOp.getOperands())) {
      auto it = portNames.find(operand);
      if (it != portNames.end())
        calleePorts[callee.getName()][index] = it->second;
    }
  });

  std::error_code error;
  llvm::raw_fd_ostream file(xmlFileName, error);
  if (error) {
    module.emitError() << "cannot open '" << xmlFileName << "'";
    return signalPassFailure();
  }

  file << "<?xml version=\"1.0\"?>\n<module>\n";
  emitFunction(file, top, /*isTop=*/true, describeParams(top, topPorts));
  for (auto callee : callees)
    emitFunction(file, callee, /*isTop=*/false,
                 describeParams(callee, calleePorts[callee.getName()]));
  file << "</module>\n";
}

} // namespace
