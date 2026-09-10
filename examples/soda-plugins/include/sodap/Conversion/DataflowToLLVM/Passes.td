//===- Passes.td - DataflowToLLVM passes -------------------*- tablegen -*-===//

#ifndef SODAP_CONVERSION_DATAFLOWTOLLVM_PASSES_TD
#define SODAP_CONVERSION_DATAFLOWTOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertDataflowToLLVM :
    Pass<"sodap-convert-dataflow-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Lower dataflow streams to Bambu ac_channel calls";
  let description = [{
    Rewrites !dataflow.stream<T, D> to !llvm.ptr and the stream ops to the
    ac_channel ABI Bambu recognizes. The layout of ac_channel<T> and the
    mangled ctor/dtor/read/write symbols are not spelled out here: the pass
    generates a C++ specialization unit, compiles it with Clang, and reads
    them back out of the generated IR. That IR is dropped in the working
    directory as ac_channel_specializations.ll; llvm-link it into the
    translated kernel to get a single module.

    Both options default to the Bambu install -DSODAP_BAMBU_ROOT named at
    configure time, through the generated sodap/Config.h.
  }];
  let options = [
    Option<"clangxx", "clangxx", "std::string",
           /*default=*/"SODAP_BAMBU_CLANGXX",
           "C++ compiler used to build the specialization unit">,
    Option<"includePandaPath", "include-panda-path", "std::string",
           /*default=*/"SODAP_BAMBU_INCLUDE",
           "Directory holding ac_channel.h and friends">
  ];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::LLVM::LLVMDialect"
  ];
}

def EmitBambuArchitecture :
    Pass<"sodap-emit-bambu-architecture", "::mlir::ModuleOp"> {
  let summary = "Emit the Bambu --architecture-xml description of the design";
  let description = [{
    Must run while the stream types are still present. Emits one <function>
    per node: the top as dataflow_top, every callee as dataflow_module, with a
    fifo bundle for every stream and an array bundle for every memref.
  }];
  let options = [
    Option<"xmlFileName", "file", "std::string",
           /*default=*/"\"architecture.xml\"", "Output file name">,
    Option<"topFunc", "top-func", "std::string", /*default=*/"",
           "Top function of the design (required)">,
    Option<"includePandaPath", "include-panda-path", "std::string",
           /*default=*/"SODAP_BAMBU_INCLUDE",
           "Directory holding ac_channel.h, named in the includes= of every "
           "stream parameter">
  ];
  let dependentDialects = ["::mlir::sodap::dataflow::DataflowDialect"];
}

#endif // SODAP_CONVERSION_DATAFLOWTOLLVM_PASSES_TD
