// ConvertDataflowToLLVM refuses an empty clangxx or include-panda-path before
// doing anything: findProgramByName asserts on an empty name rather than
// failing, and an empty include path would only surface later, as a Clang
// error about ac_channel.h.
//
// An option is empty when nothing follows the `=`, or when it is `""`: this
// MLIR does not strip quotes from a pass option, so `clangxx=""` arrives as the
// two-character string `""`, which the pass treats as empty too.
//
// REQUIRES: panda

// Run from a scratch directory, to prove that nothing was written to it.
// RUN: rm -rf %t && mkdir -p %t && cd %t
// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-convert-dataflow-to-llvm{clangxx=})" \
// RUN:     -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: clangxx and include-panda-path cannot be empty

// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-convert-dataflow-to-llvm{clangxx=\"\"})" \
// RUN:     -o /dev/null 2>&1 | FileCheck --check-prefix=QUOTED %s

// QUOTED: error: clangxx and include-panda-path cannot be empty

func.func @top() {
  %c = dataflow.stream {depth = 4 : i32} : <f32, 4>
  %v = dataflow.stream_read %c : (!dataflow.stream<f32, 4>) -> f32
  return
}
