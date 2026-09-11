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

// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-convert-dataflow-to-llvm{clangxx= include-panda-path=%sodap_bambu_include})" \
// RUN:     2> %t.err
// RUN: FileCheck --check-prefix=CLANGXX-EMPTY %s < %t.err

// CLANGXX-EMPTY: error: clangxx and include-panda-path cannot be empty

// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-convert-dataflow-to-llvm{clangxx=\"\" include-panda-path=%sodap_bambu_include})" \
// RUN:     2> %t.err
// RUN: FileCheck --check-prefix=CLANGXX-QUOTED %s < %t.err

// CLANGXX-QUOTED: error: clangxx and include-panda-path cannot be empty

// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-convert-dataflow-to-llvm{clangxx=%sodap_bambu_clangxx include-panda-path=})" \
// RUN:     2> %t.err
// RUN: FileCheck --check-prefix=INCLUDE-PANDA-PATH-EMPTY %s < %t.err

// INCLUDE-PANDA-PATH-EMPTY: error: clangxx and include-panda-path cannot be empty

// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-convert-dataflow-to-llvm{clangxx=%sodap_bambu_clangxx include-panda-path=\"\"})" \
// RUN:     2> %t.err
// RUN: FileCheck --check-prefix=INCLUDE-PANDA-PATH-QUOTED %s < %t.err

// INCLUDE-PANDA-PATH-QUOTED: error: clangxx and include-panda-path cannot be empty

func.func @top() {
  %c = dataflow.stream {depth = 4 : i32} : <f32, 4>
  %v = dataflow.stream_read %c : (!dataflow.stream<f32, 4>) -> f32
  return
}
