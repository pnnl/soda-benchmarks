// stream_peek lowers to a call to __df_peek_<T>, a wrapper around
// ac_channel<T>::peek() in the specialization unit.
//
// REQUIRES: panda

// RUN: mkdir -p %t && cd %t && mlir-opt %s \
// RUN:     --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-convert-dataflow-to-llvm)" \
// RUN:   | FileCheck --implicit-check-not=dataflow.stream %s
// RUN: FileCheck --check-prefix=CPP %s < %t/ac_channel_specializations.cpp

// CPP: extern "C" __attribute__((always_inline)) float __df_peek_f32(C_f32 *p) { return p->peek(); }

// CHECK-DAG: llvm.func @__df_peek_f32(!llvm.ptr) -> f32
// CHECK-DAG: llvm.func @__df_read_f32(!llvm.ptr) -> f32

// Peek then pop: both see the same front value.
// CHECK-LABEL: func.func @peek_then_read(
// CHECK:         llvm.call @__df_peek_f32(%{{.*}}) : (!llvm.ptr) -> f32
// CHECK:         llvm.call @__df_read_f32(%{{.*}}) : (!llvm.ptr) -> f32
func.func @peek_then_read(%c: !dataflow.stream<f32, 8>) -> f32 {
  %p = dataflow.stream_peek %c : (!dataflow.stream<f32, 8>) -> f32
  %r = dataflow.stream_read %c : (!dataflow.stream<f32, 8>) -> f32
  %s = arith.addf %p, %r : f32
  return %s : f32
}
