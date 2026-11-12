// REQUIRES: panda
// RUN: mkdir -p %t
// RUN: cd %t && mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-convert-dataflow-to-llvm)" | FileCheck %s

// A stalled publication must keep control in the retry loop. The subsequent
// DMA request and receive loop can only be reached after an accepted write.
// Unmarked writes elsewhere retain the existing blocking-channel lowering.
// CHECK-LABEL: func.func @load
// CHECK: scf.while : () -> () {
// CHECK:   %[[SENT:.*]] = llvm.call @__df_try_write_i8
// CHECK:   %[[TRUE:.*]] = arith.constant true
// CHECK:   %[[AGAIN:.*]] = arith.xori %[[SENT]], %[[TRUE]] : i1
// CHECK:   scf.condition(%[[AGAIN]])
// CHECK: } do {
// CHECK:   scf.yield
// CHECK: }
// CHECK: llvm.call @__df_write_i64
// CHECK: llvm.call @__df_read_i32
func.func @load(%ready: !dataflow.stream<i8, 1>,
                %ctrl: !dataflow.stream<i64, 4>,
                %data: !dataflow.stream<i32, 1>,
                %slot: i8, %request: i64) {
  dataflow.stream_write %ready, %slot {sodap.dma_ordered} : <i8, 1>, i8
  dataflow.stream_write %ctrl, %request : <i64, 4>, i64
  %v = dataflow.stream_read %data : (!dataflow.stream<i32, 1>) -> i32
  return
}
