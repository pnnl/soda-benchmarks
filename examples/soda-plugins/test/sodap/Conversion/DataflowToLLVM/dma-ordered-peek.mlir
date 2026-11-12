// REQUIRES: panda
// RUN: mkdir -p %t
// RUN: cd %t && mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-convert-dataflow-to-llvm)" | FileCheck %s

// Waiting for a ready output tile must dominate the write request. Polling
// must retain the token, so the producer cannot overwrite the PLM during DMA.
// CHECK-LABEL: func.func @store
// CHECK: scf.while : () -> () {
// CHECK: %[[READY:.*]] = llvm.call @__df_ready_i8
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[AGAIN:.*]] = arith.xori %[[READY]], %[[TRUE]] : i1
// CHECK: scf.condition(%[[AGAIN]])
// CHECK: } do {
// CHECK: scf.yield
// CHECK: }
// CHECK-NOT: llvm.call @__df_read_i8
// CHECK: llvm.call @__df_peek_i8
// CHECK-NOT: llvm.call @__df_read_i8
// CHECK: llvm.call @__df_write_i64
// CHECK: llvm.call @__df_write_i32
// CHECK: llvm.call @__df_read_i8
func.func @store(%ready: !dataflow.stream<i8, 1>,
                 %ctrl: !dataflow.stream<i64, 4>,
                 %data: !dataflow.stream<i32, 1>, %request: i64, %value: i32) {
  %slot = dataflow.stream_peek %ready {sodap.dma_ordered} : (!dataflow.stream<i8, 1>) -> i8
  dataflow.stream_write %ctrl, %request : <i64, 4>, i64
  dataflow.stream_write %data, %value : <i32, 1>, i32
  %released = dataflow.stream_read %ready : (!dataflow.stream<i8, 1>) -> i8
  return
}

// Unmarked peeks keep their original lowering.
// CHECK-LABEL: func.func @ordinary
// CHECK-NOT: scf.while
// CHECK: llvm.call @__df_peek_i8
// CHECK: return
func.func @ordinary(%ready: !dataflow.stream<i8, 1>) -> i8 {
  %slot = dataflow.stream_peek %ready : (!dataflow.stream<i8, 1>) -> i8
  return %slot : i8
}
