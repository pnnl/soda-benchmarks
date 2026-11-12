// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-copy-to-dma{top-func=top},canonicalize)" | FileCheck %s

// A store acquires the producer's output token before publishing any DMA
// request and releases it after copying the data. Infer the barrier from the
// DMA destination, without kernel/shape-specific annotations in the input.
// CHECK-LABEL: func.func @top
// CHECK: dataflow.task
// CHECK: dataflow.stream_peek %{{.*}} {sodap.dma_ordered}
// CHECK: dataflow.stream_write
// CHECK: scf.for
// CHECK: memref.load
// CHECK: dataflow.stream_write
// CHECK: dataflow.stream_read
func.func @top(%out: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %plm = memref.alloc() : memref<2x4xf32>
  %ready = dataflow.stream {depth = 1 : i32} : <i8, 1>
  dataflow.dispatch {
    dataflow.task attributes {store} {
      %slot = dataflow.stream_peek %ready : (!dataflow.stream<i8, 1>) -> i8
      %index = arith.index_cast %slot : i8 to index
      %tile = memref.subview %plm[%index, 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
      memref.copy %tile, %out : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
      %released = dataflow.stream_read %ready : (!dataflow.stream<i8, 1>) -> i8
      dataflow.yield
    }
    dataflow.yield
  }
  return
}
