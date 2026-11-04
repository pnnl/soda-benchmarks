// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-copy-to-dma{top-func=top dma-word-stride=2},canonicalize)" | FileCheck %s

// Three logical words become six transport words. B starts at word 6. Each
// output value is followed by zero padding; each read consumes both lanes.
// CHECK-LABEL: func.func @top
// CHECK-DAG: %[[READ:.*]] = arith.constant 25769803776 : i64
// CHECK-DAG: %[[WRITE:.*]] = arith.constant 25769803782 : i64
// CHECK: dataflow.stream_write %{{.*}}, %[[READ]]
// CHECK: scf.for
// CHECK: dataflow.stream_read
// CHECK: dataflow.stream_read
// CHECK: memref.store
// CHECK: dataflow.stream_write %{{.*}}, %[[WRITE]]
// CHECK: scf.for
// CHECK: memref.load
// CHECK: dataflow.stream_write
// CHECK: dataflow.stream_write
func.func @top(%a: memref<3xf32>, %b: memref<3xf32>) {
  %buf = memref.alloc() : memref<3xf32>
  memref.copy %a, %buf : memref<3xf32> to memref<3xf32>
  memref.copy %buf, %b : memref<3xf32> to memref<3xf32>
  return
}
