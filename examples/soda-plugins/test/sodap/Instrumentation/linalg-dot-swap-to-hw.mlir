// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-swap-op-to-hw)" |\
// RUN:   FileCheck %s

module {
  // CHECK: func.func private @sodaVectorDot(memref<8xf32>, memref<8xf32>, index, memref<f32>)
  // CHECK-LABEL: func.func @dot_example
  func.func @dot_example(%a: memref<8xf32>, %b: memref<8xf32>, %out: memref<f32>) {
    // CHECK-NOT: linalg.dot
    // CHECK: %[[LEN:.*]] = arith.constant 8 : index
    // CHECK: call @sodaVectorDot(%{{.*}}, %{{.*}}, %[[LEN]], %{{.*}}) : (memref<8xf32>, memref<8xf32>, index, memref<f32>) -> ()
    linalg.dot ins(%a, %b : memref<8xf32>, memref<8xf32>) outs(%out : memref<f32>)
    return
  }

  // A dynamically-shaped dot product is left untouched: the HW module is
  // modeled as a fixed-size vector engine.
  // CHECK-LABEL: func.func @dot_dynamic
  func.func @dot_dynamic(%a: memref<?xf32>, %b: memref<?xf32>, %out: memref<f32>) {
    // CHECK: linalg.dot
    linalg.dot ins(%a, %b : memref<?xf32>, memref<?xf32>) outs(%out : memref<f32>)
    return
  }
}
