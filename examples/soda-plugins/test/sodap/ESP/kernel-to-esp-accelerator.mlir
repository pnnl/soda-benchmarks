// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-kernel-to-esp-accelerator)" -split-input-file -verify-diagnostics | FileCheck %s

// Offsets follow the argument order: %alpha 0, %out 1, %a 7. %out is written through
// a subview, so it must be copied in as well: the second row is unchanged.

// CHECK-LABEL: func.func @forward_kernel
// CHECK-SAME:    (%[[ALPHA:.*]]: memref<f32>, %[[OUT:.*]]: memref<2x3xf32>, %[[A:.*]]: memref<2x3xf32>) {
// CHECK-NOT:     linalg
// CHECK:         %[[BYTES:.*]] = arith.constant 52 : i64
// CHECK:         %[[H:.*]] = call @esp_alloc_shared(%[[BYTES]])
// CHECK:         %[[UA:.*]] = memref.cast %[[ALPHA]] : memref<f32> to memref<*xf32>
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK:         call @esp_copy_to_shared_f32(%[[UA]], %[[H]], %[[ZERO]])
// CHECK:         %[[UOI:.*]] = memref.cast %[[OUT]] : memref<2x3xf32> to memref<*xf32>
// CHECK:         %[[ONEIN:.*]] = arith.constant 1 : i64
// CHECK:         call @esp_copy_to_shared_f32(%[[UOI]], %[[H]], %[[ONEIN]])
// CHECK:         %[[UB:.*]] = memref.cast %[[A]] : memref<2x3xf32> to memref<*xf32>
// CHECK:         %[[SEVEN:.*]] = arith.constant 7 : i64
// CHECK:         call @esp_copy_to_shared_f32(%[[UB]], %[[H]], %[[SEVEN]])
// CHECK-NEXT:    call @esp_accel_start()
// CHECK-NEXT:    call @esp_accel_wait()
// CHECK:         %[[UO:.*]] = memref.cast %[[OUT]] : memref<2x3xf32> to memref<*xf32>
// CHECK:         %[[ONE:.*]] = arith.constant 1 : i64
// CHECK:         call @esp_copy_from_shared_f32(%[[H]], %[[ONE]], %[[UO]])
// CHECK-NEXT:    call @esp_free_shared(%[[H]])
// CHECK-NEXT:    return
func.func @forward_kernel(%alpha: memref<f32>, %out: memref<2x3xf32>, %a: memref<2x3xf32>) {
  cf.br ^bb1
^bb1:
  %c0 = arith.constant 0 : index
  %s = memref.subview %out[0, 0] [1, 3] [1, 1] : memref<2x3xf32> to memref<1x3xf32, strided<[3, 1]>>
  %v = memref.load %alpha[] : memref<f32>
  linalg.fill ins(%v : f32) outs(%s : memref<1x3xf32, strided<[3, 1]>>)
  %x = memref.load %a[%c0, %c0] : memref<2x3xf32>
  return
}

// -----

// expected-error@+1 {{expected f32 memref arguments with static shapes and the identity layout}}
func.func @forward_kernel(%a: memref<?xf32>) {
  return
}

// -----

// expected-error@+1 {{no function with a body named 'forward_kernel'}}
module {
  func.func private @forward_kernel(memref<4xf32>)
}
