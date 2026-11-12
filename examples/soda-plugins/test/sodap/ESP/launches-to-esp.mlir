// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-launches-to-esp)" -split-input-file -verify-diagnostics | FileCheck %s

// One tile: sodap-kernel-to-esp-accelerator's calls, on the host's arguments.
// The result's buffer, only copied into the out-param, is the out-param: the
// accelerator's output is copied out straight into it. Offsets: %a 0, %out 4.

// CHECK-NOT:   module @forward_kernel
// CHECK-LABEL: func.func @forward_kernel
// CHECK-SAME:    (%[[A:.*]]: memref<4xf32>, %[[OUT:.*]]: memref<2xf32>) {
// CHECK-NOT:     memref.alloc
// CHECK:         %[[BYTES:.*]] = arith.constant 24 : i64
// CHECK:         %[[H:.*]] = call @esp_alloc_shared(%[[BYTES]])
// CHECK:         %[[CA:.*]] = memref.cast %[[A]]
// CHECK:         call @esp_copy_to_shared_f32(%[[CA]], %[[H]], %{{.*}})
// CHECK:         %[[CO:.*]] = memref.cast %[[OUT]]
// CHECK:         %[[FOUR:.*]] = arith.constant 4 : i64
// CHECK:         call @esp_copy_to_shared_f32(%[[CO]], %[[H]], %[[FOUR]])
// CHECK:         call @esp_accel_start() : () -> ()
// CHECK:         call @esp_accel_wait() : () -> ()
// CHECK:         %[[CO2:.*]] = memref.cast %[[OUT]]
// CHECK:         call @esp_copy_from_shared_f32(%[[H]], %{{.*}}, %[[CO2]])
// CHECK-NOT:     esp_copy_from_shared_f32
// CHECK:         call @esp_free_shared(%[[H]])
// CHECK-NOT:     memref.copy
// CHECK:         return
module @forward_kernel attributes {sodap.tile = 0 : i64} {
  llvm.func @forward_kernel(!llvm.ptr)
}
func.func @main(%a: memref<4xf32>, %out: memref<2xf32>) {
  %r = memref.alloc() : memref<2xf32>
  dataflow.launch @forward_kernel::@forward_kernel(%a) outs(%r : memref<2xf32>) : (memref<4xf32>) -> ()
  memref.copy %r, %out : memref<2xf32> to memref<2xf32>
  return
}

// -----

// expected-error@+1 {{expected one tile module, got 2}}
module {
  module @f attributes {sodap.tile = 0 : i64} {
    llvm.func @f(!llvm.ptr)
  }
  module @g attributes {sodap.tile = 1 : i64} {
    llvm.func @g(!llvm.ptr)
  }
  func.func @main(%a: memref<4xf32>, %out: memref<4xf32>) {
    dataflow.launch @f::@f(%a) outs(%out : memref<4xf32>) : (memref<4xf32>) -> ()
    return
  }
}

// -----

func.func @main(%a: memref<?xf32>, %out: memref<4xf32>) {
  // expected-error@+1 {{expected f32 memrefs with static shapes and the identity layout, got 'memref<?xf32>'}}
  dataflow.launch @k::@k(%a) outs(%out : memref<4xf32>) : (memref<?xf32>) -> ()
  return
}
module @k attributes {sodap.tile = 0 : i64} {
  llvm.func @k(!llvm.ptr)
}
