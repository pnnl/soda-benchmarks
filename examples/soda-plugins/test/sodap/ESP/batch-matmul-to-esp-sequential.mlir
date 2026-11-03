// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp,canonicalize)" | FileCheck %s
// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp{profile=true},canonicalize)" | FileCheck %s --check-prefix=PROF
// RUN: not mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp{marshal=ir})" 2>&1 | FileCheck %s --check-prefix=IR-ERROR

// Runtime marshalling frees each buffer before the next allocation.
// CHECK-LABEL: func.func @test_two_matmuls
// CHECK-NOT: linalg.batch_matmul
// CHECK: call @esp_alloc_shared
// CHECK: call @esp_float2fixed_f32
// CHECK: call @esp_float2fixed_f32
// CHECK-COUNT-7: call @esp_accel_write_reg
// CHECK: call @esp_accel_start
// CHECK: call @esp_accel_wait
// CHECK: call @esp_fixed2float_f32
// CHECK: call @esp_free_shared
// CHECK: call @esp_alloc_shared
// CHECK: call @esp_float2fixed_f32
// CHECK: call @esp_float2fixed_f32
// CHECK-COUNT-7: call @esp_accel_write_reg
// CHECK: call @esp_accel_start
// CHECK: call @esp_accel_wait
// CHECK: call @esp_fixed2float_f32
// CHECK: call @esp_free_shared
// CHECK: return
// PROF-LABEL: func.func @test_two_matmuls
// PROF-NOT: call @esp_prof_begin(%c4_i32)
// PROF: return
// IR-ERROR: error: marshal=ir supports at most one ESP offload per function; use marshal=runtime for sequential offloads
func.func @test_two_matmuls(%A: memref<1x16x32xf32>, %B: memref<1x32x16xf32>,
                           %C: memref<1x16x16xf32>, %D: memref<1x16x16xf32>)
    -> memref<1x16x16xf32> {
  linalg.batch_matmul ins(%A, %B : memref<1x16x32xf32>, memref<1x32x16xf32>)
                      outs(%C : memref<1x16x16xf32>)
  linalg.batch_matmul ins(%A, %B : memref<1x16x32xf32>, memref<1x32x16xf32>)
                      outs(%D : memref<1x16x16xf32>)
  return %D : memref<1x16x16xf32>
}
