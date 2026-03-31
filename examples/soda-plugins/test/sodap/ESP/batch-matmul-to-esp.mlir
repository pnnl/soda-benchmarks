// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp)" | \
// RUN:   FileCheck %s

// Verify that all ESP runtime functions are declared.
// CHECK-DAG: func.func private @esp_alloc_shared(i64) -> memref<*xi8>
// CHECK-DAG: func.func private @esp_free_shared(memref<*xi8>)
// CHECK-DAG: func.func private @esp_float2fixed_f32(memref<*xf32>, memref<*xi8>, i64)
// CHECK-DAG: func.func private @esp_fixed2float_f32(memref<*xi8>, i64, memref<*xf32>)
// CHECK-DAG: func.func private @esp_accel_cfg_regs(i64, i64, i64, i64, i64, i64, i64)
// CHECK-DAG: func.func private @esp_accel_start()
// CHECK-DAG: func.func private @esp_accel_wait()

// -----

// CHECK-LABEL: func.func @test_basic
// CHECK-NOT: linalg.batch_matmul
// CHECK: %[[A_UR:.*]] = memref.cast %arg0 : memref<2x4x8xf32> to memref<*xf32>
// CHECK: %[[B_UR:.*]] = memref.cast %arg1 : memref<2x8x4xf32> to memref<*xf32>
// CHECK: %[[C_UR:.*]] = memref.cast %arg2 : memref<2x4x4xf32> to memref<*xf32>
// Step 1: Allocate shared memory
// CHECK: %[[MEM:.*]] = call @esp_alloc_shared
// Step 2: Copy inputs (float -> fixed-point)
// CHECK: call @esp_float2fixed_f32(%[[A_UR]], %[[MEM]],
// CHECK: call @esp_float2fixed_f32(%[[B_UR]], %[[MEM]],
// Step 3: Configure registers
// CHECK: call @esp_accel_cfg_regs(
// Step 4: Start accelerator
// CHECK: call @esp_accel_start()
// Step 5: Wait for completion
// CHECK: call @esp_accel_wait()
// Step 6: Copy output (fixed-point -> float)
// CHECK: call @esp_fixed2float_f32(%[[MEM]],
// Step 7: Free shared memory
// CHECK: call @esp_free_shared(%[[MEM]])
// CHECK: return
func.func @test_basic(%A: memref<2x4x8xf32>, %B: memref<2x8x4xf32>,
                       %C: memref<2x4x4xf32>) -> memref<2x4x4xf32> {
  linalg.batch_matmul ins(%A, %B : memref<2x4x8xf32>, memref<2x8x4xf32>)
                      outs(%C : memref<2x4x4xf32>)
  return %C : memref<2x4x4xf32>
}

// Verify that two batch_matmul ops each get their own full call sequence,
// but the runtime functions are declared only once.
// CHECK-LABEL: func.func @test_two_matmuls
// CHECK-NOT: linalg.batch_matmul
// CHECK: call @esp_alloc_shared
// CHECK: call @esp_float2fixed_f32
// CHECK: call @esp_float2fixed_f32
// CHECK: call @esp_accel_cfg_regs
// CHECK: call @esp_accel_start
// CHECK: call @esp_accel_wait
// CHECK: call @esp_fixed2float_f32
// CHECK: call @esp_free_shared
// Second matmul:
// CHECK: call @esp_alloc_shared
// CHECK: call @esp_float2fixed_f32
// CHECK: call @esp_float2fixed_f32
// CHECK: call @esp_accel_cfg_regs
// CHECK: call @esp_accel_start
// CHECK: call @esp_accel_wait
// CHECK: call @esp_fixed2float_f32
// CHECK: call @esp_free_shared
// CHECK: return
func.func @test_two_matmuls(%A: memref<1x16x32xf32>, %B: memref<1x32x16xf32>,
                             %C: memref<1x16x16xf32>, %D: memref<1x16x16xf32>)
    -> memref<1x16x16xf32> {
  linalg.batch_matmul ins(%A, %B : memref<1x16x32xf32>, memref<1x32x16xf32>)
                      outs(%C : memref<1x16x16xf32>)
  linalg.batch_matmul ins(%A, %B : memref<1x16x32xf32>, memref<1x32x16xf32>)
                      outs(%D : memref<1x16x16xf32>)
  return %D : memref<1x16x16xf32>
}