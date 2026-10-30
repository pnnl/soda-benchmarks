// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --transform-preload-library=transform-library-paths=%S/../../../../../sb_cli/recipes/esp-ir/transform.mlir \
// RUN:   --transform-interpreter | FileCheck %s --implicit-check-not=esp_float2fixed_f32 --implicit-check-not=esp_fixed2float_f32 --implicit-check-not=linalg.batch_matmul --implicit-check-not=linalg.generic

// Apply the shipped CLI recipe, not a duplicate schedule. This checks recipe
// loading, pass names, conversion to loops, and the shared-buffer lifetime.
// It does not claim numerical accelerator validation or multi-offload support.
// CHECK-LABEL: func.func @recipe_gemm
// CHECK: %[[BUFFER:.*]] = call @esp_alloc_shared
// CHECK: arith.fptosi
// CHECK: call @esp_accel_write_reg
// CHECK: call @esp_accel_start
// CHECK: call @esp_accel_wait
// CHECK: arith.sitofp
// CHECK: call @esp_free_shared(%[[BUFFER]])
// CHECK: return
func.func @recipe_gemm(%A: memref<1x4x8xf32>, %B: memref<1x8x4xf32>,
                       %C: memref<1x4x4xf32>) {
  linalg.batch_matmul ins(%A, %B : memref<1x4x8xf32>, memref<1x8x4xf32>)
                      outs(%C : memref<1x4x4xf32>)
  return
}
