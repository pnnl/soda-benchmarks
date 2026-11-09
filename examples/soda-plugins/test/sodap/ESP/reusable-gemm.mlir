// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline='builtin.module(sodap-reusable-gemm-to-esp,canonicalize,cse)' | FileCheck %s
// One arena, two invocations, no copy of H between them, and one device binding.
// CHECK-LABEL: func.func @forward_kernel
// CHECK-SAME: sodap.gemm_invocations = 2
// CHECK: %[[CTX:.*]] = call @esp_gemm_open
// CHECK-COUNT-3: call @esp_gemm_copy_in
// CHECK: call @esp_gemm_invoke
// CHECK-SAME: sodap.instance = 0
// CHECK-SAME: sodap.invocation = 0
// CHECK-NOT: call @esp_gemm_copy
// CHECK-NOT: call @esp_gemm_open
// CHECK: call @esp_gemm_invoke
// CHECK-SAME: sodap.instance = 0
// CHECK-SAME: sodap.invocation = 1
// CHECK-COUNT-2: call @esp_gemm_copy_out(%[[CTX]]
// CHECK: call @esp_gemm_close(%[[CTX]])
// CHECK-NEXT: return
func.func @forward_kernel(%a: memref<3x5xf32>, %b: memref<5x7xf32>,
                         %w: memref<7x2xf32>, %h: memref<3x7xf32>,
                         %y: memref<3x2xf32>) {
  %zero = arith.constant 0.0 : f32
  linalg.fill ins(%zero : f32) outs(%h : memref<3x7xf32>)
  linalg.matmul ins(%a, %b : memref<3x5xf32>, memref<5x7xf32>) outs(%h : memref<3x7xf32>)
  linalg.fill ins(%zero : f32) outs(%y : memref<3x2xf32>)
  linalg.matmul ins(%h, %w : memref<3x7xf32>, memref<7x2xf32>) outs(%y : memref<3x2xf32>)
  return
}
