// RUN: mlir-opt %s --split-input-file --verify-diagnostics \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline='builtin.module(sodap-reusable-gemm-to-esp)' -o /dev/null
// expected-error @+1 {{initial GEMM isolation requires static contiguous f32 arguments}}
func.func @forward_kernel(%a: memref<?x5xf32>) { return }
// -----
func.func @forward_kernel(%a: memref<3x5xf32>, %b: memref<5x7xf32>, %c: memref<3x7xf32>) {
  %one = arith.constant 1.0 : f32
  // expected-error @+1 {{reusable GEMM supports one zero initialization per output}}
  linalg.fill ins(%one : f32) outs(%c : memref<3x7xf32>)
  linalg.matmul ins(%a, %b : memref<3x5xf32>, memref<5x7xf32>) outs(%c : memref<3x7xf32>)
  return
}
// -----
func.func @forward_kernel(%a: memref<3x5xf32>, %b: memref<5x7xf32>, %c: memref<3x7xf32>) {
  // expected-error @+1 {{expected compatible GEMM shapes and a distinct zero-initialized output}}
  linalg.matmul ins(%a, %b : memref<3x5xf32>, memref<5x7xf32>) outs(%c : memref<3x7xf32>)
  return
}
