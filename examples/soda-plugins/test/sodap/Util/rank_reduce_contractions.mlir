// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-linalg-rank-reduce-contractions))" | FileCheck %s

// A unit batch, as TOSA gives every matmul: batch_matmul becomes matmul.
// CHECK-LABEL: func @unit_batch
// CHECK-NOT:     linalg.batch_matmul
// CHECK:         linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<60x60xf32>, tensor<60x80xf32>) outs(%{{.*}} : tensor<60x80xf32>)
func.func @unit_batch(%a: tensor<1x60x60xf32>, %b: tensor<1x60x80xf32>, %c: tensor<1x60x80xf32>) -> tensor<1x60x80xf32> {
  %0 = linalg.batch_matmul ins(%a, %b : tensor<1x60x60xf32>, tensor<1x60x80xf32>) outs(%c : tensor<1x60x80xf32>) -> tensor<1x60x80xf32>
  return %0 : tensor<1x60x80xf32>
}

// No unit dim: left alone.
// CHECK-LABEL: func @no_unit_dim
// CHECK:         linalg.batch_matmul
func.func @no_unit_dim(%a: tensor<2x60x60xf32>, %b: tensor<2x60x80xf32>, %c: tensor<2x60x80xf32>) -> tensor<2x60x80xf32> {
  %0 = linalg.batch_matmul ins(%a, %b : tensor<2x60x60xf32>, tensor<2x60x80xf32>) outs(%c : tensor<2x60x80xf32>) -> tensor<2x60x80xf32>
  return %0 : tensor<2x60x80xf32>
}
