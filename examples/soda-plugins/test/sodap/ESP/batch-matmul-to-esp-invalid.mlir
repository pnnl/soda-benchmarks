// RUN: mlir-opt %s --split-input-file --verify-diagnostics \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp)"
// RUN: mlir-opt %s --split-input-file --verify-diagnostics \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp{marshal=ir})"

func.func @batch_two(%a: memref<2x4x8xf32>, %b: memref<2x8x4xf32>, %c: memref<2x4x4xf32>) {
  // expected-error @+1 {{ESP lowering requires a statically known batch size of 1 for every operand}}
  linalg.batch_matmul ins(%a, %b : memref<2x4x8xf32>, memref<2x8x4xf32>) outs(%c : memref<2x4x4xf32>)
  return
}

// -----

func.func @dynamic_batch(%a: memref<?x4x8xf32>, %b: memref<?x8x4xf32>, %c: memref<?x4x4xf32>) {
  // expected-error @+1 {{ESP lowering requires a statically known batch size of 1 for every operand}}
  linalg.batch_matmul ins(%a, %b : memref<?x4x8xf32>, memref<?x8x4xf32>) outs(%c : memref<?x4x4xf32>)
  return
}

// -----

func.func @dynamic_rhs_batch(%a: memref<1x4x8xf32>, %b: memref<?x8x4xf32>, %c: memref<1x4x4xf32>) {
  // expected-error @+1 {{ESP lowering requires a statically known batch size of 1 for every operand}}
  linalg.batch_matmul ins(%a, %b : memref<1x4x8xf32>, memref<?x8x4xf32>) outs(%c : memref<1x4x4xf32>)
  return
}

// -----

func.func @dynamic_output_batch(%a: memref<1x4x8xf32>, %b: memref<1x8x4xf32>, %c: memref<?x4x4xf32>) {
  // expected-error @+1 {{ESP lowering requires a statically known batch size of 1 for every operand}}
  linalg.batch_matmul ins(%a, %b : memref<1x4x8xf32>, memref<1x8x4xf32>) outs(%c : memref<?x4x4xf32>)
  return
}

// -----

func.func @wrong_element_type(%a: memref<1x4x8xf64>, %b: memref<1x8x4xf64>, %c: memref<1x4x4xf64>) {
  // expected-error @+1 {{ESP lowering requires rank-3 f32 memref operands}}
  linalg.batch_matmul ins(%a, %b : memref<1x4x8xf64>, memref<1x8x4xf64>) outs(%c : memref<1x4x4xf64>)
  return
}
