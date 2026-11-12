// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map})" -split-input-file -verify-diagnostics | FileCheck %s

// The launch reads its operands; its result becomes a buffer allocated before
// it and passed in outs, as the kernel's out-param.

module @k attributes {sodap.tile = 0 : i64} {
  func.func private @k(memref<4xf32>, memref<4xf32>)
}

// CHECK-LABEL: func.func @main
// CHECK-SAME:    (%[[A:.*]]: memref<4xf32>) -> memref<4xf32>
// CHECK:         %[[R:.*]] = memref.alloc() {{.*}}: memref<4xf32>
// CHECK:         dataflow.launch @k::@k(%[[A]]) outs(%[[R]] : memref<4xf32>) : (memref<4xf32>) -> ()
// CHECK:         return %[[R]]
func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
  %r = dataflow.launch @k::@k(%a) : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

func.func @main(%a: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{kernel '@nope::@nope' not found}}
  %r = dataflow.launch @nope::@nope(%a) : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
