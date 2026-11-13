// RUN: mlir-opt %s --split-input-file --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-outline-operators,func.func(canonicalize))" | FileCheck %s

// threemm's shape: one tile per matmul, 2-D interfaces, the model's reshapes
// cancel on the host and the third tile reads the first two's results.

// CHECK:      module @matmul_0 attributes {sodap.operator = "matmul", sodap.tile = 0 : i64, sodap.tile_dir = "tiles/matmul_0"}
// CHECK:        func.func @matmul_0(%{{.*}}: tensor<4x6xf32>, %{{.*}}: tensor<6x5xf32>) -> tensor<4x5xf32>
// CHECK:          tosa.reshape
// CHECK:          tosa.matmul
// CHECK:      module @matmul_1 attributes {sodap.operator = "matmul", sodap.tile = 1 : i64
// CHECK:      module @matmul_2 attributes {sodap.operator = "matmul", sodap.tile = 2 : i64
// CHECK:      func.func @forward(%[[A:.*]]: tensor<4x6xf32>, %[[B:.*]]: tensor<6x5xf32>, %[[C:.*]]: tensor<5x8xf32>, %[[D:.*]]: tensor<8x7xf32>)
// CHECK-NEXT:   %[[E:.*]] = dataflow.launch @matmul_0::@matmul_0(%[[A]], %[[B]])
// CHECK-NEXT:   %[[F:.*]] = dataflow.launch @matmul_1::@matmul_1(%[[C]], %[[D]])
// CHECK-NEXT:   %[[G:.*]] = dataflow.launch @matmul_2::@matmul_2(%[[E]], %[[F]]) : (tensor<4x5xf32>, tensor<5x7xf32>) -> tensor<4x7xf32>
// CHECK-NEXT:   return %[[G]]
func.func @forward(%a: tensor<4x6xf32>, %b: tensor<6x5xf32>, %c: tensor<5x8xf32>, %d: tensor<8x7xf32>) -> tensor<4x7xf32> {
  %0 = tosa.reshape %a {new_shape = array<i64: 1, 4, 6>} : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
  %1 = tosa.reshape %b {new_shape = array<i64: 1, 6, 5>} : (tensor<6x5xf32>) -> tensor<1x6x5xf32>
  %2 = tosa.matmul %0, %1 : (tensor<1x4x6xf32>, tensor<1x6x5xf32>) -> tensor<1x4x5xf32>
  %3 = tosa.reshape %c {new_shape = array<i64: 1, 5, 8>} : (tensor<5x8xf32>) -> tensor<1x5x8xf32>
  %4 = tosa.reshape %d {new_shape = array<i64: 1, 8, 7>} : (tensor<8x7xf32>) -> tensor<1x8x7xf32>
  %5 = tosa.matmul %3, %4 : (tensor<1x5x8xf32>, tensor<1x8x7xf32>) -> tensor<1x5x7xf32>
  %6 = tosa.matmul %2, %5 : (tensor<1x4x5xf32>, tensor<1x5x7xf32>) -> tensor<1x4x7xf32>
  %7 = tosa.reshape %6 {new_shape = array<i64: 4, 7>} : (tensor<1x4x7xf32>) -> tensor<4x7xf32>
  return %7 : tensor<4x7xf32>
}

// -----

// Two matmuls with one signature: one tile, launched twice.

// CHECK:      module @matmul_0
// CHECK-NOT:  module @matmul_1
// CHECK:      func.func @forward
// CHECK-NEXT:   %[[T:.*]] = dataflow.launch @matmul_0::@matmul_0(%arg0, %arg1)
// CHECK-NEXT:   dataflow.launch @matmul_0::@matmul_0(%[[T]], %arg2)
func.func @forward(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = tosa.reshape %a {new_shape = array<i64: 1, 4, 4>} : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
  %1 = tosa.reshape %b {new_shape = array<i64: 1, 4, 4>} : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
  %2 = tosa.matmul %0, %1 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  %3 = tosa.reshape %c {new_shape = array<i64: 1, 4, 4>} : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
  %4 = tosa.matmul %2, %3 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  %5 = tosa.reshape %4 {new_shape = array<i64: 4, 4>} : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
  return %5 : tensor<4x4xf32>
}
