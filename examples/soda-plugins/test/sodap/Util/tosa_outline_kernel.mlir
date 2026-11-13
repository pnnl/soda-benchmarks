// RUN: mlir-opt %s --split-input-file --verify-diagnostics --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-tosa-raise-transposes,sodap-tosa-fold-constant-transposes),sodap-tosa-outline-kernel)" | FileCheck %s

// atax's shape: %a is read as it is and transposed. The tile gets both, the
// host transposes.

// CHECK:       module @forward_kernel attributes {sodap.tile = 0 : i64, sodap.tile_dir = "tiles/forward_kernel"}
// CHECK:         func.func @forward_kernel(%[[A:.*]]: tensor<2x3xf32>, %[[AT:.*]]: tensor<3x2xf32>, %[[B:.*]]: tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK-NOT:       tosa.transpose
// CHECK:           tosa.add %[[AT]], %[[B]]
// CHECK:       func.func @forward(%[[HA:.*]]: tensor<2x3xf32>, %[[HB:.*]]: tensor<3x2xf32>) -> tensor<3x2xf32>
// CHECK:         %[[T:.*]] = tosa.transpose %[[HA]]
// CHECK:         %[[R:.*]] = dataflow.launch @forward_kernel::@forward_kernel(%[[HA]], %[[T]], %[[HB]])
// CHECK:         return %[[R]]
func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %p = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %r = tosa.reshape %a {new_shape = array<i64: 3, 2>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %t = tosa.transpose %a, %p : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  %s = tosa.add %t, %b : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  %o = tosa.add %s, %r : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  return %o : tensor<3x2xf32>
}

// -----

// trmm's shape: T(add(mul(a, m), c)). The transpose climbs to %a, the masks
// fold, the tile reads only the transposed %a.

// CHECK:       func.func @forward_kernel(%[[AT:.*]]: tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NOT:     tosa.transpose
// CHECK:         %[[M:.*]] = "tosa.const"() <{value = dense<{{\[\[}}1.000000e+00, 3.000000e+00], [2.000000e+00, 4.000000e+00]]>
// CHECK:         tosa.mul %[[AT]], %[[M]]
// CHECK:       func.func @forward(%[[A:.*]]: tensor<2x2xf32>)
// CHECK:         %[[T:.*]] = tosa.transpose %[[A]]
// CHECK:         dataflow.launch @forward_kernel::@forward_kernel(%[[T]])
func.func @forward(%a: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %m = "tosa.const"() <{value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
  %c = "tosa.const"() <{value = dense<[[0.0, 1.0], [0.0, 0.0]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
  %p = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %0 = tosa.mul %a, %m {shift = 0 : i8} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = tosa.add %0, %c : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %2 = tosa.transpose %1, %p : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// -----

// A transposed result: the tile returns it as computed, the host transposes.

// CHECK:       func.func @forward_kernel(%{{.*}}: tensor<1x2x3xf32>, %{{.*}}: tensor<1x3x4xf32>) -> tensor<2x4xf32>
// CHECK-NOT:     tosa.transpose
// CHECK:       func.func @forward(%[[HA:.*]]: tensor<1x2x3xf32>, %[[HB:.*]]: tensor<1x3x4xf32>) -> tensor<4x2xf32>
// CHECK:         %[[R:.*]] = dataflow.launch @forward_kernel::@forward_kernel(%[[HA]], %[[HB]])
// CHECK:         %[[T:.*]] = tosa.transpose %[[R]]
// CHECK:         return %[[T]]
func.func @forward(%a: tensor<1x2x3xf32>, %b: tensor<1x3x4xf32>) -> tensor<4x2xf32> {
  %p = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %0 = tosa.matmul %a, %b : (tensor<1x2x3xf32>, tensor<1x3x4xf32>) -> tensor<1x2x4xf32>
  %1 = tosa.reshape %0 {new_shape = array<i64: 2, 4>} : (tensor<1x2x4xf32>) -> tensor<2x4xf32>
  %2 = tosa.transpose %1, %p : (tensor<2x4xf32>, tensor<2xi32>) -> tensor<4x2xf32>
  return %2 : tensor<4x2xf32>
}

// -----

// A transpose between two ops that can't reach an argument.
func.func @forward(%a: tensor<1x2x3xf32>, %b: tensor<1x3x2xf32>) -> tensor<3x3xf32> {
  %p = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %0 = tosa.matmul %b, %a : (tensor<1x3x2xf32>, tensor<1x2x3xf32>) -> tensor<1x3x3xf32>
  %1 = tosa.reshape %0 {new_shape = array<i64: 3, 3>} : (tensor<1x3x3xf32>) -> tensor<3x3xf32>
  // expected-error@+1 {{transpose between two ops: the accelerator must not transpose}}
  %2 = tosa.transpose %1, %p : (tensor<3x3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
  %3 = tosa.add %2, %1 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
  return %3 : tensor<3x3xf32>
}

// -----

// A transposed constant with another use.
func.func @forward(%a: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %m = "tosa.const"() <{value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
  %p = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  // expected-error@+1 {{transpose of a constant that has other uses}}
  %t = tosa.transpose %m, %p : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  %0 = tosa.add %a, %m : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = tosa.add %0, %t : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
