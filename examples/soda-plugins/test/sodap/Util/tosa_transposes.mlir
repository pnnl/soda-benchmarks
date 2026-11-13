// RUN: mlir-opt %s --split-input-file --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-tosa-sink-transposes))" | FileCheck %s --check-prefix=SINK
// RUN: mlir-opt %s --split-input-file --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-tosa-sink-transposes),sodap-tosa-hoist-transposes,func.func(sodap-tosa-sink-transposes),sodap-tosa-hoist-transposes,canonicalize)" | FileCheck %s --check-prefix=HOIST

// conv -> to NCHW -> ReLU -> to NHWC: the transpose goes below the clamp and
// cancels with the next one.
// SINK-LABEL: func.func @through_relu
// SINK-NOT: tosa.transpose
// SINK: %[[R:.*]] = tosa.clamp %arg0 {{.*}} -> tensor<1x30x30x16xf32>
// SINK-NOT: tosa.transpose
// SINK: return %[[R]]
func.func @through_relu(%y: tensor<1x30x30x16xf32>) -> tensor<1x30x30x16xf32> {
  %to_nchw = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %to_nhwc = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %a = tosa.transpose %y, %to_nchw : (tensor<1x30x30x16xf32>, tensor<4xi32>) -> tensor<1x16x30x30xf32>
  %r = tosa.clamp %a {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>
  %b = tosa.transpose %r, %to_nhwc : (tensor<1x16x30x30xf32>, tensor<4xi32>) -> tensor<1x30x30x16xf32>
  return %b : tensor<1x30x30x16xf32>
}

// -----

// Two operands transposed alike: one add, then one transpose.
// SINK-LABEL: func.func @both_transposed
// SINK: %[[S:.*]] = tosa.add %arg0, %arg1 : (tensor<1x8x8x4xf32>, tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32>
// SINK: %[[T:.*]] = tosa.transpose %[[S]], %{{.*}} -> tensor<1x4x8x8xf32>
// SINK: return %[[T]]
func.func @both_transposed(%x: tensor<1x8x8x4xf32>, %y: tensor<1x8x8x4xf32>) -> tensor<1x4x8x8xf32> {
  %p = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %a = tosa.transpose %x, %p : (tensor<1x8x8x4xf32>, tensor<4xi32>) -> tensor<1x4x8x8xf32>
  %b = tosa.transpose %y, %p : (tensor<1x8x8x4xf32>, tensor<4xi32>) -> tensor<1x4x8x8xf32>
  %s = tosa.add %a, %b : (tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
  return %s : tensor<1x4x8x8xf32>
}

// -----

// A per-channel constant, broadcast in NCHW, gets the inverse transpose: its
// channels move to the last dim (folded here: it only moves unit dims).
// SINK-LABEL: func.func @constant
// SINK: %[[C:.*]] = "tosa.const"() <{value = dense<{{\[\[\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]]]> : tensor<1x1x1x4xf32>}>
// SINK: %[[S:.*]] = tosa.add %arg0, %[[C]] : (tensor<1x8x8x4xf32>, tensor<1x1x1x4xf32>) -> tensor<1x8x8x4xf32>
// SINK: tosa.transpose %[[S]]
func.func @constant(%x: tensor<1x8x8x4xf32>) -> tensor<1x4x8x8xf32> {
  %p = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %bias = "tosa.const"() <{value = dense<[[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]]> : tensor<1x4x1x1xf32>}> : () -> tensor<1x4x1x1xf32>
  %a = tosa.transpose %x, %p : (tensor<1x8x8x4xf32>, tensor<4xi32>) -> tensor<1x4x8x8xf32>
  %s = tosa.add %a, %bias : (tensor<1x4x8x8xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x8x8xf32>
  return %s : tensor<1x4x8x8xf32>
}

// -----

// Left alone: two permutations, and an operand neither transposed nor
// constant.
// SINK-LABEL: func.func @unchanged
// SINK: tosa.transpose %arg0
// SINK: tosa.transpose %arg1
// SINK: tosa.add
// SINK: tosa.transpose %arg0
// SINK: tosa.mul
func.func @unchanged(%x: tensor<4x4x4x4xf32>, %y: tensor<4x4x4x4xf32>) -> (tensor<4x4x4x4xf32>, tensor<4x4x4x4xf32>) {
  %p = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %q = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %a = tosa.transpose %x, %p : (tensor<4x4x4x4xf32>, tensor<4xi32>) -> tensor<4x4x4x4xf32>
  %b = tosa.transpose %y, %q : (tensor<4x4x4x4xf32>, tensor<4xi32>) -> tensor<4x4x4x4xf32>
  %s = tosa.add %a, %b : (tensor<4x4x4x4xf32>, tensor<4x4x4x4xf32>) -> tensor<4x4x4x4xf32>
  %c = tosa.transpose %x, %p : (tensor<4x4x4x4xf32>, tensor<4xi32>) -> tensor<4x4x4x4xf32>
  %m = tosa.mul %c, %y {shift = 0 : i8} : (tensor<4x4x4x4xf32>, tensor<4x4x4x4xf32>) -> tensor<4x4x4x4xf32>
  return %s, %m : tensor<4x4x4x4xf32>, tensor<4x4x4x4xf32>
}

// -----

// A self-inverse permutation with a constant: the constant's transpose is no
// transpose to move, so the pattern stops after one step.
// SINK-LABEL: func.func @self_inverse
// SINK: tosa.transpose
// SINK: tosa.add %arg0
// SINK: tosa.transpose
// SINK-NOT: tosa.transpose
// SINK: return
func.func @self_inverse(%x: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %p = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %k = "tosa.const"() <{value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>}> : () -> tensor<3x2xf32>
  %a = tosa.transpose %x, %p : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  %s = tosa.add %a, %k : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  return %s : tensor<3x2xf32>
}

// -----

// Two blocks in the style of compile_modular.py: each transposes its input
// and weight in and its result out. Hoisted, each block works in NHWC and
// transposes nothing; in @main the pair between the blocks cancels, and
// only the model's input and output stay transposed.
// HOIST-LABEL: func.func private @block(%arg0: tensor<1x8x8x2xf32>, %arg1: tensor<2x3x3x2xf32>, %arg2: tensor<2xf32>) -> tensor<1x8x8x2xf32>
// HOIST-NOT: tosa.transpose
// HOIST: tosa.conv2d %arg0, %arg1, %arg2
// HOIST: tosa.clamp
// HOIST-NOT: tosa.transpose
// HOIST: return
// HOIST-LABEL: func.func @main
// HOIST: tosa.transpose %arg0
// HOIST: tosa.transpose %arg1
// HOIST: %[[B1:.*]] = call @block
// HOIST-NOT: tosa.transpose %[[B1]]
// HOIST: %[[B2:.*]] = call @block(%[[B1]]
// HOIST: tosa.transpose %[[B2]]
func.func private @block(%x: tensor<1x2x8x8xf32>, %w: tensor<2x2x3x3xf32>, %b: tensor<2xf32>) -> tensor<1x2x8x8xf32> {
  %to_nhwc = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %to_nchw = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %xt = tosa.transpose %x, %to_nhwc : (tensor<1x2x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x2xf32>
  %wt = tosa.transpose %w, %to_nhwc : (tensor<2x2x3x3xf32>, tensor<4xi32>) -> tensor<2x3x3x2xf32>
  %c = tosa.conv2d %xt, %wt, %b {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x8x8x2xf32>, tensor<2x3x3x2xf32>, tensor<2xf32>) -> tensor<1x8x8x2xf32>
  %ct = tosa.transpose %c, %to_nchw : (tensor<1x8x8x2xf32>, tensor<4xi32>) -> tensor<1x2x8x8xf32>
  %r = tosa.clamp %ct {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32>
  return %r : tensor<1x2x8x8xf32>
}

func.func @main(%x: tensor<1x2x8x8xf32>, %w: tensor<2x2x3x3xf32>, %b: tensor<2xf32>) -> tensor<1x2x8x8xf32> {
  %0 = call @block(%x, %w, %b) : (tensor<1x2x8x8xf32>, tensor<2x2x3x3xf32>, tensor<2xf32>) -> tensor<1x2x8x8xf32>
  %1 = call @block(%0, %w, %b) : (tensor<1x2x8x8xf32>, tensor<2x2x3x3xf32>, tensor<2xf32>) -> tensor<1x2x8x8xf32>
  return %1 : tensor<1x2x8x8xf32>
}

// -----

// A residual block: the input is read by the conv (transposed) and by the
// add (as is). The add gets the inverse transpose, both its operands are then
// transposed alike, and the second sink and hoist leave the block in NHWC.
// The 1x1 weight's transpose only moves unit dims: canonicalize makes it a
// reshape, the same bytes, and it stays.
// HOIST-LABEL: func.func private @residual(%arg0: tensor<1x8x8x2xf32>, %arg1: tensor<2x2x1x1xf32>, %arg2: tensor<2xf32>) -> tensor<1x8x8x2xf32>
// HOIST-NOT: tosa.transpose
// HOIST: tosa.reshape %arg1
// HOIST: %[[C:.*]] = tosa.conv2d %arg0
// HOIST: tosa.add {{.*}}%arg0{{.*}}%[[C]]
// HOIST-NOT: tosa.transpose
// HOIST: return
func.func private @residual(%x: tensor<1x2x8x8xf32>, %w: tensor<2x2x1x1xf32>, %b: tensor<2xf32>) -> tensor<1x2x8x8xf32> {
  %to_nhwc = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %to_nchw = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %xt = tosa.transpose %x, %to_nhwc : (tensor<1x2x8x8xf32>, tensor<4xi32>) -> tensor<1x8x8x2xf32>
  %wt = tosa.transpose %w, %to_nhwc : (tensor<2x2x1x1xf32>, tensor<4xi32>) -> tensor<2x1x1x2xf32>
  %c = tosa.conv2d %xt, %wt, %b {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x8x8x2xf32>, tensor<2x1x1x2xf32>, tensor<2xf32>) -> tensor<1x8x8x2xf32>
  %ct = tosa.transpose %c, %to_nchw : (tensor<1x8x8x2xf32>, tensor<4xi32>) -> tensor<1x2x8x8xf32>
  %s = tosa.add %x, %ct : (tensor<1x2x8x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32>
  return %s : tensor<1x2x8x8xf32>
}

func.func @use_residual(%x: tensor<1x2x8x8xf32>, %w: tensor<2x2x1x1xf32>, %b: tensor<2xf32>) -> tensor<1x2x8x8xf32> {
  %0 = call @residual(%x, %w, %b) : (tensor<1x2x8x8xf32>, tensor<2x2x1x1xf32>, tensor<2xf32>) -> tensor<1x2x8x8xf32>
  return %0 : tensor<1x2x8x8xf32>
}
