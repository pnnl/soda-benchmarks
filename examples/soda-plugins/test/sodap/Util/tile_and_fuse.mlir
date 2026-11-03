// RUN: mlir-opt %s --split-input-file --verify-diagnostics --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>
#a = affine_map<(d0, d1, d2) -> (d0, d2)>
#b = affine_map<(d0, d1, d2) -> (d2, d1)>
#c = affine_map<(d0, d1, d2) -> (d0, d1)>

// gemm: the epilogue (PP) produces the result. i and j are tiled with the fill
// and the contraction (PPR) fused in; only the contraction gets a k loop.
// CHECK-LABEL: func.func @epilogue
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: scf.for %{{.*}} step %[[C4]]
// CHECK:   scf.for %{{.*}} step %[[C4]]
// CHECK:     linalg.fill
// CHECK:     scf.for %{{.*}} step %[[C4]]
// CHECK:       linalg.generic {{.*}}"parallel", "parallel", "reduction"
// CHECK:     }
// CHECK:     linalg.generic {{.*}}iterator_types = ["parallel", "parallel"]
// CHECK-NOT: scf.for
func.func @epilogue(%A: tensor<8x12xf32>, %B: tensor<12x16xf32>, %C: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x16xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x16xf32>) -> tensor<8x16xf32>
  %mm = linalg.generic {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%A, %B : tensor<8x12xf32>, tensor<12x16xf32>) outs(%fill : tensor<8x16xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %0 = arith.mulf %x, %y : f32
    %1 = arith.addf %acc, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<8x16xf32>
  %out = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%mm, %C : tensor<8x16xf32>, tensor<8x16xf32>) outs(%empty : tensor<8x16xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %0 = arith.addf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<8x16xf32>
  return %out : tensor<8x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.sodap.tile_and_fuse %func {max_tile_size = 4 : i64} : !transform.any_op
    transform.yield
  }
}

// -----

#m = affine_map<(d0, d1) -> (d0, d1)>
#v = affine_map<(d0, d1) -> (d1)>
#o = affine_map<(d0, d1) -> (d0)>

// bicg: two results, of another rank, each with a nest of its own.
// CHECK-LABEL: func.func @two_results
// CHECK: scf.for
// CHECK:   linalg.fill
// CHECK:   scf.for
// CHECK:     linalg.generic {{.*}}"parallel", "reduction"
// CHECK: scf.for
// CHECK:   linalg.fill
// CHECK:   scf.for
// CHECK:     linalg.generic {{.*}}"parallel", "reduction"
// CHECK: return
func.func @two_results(%A: tensor<8x12xf32>, %x: tensor<12xf32>, %y: tensor<12xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %f0 = linalg.fill ins(%zero : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %r0 = linalg.generic {indexing_maps = [#m, #v, #o], iterator_types = ["parallel", "reduction"]}
      ins(%A, %x : tensor<8x12xf32>, tensor<12xf32>) outs(%f0 : tensor<8xf32>) {
  ^bb0(%a: f32, %b: f32, %acc: f32):
    %0 = arith.mulf %a, %b : f32
    %1 = arith.addf %acc, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<8xf32>
  %f1 = linalg.fill ins(%zero : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %r1 = linalg.generic {indexing_maps = [#m, #v, #o], iterator_types = ["parallel", "reduction"]}
      ins(%A, %y : tensor<8x12xf32>, tensor<12xf32>) outs(%f1 : tensor<8xf32>) {
  ^bb0(%a: f32, %b: f32, %acc: f32):
    %0 = arith.mulf %a, %b : f32
    %1 = arith.addf %acc, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<8xf32>
  return %r0, %r1 : tensor<8xf32>, tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.sodap.tile_and_fuse %func {max_tile_size = 4 : i64} : !transform.any_op
    transform.yield
  }
}

// -----

// A result that no linalg op produces is an error, not a nest left untiled.
func.func @not_linalg(%A: tensor<8xf32>) -> tensor<8xf32> {
  // expected-note @below {{result}}
  %empty = tensor.empty() : tensor<8xf32>
  return %empty : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected a result produced by a linalg op with a parallel dim}}
    transform.sodap.tile_and_fuse %func {max_tile_size = 4 : i64} : !transform.any_op
    transform.yield
  }
}

// -----

#a = affine_map<(d0, d1, d2) -> (d0, d2)>
#b = affine_map<(d0, d1, d2) -> (d2, d1)>
#c = affine_map<(d0, d1, d2) -> (d0, d1)>

// gemm SMALL: the loop ranges 60, 70, 80 have GCD 10, the largest divisor up
// to 16, so every loop is tiled by 10.
// CHECK-LABEL: func.func @gcd
// CHECK: %[[C10:.*]] = arith.constant 10 : index
// CHECK: scf.for %{{.*}} step %[[C10]]
// CHECK:   scf.for %{{.*}} step %[[C10]]
// CHECK:     linalg.fill
// CHECK:     scf.for %{{.*}} step %[[C10]]
// CHECK:       linalg.generic {{.*}}"parallel", "parallel", "reduction"
func.func @gcd(%A: tensor<60x80xf32>, %B: tensor<80x70xf32>) -> tensor<60x70xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<60x70xf32>
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<60x70xf32>) -> tensor<60x70xf32>
  %mm = linalg.generic {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%A, %B : tensor<60x80xf32>, tensor<80x70xf32>) outs(%fill : tensor<60x70xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %0 = arith.mulf %x, %y : f32
    %1 = arith.addf %acc, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<60x70xf32>
  return %mm : tensor<60x70xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.sodap.tile_and_fuse %func {max_tile_size = 16 : i64} : !transform.any_op
    transform.yield
  }
}

// -----

// A dynamic loop range has no GCD: an error, not a partial tile.
func.func @dynamic(%A: tensor<?x8xf32>) -> tensor<?x8xf32> {
  %zero = arith.constant 0.0 : f32
  // expected-note @below {{op}}
  %fill = linalg.fill ins(%zero : f32) outs(%A : tensor<?x8xf32>) -> tensor<?x8xf32>
  return %fill : tensor<?x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected static loop ranges}}
    transform.sodap.tile_and_fuse %func {max_tile_size = 4 : i64} : !transform.any_op
    transform.yield
  }
}
