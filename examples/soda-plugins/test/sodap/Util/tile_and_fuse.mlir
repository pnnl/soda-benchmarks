// RUN: mlir-opt %s --split-input-file --verify-diagnostics --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>
#a = affine_map<(d0, d1, d2) -> (d0, d2)>
#b = affine_map<(d0, d1, d2) -> (d2, d1)>
#c = affine_map<(d0, d1, d2) -> (d0, d1)>

// gemm in one task: the epilogue (PP) produces the yielded value. i and j are
// tiled with the fill and the contraction (PPR) fused in; only the contraction
// gets a k loop. The nest writes the edge, the alloc_tensor in the dispatch.
// CHECK-LABEL: func.func @epilogue
// CHECK: %[[EDGE:.*]] = bufferization.alloc_tensor()
// CHECK: dataflow.task
// CHECK:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:   scf.for %{{.*}} step %[[C4]] iter_args(%{{.*}} = %[[EDGE]])
// CHECK:     scf.for %{{.*}} step %[[C4]]
// CHECK:       linalg.fill
// CHECK:       scf.for %{{.*}} step %[[C4]]
// CHECK:         linalg.generic {{.*}}"parallel", "parallel", "reduction"
// CHECK:       }
// CHECK:       linalg.generic {{.*}}iterator_types = ["parallel", "parallel"]
// CHECK-NOT: scf.for
func.func @epilogue(%A: tensor<8x12xf32>, %B: tensor<12x16xf32>, %C: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %zero = arith.constant 0.0 : f32
  %r = dataflow.dispatch : tensor<8x16xf32> {
    %edge = bufferization.alloc_tensor() : tensor<8x16xf32>
    %t = dataflow.task : tensor<8x16xf32> {
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
          ins(%mm, %C : tensor<8x16xf32>, tensor<8x16xf32>) outs(%edge : tensor<8x16xf32>) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %0 = arith.addf %x, %y : f32
        linalg.yield %0 : f32
      } -> tensor<8x16xf32>
      dataflow.yield %out : tensor<8x16xf32>
    }
    dataflow.yield %t : tensor<8x16xf32>
  }
  return %r : tensor<8x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %tasks = transform.structured.match ops{["dataflow.task"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.sodap.tile_and_fuse %tasks {max_tile_size = 4 : i64} : !transform.any_op
    transform.yield
  }
}

// -----

// trmm's two tasks: each gets its own nest, and a producer is fused only in
// its own task. The add is fused into the transpose's tiles, in the first
// task. The second task reads the first one's result, a row block per (i, j)
// (the window of the plan), and recomputes nothing of it: the first task is
// not a tileable op, so it isn't fused (LLVM 19 would leave a copy of it in
// the loop).
// CHECK-LABEL: func.func @two_tasks
// CHECK: %[[T:.*]] = dataflow.task
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       linalg.add
// CHECK:       linalg.transpose
// CHECK: dataflow.task
// CHECK-NOT: dataflow.task
// CHECK-NOT: linalg.add
// CHECK-NOT: linalg.transpose
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       tensor.extract_slice %[[T]][%{{.*}}, 0] [4, 8] [1, 1]
// CHECK:       linalg.fill
// CHECK:       scf.for
// CHECK:         linalg.matmul
func.func @two_tasks(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %zero = arith.constant 0.0 : f32
  %r = dataflow.dispatch : tensor<8x8xf32> {
    %t_buf = bufferization.alloc_tensor() : tensor<8x8xf32>
    %t = dataflow.task : tensor<8x8xf32> {
      %e = tensor.empty() : tensor<8x8xf32>
      %low = linalg.add ins(%A, %A : tensor<8x8xf32>, tensor<8x8xf32>) outs(%e : tensor<8x8xf32>) -> tensor<8x8xf32>
      %tr = linalg.transpose ins(%low : tensor<8x8xf32>) outs(%t_buf : tensor<8x8xf32>) permutation = [1, 0]
      dataflow.yield %tr : tensor<8x8xf32>
    }
    %o_buf = bufferization.alloc_tensor() : tensor<8x8xf32>
    %o = dataflow.task : tensor<8x8xf32> {
      %f = linalg.fill ins(%zero : f32) outs(%o_buf : tensor<8x8xf32>) -> tensor<8x8xf32>
      %mm = linalg.matmul ins(%t, %B : tensor<8x8xf32>, tensor<8x8xf32>) outs(%f : tensor<8x8xf32>) -> tensor<8x8xf32>
      dataflow.yield %mm : tensor<8x8xf32>
    }
    dataflow.yield %o : tensor<8x8xf32>
  }
  return %r : tensor<8x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %tasks = transform.structured.match ops{["dataflow.task"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.sodap.tile_and_fuse %tasks {max_tile_size = 4 : i64} : !transform.any_op
    transform.yield
  }
}

// -----

#m = affine_map<(d0, d1) -> (d0, d1)>
#v = affine_map<(d0, d1) -> (d1)>
#o = affine_map<(d0, d1) -> (d0)>

// bicg: a task yielding two values, of another rank, each with a nest of its
// own.
// CHECK-LABEL: func.func @two_results
// CHECK: dataflow.task
// CHECK:   scf.for
// CHECK:     linalg.fill
// CHECK:     scf.for
// CHECK:       linalg.generic {{.*}}"parallel", "reduction"
// CHECK:   scf.for
// CHECK:     linalg.fill
// CHECK:     scf.for
// CHECK:       linalg.generic {{.*}}"parallel", "reduction"
// CHECK: return
func.func @two_results(%A: tensor<8x12xf32>, %x: tensor<12xf32>, %y: tensor<12xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %zero = arith.constant 0.0 : f32
  %r:2 = dataflow.dispatch : tensor<8xf32>, tensor<8xf32> {
    %e0 = bufferization.alloc_tensor() : tensor<8xf32>
    %e1 = bufferization.alloc_tensor() : tensor<8xf32>
    %t:2 = dataflow.task : tensor<8xf32>, tensor<8xf32> {
      %f0 = linalg.fill ins(%zero : f32) outs(%e0 : tensor<8xf32>) -> tensor<8xf32>
      %r0 = linalg.generic {indexing_maps = [#m, #v, #o], iterator_types = ["parallel", "reduction"]}
          ins(%A, %x : tensor<8x12xf32>, tensor<12xf32>) outs(%f0 : tensor<8xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %0 = arith.mulf %a, %b : f32
        %1 = arith.addf %acc, %0 : f32
        linalg.yield %1 : f32
      } -> tensor<8xf32>
      %f1 = linalg.fill ins(%zero : f32) outs(%e1 : tensor<8xf32>) -> tensor<8xf32>
      %r1 = linalg.generic {indexing_maps = [#m, #v, #o], iterator_types = ["parallel", "reduction"]}
          ins(%A, %y : tensor<8x12xf32>, tensor<12xf32>) outs(%f1 : tensor<8xf32>) {
      ^bb0(%a: f32, %b: f32, %acc: f32):
        %0 = arith.mulf %a, %b : f32
        %1 = arith.addf %acc, %0 : f32
        linalg.yield %1 : f32
      } -> tensor<8xf32>
      dataflow.yield %r0, %r1 : tensor<8xf32>, tensor<8xf32>
    }
    dataflow.yield %t#0, %t#1 : tensor<8xf32>, tensor<8xf32>
  }
  return %r#0, %r#1 : tensor<8xf32>, tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %tasks = transform.structured.match ops{["dataflow.task"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.sodap.tile_and_fuse %tasks {max_tile_size = 4 : i64} : !transform.any_op
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
  %r = dataflow.dispatch : tensor<60x70xf32> {
    %edge = bufferization.alloc_tensor() : tensor<60x70xf32>
    %t = dataflow.task : tensor<60x70xf32> {
      %fill = linalg.fill ins(%zero : f32) outs(%edge : tensor<60x70xf32>) -> tensor<60x70xf32>
      %mm = linalg.generic {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%A, %B : tensor<60x80xf32>, tensor<80x70xf32>) outs(%fill : tensor<60x70xf32>) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %0 = arith.mulf %x, %y : f32
        %1 = arith.addf %acc, %0 : f32
        linalg.yield %1 : f32
      } -> tensor<60x70xf32>
      dataflow.yield %mm : tensor<60x70xf32>
    }
    dataflow.yield %t : tensor<60x70xf32>
  }
  return %r : tensor<60x70xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %tasks = transform.structured.match ops{["dataflow.task"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.sodap.tile_and_fuse %tasks {max_tile_size = 16 : i64} : !transform.any_op
    transform.yield
  }
}

// -----

// A yielded value that no linalg op produces is an error, not a nest left
// untiled.
func.func @not_linalg(%A: tensor<8xf32>) -> tensor<8xf32> {
  %r = dataflow.dispatch : tensor<8xf32> {
    %t = dataflow.task : tensor<8xf32> {
      // expected-note @below {{result}}
      %empty = tensor.empty() : tensor<8xf32>
      dataflow.yield %empty : tensor<8xf32>
    }
    dataflow.yield %t : tensor<8xf32>
  }
  return %r : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %tasks = transform.structured.match ops{["dataflow.task"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected a result produced by a linalg op with a parallel dim}}
    transform.sodap.tile_and_fuse %tasks {max_tile_size = 4 : i64} : !transform.any_op
    transform.yield
  }
}

// -----

// A dynamic loop range has no GCD: an error, not a partial tile.
func.func @dynamic(%A: tensor<?x8xf32>) -> tensor<?x8xf32> {
  %zero = arith.constant 0.0 : f32
  %r = dataflow.dispatch : tensor<?x8xf32> {
    %t = dataflow.task : tensor<?x8xf32> {
      // expected-note @below {{op}}
      %fill = linalg.fill ins(%zero : f32) outs(%A : tensor<?x8xf32>) -> tensor<?x8xf32>
      dataflow.yield %fill : tensor<?x8xf32>
    }
    dataflow.yield %t : tensor<?x8xf32>
  }
  return %r : tensor<?x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %tasks = transform.structured.match ops{["dataflow.task"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected static loop ranges}}
    transform.sodap.tile_and_fuse %tasks {max_tile_size = 4 : i64} : !transform.any_op
    transform.yield
  }
}

// -----

// Only tasks: a function is tiled through the tasks create_tasks makes.
func.func @function(%A: tensor<8xf32>) -> tensor<8xf32> {
  return %A : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected a dataflow.task in a func.func}}
    transform.sodap.tile_and_fuse %func {max_tile_size = 4 : i64} : !transform.any_op
    transform.yield
  }
}
