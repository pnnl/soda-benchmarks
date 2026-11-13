// RUN: mlir-opt %s --split-input-file --verify-diagnostics --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>
#sc = affine_map<(d0, d1) -> ()>

// trmm's shape, with no groups: ops chained through their inits share a task
// (the fill and the matmul), every other op gets its own. Each edge is an
// alloc_tensor in the dispatch, right before the task writing it.
// CHECK-LABEL: func.func @default_partition
// CHECK-SAME:    (%[[AL:.*]]: tensor<f32>, %[[A:.*]]: tensor<4x4xf32>, %[[B:.*]]: tensor<4x4xf32>)
// CHECK:         %[[R:.*]] = dataflow.dispatch : tensor<4x4xf32> {
// CHECK:           %[[E0:.*]] = bufferization.alloc_tensor() : tensor<4x4xf32>
// CHECK:           %[[LOW:.*]] = dataflow.task attributes {compute} : tensor<4x4xf32> {
// CHECK:             linalg.add ins(%[[A]], %[[A]] : {{.*}}) outs(%[[E0]] : {{.*}})
// CHECK:           %[[E1:.*]] = bufferization.alloc_tensor() : tensor<4x4xf32>
// CHECK:           %[[T:.*]] = dataflow.task attributes {compute} : tensor<4x4xf32> {
// CHECK:             linalg.transpose ins(%[[LOW]] : {{.*}}) outs(%[[E1]] : {{.*}})
// CHECK:           %[[E2:.*]] = bufferization.alloc_tensor() : tensor<4x4xf32>
// CHECK:           %[[MM:.*]] = dataflow.task attributes {compute} : tensor<4x4xf32> {
// CHECK:             %[[F:.*]] = linalg.fill {{.*}} outs(%[[E2]] : {{.*}})
// CHECK:             linalg.matmul ins(%[[T]], %[[B]] : {{.*}}) outs(%[[F]] : {{.*}})
// CHECK:           %[[E3:.*]] = bufferization.alloc_tensor() : tensor<4x4xf32>
// CHECK:           %[[O:.*]] = dataflow.task attributes {compute} : tensor<4x4xf32> {
// CHECK:             linalg.generic {{.*}} ins(%[[AL]], %[[MM]] : {{.*}}) outs(%[[E3]] : {{.*}})
// CHECK:           dataflow.yield %[[O]] : tensor<4x4xf32>
// CHECK-NOT:     tensor.empty
// CHECK:         return %[[R]]
func.func @default_partition(%alpha: tensor<f32>, %A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %e0 = tensor.empty() : tensor<4x4xf32>
  %low = linalg.add ins(%A, %A : tensor<4x4xf32>, tensor<4x4xf32>) outs(%e0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %e1 = tensor.empty() : tensor<4x4xf32>
  %t = linalg.transpose ins(%low : tensor<4x4xf32>) outs(%e1 : tensor<4x4xf32>) permutation = [1, 0]
  %e2 = tensor.empty() : tensor<4x4xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e2 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %mm = linalg.matmul ins(%t, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%f : tensor<4x4xf32>) -> tensor<4x4xf32>
  %e3 = tensor.empty() : tensor<4x4xf32>
  %o = linalg.generic {indexing_maps = [#sc, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%alpha, %mm : tensor<f32>, tensor<4x4xf32>) outs(%e3 : tensor<4x4xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %0 = arith.mulf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<4x4xf32>
  return %o : tensor<4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %d = transform.sodap.create_tasks %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>
#sc = affine_map<(d0, d1) -> ()>

// The same, grouped as in the plan: the transpose with its producer, the
// matmul (and its fill) with its consumer. The values that don't leave a task
// keep a tensor.empty in it.
// CHECK-LABEL: func.func @groups
// CHECK:         dataflow.dispatch : tensor<4x4xf32> {
// CHECK:           %[[E1:.*]] = bufferization.alloc_tensor() : tensor<4x4xf32>
// CHECK:           %[[T:.*]] = dataflow.task attributes {compute} : tensor<4x4xf32> {
// CHECK:             %[[P0:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK:             %[[LOW:.*]] = linalg.add {{.*}} outs(%[[P0]] : {{.*}})
// CHECK:             linalg.transpose ins(%[[LOW]] : {{.*}}) outs(%[[E1]] : {{.*}})
// CHECK:           %[[E3:.*]] = bufferization.alloc_tensor() : tensor<4x4xf32>
// CHECK:           %[[O:.*]] = dataflow.task attributes {compute} : tensor<4x4xf32> {
// CHECK:             %[[P2:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK:             %[[F:.*]] = linalg.fill {{.*}} outs(%[[P2]] : {{.*}})
// CHECK:             %[[MM:.*]] = linalg.matmul ins(%[[T]], %{{.*}} : {{.*}}) outs(%[[F]] : {{.*}})
// CHECK:             linalg.generic {{.*}} ins(%{{.*}}, %[[MM]] : {{.*}}) outs(%[[E3]] : {{.*}})
// CHECK:           dataflow.yield %[[O]] : tensor<4x4xf32>
// CHECK-NOT:     dataflow.task
// CHECK:         return
func.func @groups(%alpha: tensor<f32>, %A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %e0 = tensor.empty() : tensor<4x4xf32>
  %low = linalg.add ins(%A, %A : tensor<4x4xf32>, tensor<4x4xf32>) outs(%e0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %e1 = tensor.empty() : tensor<4x4xf32>
  %t = linalg.transpose ins(%low : tensor<4x4xf32>) outs(%e1 : tensor<4x4xf32>) permutation = [1, 0]
  %e2 = tensor.empty() : tensor<4x4xf32>
  %f = linalg.fill ins(%zero : f32) outs(%e2 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %mm = linalg.matmul ins(%t, %B : tensor<4x4xf32>, tensor<4x4xf32>) outs(%f : tensor<4x4xf32>) -> tensor<4x4xf32>
  %e3 = tensor.empty() : tensor<4x4xf32>
  %o = linalg.generic {indexing_maps = [#sc, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%alpha, %mm : tensor<f32>, tensor<4x4xf32>) outs(%e3 : tensor<4x4xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %0 = arith.mulf %x, %y : f32
    linalg.yield %0 : f32
  } -> tensor<4x4xf32>
  return %o : tensor<4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %tr = transform.structured.match ops{["linalg.transpose"]} in %func : (!transform.any_op) -> !transform.any_op
    %pre = transform.get_producer_of_operand %tr[0] : (!transform.any_op) -> !transform.any_op
    %mm = transform.structured.match ops{["linalg.matmul"]} in %func : (!transform.any_op) -> !transform.any_op
    %epi = transform.get_consumers_of_result %mm[0] : (!transform.any_op) -> !transform.any_op
    %g1 = transform.merge_handles %pre, %tr : !transform.any_op
    %g2 = transform.merge_handles %mm, %epi : !transform.any_op
    %d = transform.sodap.create_tasks %func groups(%g1, %g2) : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A group may not wrap another one: the first add and the last would be one
// task, needing the middle one, which needs the first.
func.func @cycle(%A: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %e0 = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.add ins(%A, %A : tensor<4x4xf32>, tensor<4x4xf32>) outs(%e0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %e1 = tensor.empty() : tensor<4x4xf32>
  %1 = linalg.mul ins(%0, %0 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%e1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %e2 = tensor.empty() : tensor<4x4xf32>
  %2 = linalg.sub ins(%1, %A : tensor<4x4xf32>, tensor<4x4xf32>) outs(%e2 : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %ops = transform.structured.match ops{["linalg.add", "linalg.sub"]} in %func : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected groups that don't depend on each other}}
    %d = transform.sodap.create_tasks %func groups(%ops) : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// An edge must be written into a tensor.empty, to be allocated in the
// dispatch: not into a function argument.
func.func @edge_into_argument(%A: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // expected-note @below {{op}}
  %0 = linalg.add ins(%A, %A : tensor<4x4xf32>, tensor<4x4xf32>) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected an edge written into a tensor.empty}}
    %d = transform.sodap.create_tasks %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Only linalg ops, tensor.empty and constants: a reshape between linalg ops
// would have to go in a task too.
func.func @unsupported_op(%A: tensor<4x4xf32>) -> tensor<16xf32> {
  %e0 = tensor.empty() : tensor<4x4xf32>
  %0 = linalg.add ins(%A, %A : tensor<4x4xf32>, tensor<4x4xf32>) outs(%e0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  // expected-note @below {{op}}
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<4x4xf32> into tensor<16xf32>
  return %1 : tensor<16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected only linalg ops on tensors, tensor.empty and constants}}
    %d = transform.sodap.create_tasks %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
