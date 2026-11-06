// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,buffer-results-to-out-params{hoist-static-allocs=true})" | FileCheck %s

// A load, two compute tasks and a store, on tensors. Every edge is a
// `bufferization.alloc_tensor` outside the task that writes it and becomes its
// own alloc there; a `tensor.empty` inside a task stays private to it. The
// store's destination is in the function, so it becomes the out-param: the
// store writes the output, with no copy after the dispatch.

// CHECK-LABEL: func.func @edges
// CHECK-SAME:    (%[[IN:.*]]: memref<4x4xf32>, %[[OUT:.*]]: memref<4x4xf32>)
// CHECK:         dataflow.dispatch {
// CHECK:           %[[A:.*]] = memref.alloc() {{.*}} : memref<4x4xf32>
// CHECK:           dataflow.task {
// CHECK:             linalg.copy ins(%[[IN]] : {{.*}}) outs(%[[A]] : {{.*}})
// CHECK:           %[[T:.*]] = memref.alloc() {{.*}} : memref<4x4xf32>
// CHECK:           dataflow.task {
// CHECK:             %[[TMP:.*]] = memref.alloc() {{.*}} : memref<4x4xf32>
// CHECK:             linalg.add ins(%[[A]], %[[A]] : {{.*}}) outs(%[[TMP]] : {{.*}})
// CHECK:             linalg.transpose ins(%[[TMP]] : {{.*}}) outs(%[[T]] : {{.*}})
// CHECK:           %[[O:.*]] = memref.alloc() {{.*}} : memref<4x4xf32>
// CHECK:           dataflow.task {
// CHECK:             linalg.fill {{.*}} outs(%[[O]] : {{.*}})
// CHECK:             linalg.matmul ins(%[[T]], %[[A]] : {{.*}}) outs(%[[O]] : {{.*}})
// CHECK:           dataflow.task {
// CHECK:             linalg.copy ins(%[[O]] : {{.*}}) outs(%[[OUT]] : {{.*}})
// CHECK-NOT:     memref.copy
// CHECK:         return
func.func @edges(%in: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %cst = arith.constant 0.0 : f32
  %s_buf = bufferization.alloc_tensor() : tensor<4x4xf32>
  %r = dataflow.dispatch : tensor<4x4xf32> {
    %a_buf = bufferization.alloc_tensor() : tensor<4x4xf32>
    %a = dataflow.task : tensor<4x4xf32> {
      %0 = linalg.copy ins(%in : tensor<4x4xf32>) outs(%a_buf : tensor<4x4xf32>) -> tensor<4x4xf32>
      dataflow.yield %0 : tensor<4x4xf32>
    }
    %t_buf = bufferization.alloc_tensor() : tensor<4x4xf32>
    %t = dataflow.task : tensor<4x4xf32> {
      %e = tensor.empty() : tensor<4x4xf32>
      %0 = linalg.add ins(%a, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
      %1 = linalg.transpose ins(%0 : tensor<4x4xf32>) outs(%t_buf : tensor<4x4xf32>) permutation = [1, 0]
      dataflow.yield %1 : tensor<4x4xf32>
    }
    %o_buf = bufferization.alloc_tensor() : tensor<4x4xf32>
    %o = dataflow.task : tensor<4x4xf32> {
      %0 = linalg.fill ins(%cst : f32) outs(%o_buf : tensor<4x4xf32>) -> tensor<4x4xf32>
      %1 = linalg.matmul ins(%t, %a : tensor<4x4xf32>, tensor<4x4xf32>) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
      dataflow.yield %1 : tensor<4x4xf32>
    }
    %s = dataflow.task : tensor<4x4xf32> {
      %0 = linalg.copy ins(%o : tensor<4x4xf32>) outs(%s_buf : tensor<4x4xf32>) -> tensor<4x4xf32>
      dataflow.yield %0 : tensor<4x4xf32>
    }
    dataflow.yield %s : tensor<4x4xf32>
  }
  return %r : tensor<4x4xf32>
}
