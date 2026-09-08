// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext -canonicalize -split-input-file | FileCheck %s

// A dispatch holding no tasks is pure scaffolding, so it gets inlined away.
// CHECK-LABEL: func.func @inline_dispatch_without_tasks
func.func @inline_dispatch_without_tasks(%arg0: memref<8xf32>) {
  // CHECK-NOT: dataflow.dispatch
  // CHECK: affine.load
  // CHECK: affine.store
  dataflow.dispatch {
    %v = affine.load %arg0[0] : memref<8xf32>
    affine.store %v, %arg0[1] : memref<8xf32>
    dataflow.yield
  }
  return
}

// -----

// An empty task contributes nothing to the partition and is folded away. Note
// the terminator counts as an op in the body, so this fires only when the task
// holds nothing else.
// CHECK-LABEL: func.func @inline_empty_task
func.func @inline_empty_task(%arg0: memref<8xf32>, %c: !dataflow.stream<f32, 8>) {
  // CHECK: dataflow.dispatch
  dataflow.dispatch {
    // CHECK-COUNT-1: dataflow.task
    // CHECK-NOT: dataflow.task
    dataflow.task {
      dataflow.yield
    }
    dataflow.task {
      affine.for %i = 0 to 8 {
        %v = affine.load %arg0[%i] : memref<8xf32>
        dataflow.stream_write %c, %v : <f32, 8>, f32
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// Real stages are preserved: neither the dispatch nor its tasks may be folded
// away, since that would destroy the partition the pipeline just computed.
// CHECK-LABEL: func.func @preserve_real_stages
func.func @preserve_real_stages(%arg0: memref<8xf32>, %c: !dataflow.stream<f32, 8>) {
  // CHECK: dataflow.dispatch
  dataflow.dispatch {
    // CHECK-COUNT-2: dataflow.task
    dataflow.task {
      affine.for %i = 0 to 8 {
        %v = affine.load %arg0[%i] : memref<8xf32>
        dataflow.stream_write %c, %v : <f32, 8>, f32
      }
      dataflow.yield
    }
    dataflow.task {
      affine.for %i = 0 to 8 {
        %r = dataflow.stream_read %c : (!dataflow.stream<f32, 8>) -> f32
        affine.store %r, %arg0[%i] : memref<8xf32>
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// Results that nothing consumes are dropped from the dispatch's port list. The
// two tasks keep the dispatch itself from being inlined, isolating the
// port-pruning rewrite.
// CHECK-LABEL: func.func @drop_unused_dispatch_result
func.func @drop_unused_dispatch_result(%arg0: memref<8xf32>, %c: !dataflow.stream<f32, 8>, %f: f32) -> f32 {
  // CHECK: dataflow.dispatch : f32 {
  %0:2 = dataflow.dispatch : f32, f32 {
    %a = dataflow.task : f32 {
      affine.for %i = 0 to 8 {
        %v = affine.load %arg0[%i] : memref<8xf32>
        dataflow.stream_write %c, %v : <f32, 8>, f32
      }
      %x = arith.addf %f, %f : f32
      dataflow.yield %x : f32
    }
    %b = dataflow.task : f32 {
      affine.for %i = 0 to 8 {
        %r = dataflow.stream_read %c : (!dataflow.stream<f32, 8>) -> f32
        affine.store %r, %arg0[%i] : memref<8xf32>
      }
      %y = arith.mulf %f, %f : f32
      dataflow.yield %y : f32
    }
    // CHECK: dataflow.yield %{{.*}} : f32
    dataflow.yield %a, %b : f32, f32
  }
  return %0#0 : f32
}
