// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-create-dataflow-tasks))" -split-input-file | FileCheck %s

// Two stream-connected loop bands become two tasks inside one dispatch, and
// the channel declaration is hoisted so both tasks can see it.
// CHECK-LABEL: func.func @load_store
func.func @load_store(%in: memref<8xf32>, %out: memref<8xf32>) {
  // CHECK: dataflow.dispatch {
  // CHECK-NEXT: %[[C:.*]] = dataflow.stream {depth = 8 : i32} : <f32, 8>
  %c = dataflow.stream {depth = 8 : i32} : <f32, 8>
  // CHECK: dataflow.task {
  // CHECK: dataflow.stream_write %[[C]]
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    dataflow.stream_write %c, %v : <f32, 8>, f32
  }
  // CHECK: dataflow.task {
  // CHECK: dataflow.stream_read %[[C]]
  affine.for %i = 0 to 8 {
    %r = dataflow.stream_read %c : (!dataflow.stream<f32, 8>) -> f32
    affine.store %r, %out[%i] : memref<8xf32>
  }
  return
}

// -----

// The stream is declared after the first loop in the input; hoisting must
// still place it ahead of every task that refers to it.
// CHECK-LABEL: func.func @hoist_stream_decl
func.func @hoist_stream_decl(%in: memref<8xf32>, %out: memref<8xf32>) {
  // CHECK: dataflow.dispatch {
  // CHECK-NEXT: dataflow.stream
  // CHECK-NEXT: dataflow.task
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
  }
  %c = dataflow.stream {depth = 8 : i32} : <f32, 8>
  affine.for %i = 0 to 8 {
    %r = dataflow.stream_read %c : (!dataflow.stream<f32, 8>) -> f32
    affine.store %r, %out[%i] : memref<8xf32>
  }
  return
}

// -----

// A buffer used by only one band is pulled into that band's task.
// CHECK-LABEL: func.func @exclusive_buffer_moves_into_task
func.func @exclusive_buffer_moves_into_task(%out: memref<8xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %buf = memref.alloc() : memref<8xf32>
  // CHECK: dataflow.task {
  // CHECK-NEXT: memref.alloc
  affine.for %i = 0 to 8 {
    affine.store %cst, %buf[%i] : memref<8xf32>
    %v = affine.load %buf[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
  }
  return
}

// -----

// A buffer shared between two bands must stay in the dispatch region: sinking
// it into the first task would put it out of scope for the second.
// CHECK-LABEL: func.func @shared_buffer_stays_in_dispatch
func.func @shared_buffer_stays_in_dispatch(%in: memref<8xf32>, %out: memref<8xf32>) {
  %buf = memref.alloc() : memref<8xf32>
  // CHECK: dataflow.dispatch {
  // CHECK-NEXT: memref.alloc
  // CHECK-NEXT: dataflow.task
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %v, %buf[%i] : memref<8xf32>
  }
  affine.for %i = 0 to 8 {
    %v = affine.load %buf[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
  }
  return
}

// -----

// Running the pass on IR that already has a dispatch must not nest a second
// one.
// CHECK-LABEL: func.func @idempotent
// CHECK-COUNT-1: dataflow.dispatch
// CHECK-NOT: dataflow.dispatch
func.func @idempotent(%in: memref<8xf32>) {
  dataflow.dispatch {
    dataflow.task {
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf32>
        affine.store %v, %in[%i] : memref<8xf32>
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}
