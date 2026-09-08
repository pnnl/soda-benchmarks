// Parse, print, then parse and print again: the two printed forms must match,
// which catches any op or type whose assembly format does not round-trip.
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext | mlir-opt --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext | FileCheck %s

// CHECK-LABEL: func.func @stream_channels
func.func @stream_channels(%arg0: memref<8xf32>) {
  // CHECK: %[[C:.*]] = dataflow.stream {depth = 8 : i32} : <f32, 8>
  %c = dataflow.stream {depth = 8 : i32} : <f32, 8>
  %v = affine.load %arg0[0] : memref<8xf32>
  // CHECK: dataflow.stream_write %[[C]], %{{.*}} : <f32, 8>, f32
  dataflow.stream_write %c, %v : <f32, 8>, f32
  // CHECK: %{{.*}} = dataflow.stream_read %[[C]] : (!dataflow.stream<f32, 8>) -> f32
  %r = dataflow.stream_read %c : (!dataflow.stream<f32, 8>) -> f32
  affine.store %r, %arg0[1] : memref<8xf32>
  return
}

// A stream_read with no result drops the popped value.
// CHECK-LABEL: func.func @stream_read_dropped
func.func @stream_read_dropped(%c: !dataflow.stream<i32, 4>) {
  // CHECK: dataflow.stream_read %{{.*}} : (!dataflow.stream<i32, 4>) -> ()
  dataflow.stream_read %c : (!dataflow.stream<i32, 4>) -> ()
  return
}

// Stream of a non-scalar element type.
// CHECK-LABEL: func.func @stream_of_vector
func.func @stream_of_vector(%c: !dataflow.stream<vector<4xf32>, 16>) {
  // CHECK: dataflow.stream_read %{{.*}} : (!dataflow.stream<vector<4xf32>, 16>) -> vector<4xf32>
  %r = dataflow.stream_read %c : (!dataflow.stream<vector<4xf32>, 16>) -> vector<4xf32>
  return
}

// CHECK-LABEL: func.func @dispatch_and_tasks
func.func @dispatch_and_tasks(%arg0: memref<8xf32>) {
  %c = dataflow.stream {depth = 8 : i32} : <f32, 8>
  // CHECK: dataflow.dispatch {
  dataflow.dispatch {
    // CHECK: dataflow.task {
    dataflow.task {
      %v = affine.load %arg0[0] : memref<8xf32>
      dataflow.stream_write %c, %v : <f32, 8>, f32
      dataflow.yield
    }
    // CHECK: dataflow.task {
    dataflow.task {
      %r = dataflow.stream_read %c : (!dataflow.stream<f32, 8>) -> f32
      affine.store %r, %arg0[1] : memref<8xf32>
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// A dispatch that yields results, so the optional result-type clause prints.
// CHECK-LABEL: func.func @dispatch_with_results
func.func @dispatch_with_results(%arg0: f32) -> f32 {
  // CHECK: %{{.*}} = dataflow.dispatch : f32 {
  %0 = dataflow.dispatch : f32 {
    // CHECK: %{{.*}} = dataflow.task : f32 {
    %1 = dataflow.task : f32 {
      %2 = arith.addf %arg0, %arg0 : f32
      // CHECK: dataflow.yield %{{.*}} : f32
      dataflow.yield %2 : f32
    }
    dataflow.yield %1 : f32
  }
  return %0 : f32
}

// CHECK-LABEL: func.func @nodes
func.func @nodes(%arg0: memref<8xf32>, %arg1: memref<8xf32>) {
  %c = dataflow.stream {depth = 8 : i32} : <f32, 8>
  dataflow.dispatch {
    // CHECK: dataflow.node(%{{.*}}) -> (%{{.*}}) {inputTaps = [0 : i32]} : (memref<8xf32>) -> !dataflow.stream<f32, 8>
    dataflow.node (%arg0) -> (%c) {inputTaps = [0 : i32]}
        : (memref<8xf32>) -> (!dataflow.stream<f32, 8>) {
    ^bb0(%am: memref<8xf32>, %ac: !dataflow.stream<f32, 8>):
      affine.for %i = 0 to 8 {
        %v = affine.load %am[%i] : memref<8xf32>
        dataflow.stream_write %ac, %v : <f32, 8>, f32
      }
    }
    dataflow.yield
  }
  return
}

// A node carrying scalar params in the optional param clause.
// CHECK-LABEL: func.func @node_with_params
func.func @node_with_params(%arg0: memref<8xf32>, %p: index) {
  dataflow.dispatch {
    // CHECK: dataflow.node(%{{.*}}) -> (%{{.*}}) [%{{.*}}] {inputTaps = [0 : i32]} : (memref<8xf32>) -> memref<8xf32>[index]
    dataflow.node (%arg0) -> (%arg0) [%p] {inputTaps = [0 : i32]}
        : (memref<8xf32>) -> (memref<8xf32>) [index] {
    ^bb0(%ai: memref<8xf32>, %ao: memref<8xf32>, %ap: index):
      %v = memref.load %ai[%ap] : memref<8xf32>
      memref.store %v, %ao[%ap] : memref<8xf32>
    }
    dataflow.yield
  }
  return
}
