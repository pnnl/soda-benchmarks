// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-nodes-to-func)" -split-input-file | FileCheck %s

// Each node is outlined into its own function and the dispatch is unwrapped,
// leaving the flat call-graph form an HLS backend consumes. Nodes are numbered
// in source order, so node0 is the producer.
// CHECK-LABEL: func.func @top_node0
// CHECK-SAME: (%{{.*}}: memref<8xf32>, %{{.*}}: !dataflow.stream<f32, 8>)
// CHECK: dataflow.stream_write
// CHECK: return

// CHECK-LABEL: func.func @top_node1
// CHECK-SAME: (%{{.*}}: !dataflow.stream<f32, 8>, %{{.*}}: memref<8xf32>)
// CHECK: dataflow.stream_read
// CHECK: return

// CHECK-LABEL: func.func @top
// CHECK-NOT: dataflow.dispatch
// CHECK: %[[C:.*]] = dataflow.stream
// CHECK: call @top_node0(%{{.*}}, %[[C]])
// CHECK: call @top_node1(%[[C]], %{{.*}})
func.func @top(%in: memref<8xf32>, %out: memref<8xf32>) {
  %c = dataflow.stream {depth = 8 : i32} : <f32, 8>
  dataflow.dispatch {
    dataflow.node (%in) -> (%c) {inputTaps = [0 : i32]}
        : (memref<8xf32>) -> (!dataflow.stream<f32, 8>) {
    ^bb0(%ai: memref<8xf32>, %ac: !dataflow.stream<f32, 8>):
      affine.for %i = 0 to 8 {
        %v = affine.load %ai[%i] : memref<8xf32>
        dataflow.stream_write %ac, %v : <f32, 8>, f32
      }
    }
    dataflow.node (%c) -> (%out) {inputTaps = [0 : i32]}
        : (!dataflow.stream<f32, 8>) -> (memref<8xf32>) {
    ^bb0(%ac: !dataflow.stream<f32, 8>, %ao: memref<8xf32>):
      affine.for %i = 0 to 8 {
        %r = dataflow.stream_read %ac : (!dataflow.stream<f32, 8>) -> f32
        affine.store %r, %ao[%i] : memref<8xf32>
      }
    }
    dataflow.yield
  }
  return
}

// -----

// Names are qualified by the parent function, so two dataflow designs in one
// module do not collide.
// CHECK-DAG: func.func @first_node0
// CHECK-DAG: func.func @second_node0
func.func @first(%in: memref<8xf32>) {
  dataflow.dispatch {
    dataflow.node (%in) -> () {inputTaps = [0 : i32]} : (memref<8xf32>) -> () {
    ^bb0(%ai: memref<8xf32>):
      affine.for %i = 0 to 8 {
        %v = affine.load %ai[%i] : memref<8xf32>
      }
    }
    dataflow.yield
  }
  return
}

func.func @second(%in: memref<8xf32>) {
  dataflow.dispatch {
    dataflow.node (%in) -> () {inputTaps = [0 : i32]} : (memref<8xf32>) -> () {
    ^bb0(%ai: memref<8xf32>):
      affine.for %i = 0 to 8 {
        %v = affine.load %ai[%i] : memref<8xf32>
      }
    }
    dataflow.yield
  }
  return
}
