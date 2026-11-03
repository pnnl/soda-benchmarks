// End-to-end: affine IR with intermediate buffers becomes a dataflow design.
//
// The kernel below is the affine analogue of the load/compute/store pipeline
// in examples/bambu-esp-example/bambu_dma_example/pipe_dataflow_stream.cpp:
// three stages communicating through explicit channels. What comes out is a
// dispatch region -- the `#pragma HLS DATAFLOW` scope -- holding one node per
// stage, wired by two `!dataflow.stream` values.
//
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-affine-to-dataflow))" | FileCheck %s

// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-affine-to-dataflow),sodap-dataflow-nodes-to-func)" | FileCheck %s --check-prefix=FUNC

// CHECK-LABEL: func.func @top
// CHECK: dataflow.dispatch {
// The two intermediate buffers are gone, replaced by channels.
// CHECK-NOT: memref.alloc
// CHECK: %[[C0:.*]] = dataflow.stream {depth = 128 : i32} : <i32, 128>
// CHECK: %[[C1:.*]] = dataflow.stream {depth = 128 : i32} : <i32, 128>

// Stage 1 -- load: reads the input memref, writes the first channel.
// CHECK: dataflow.node(%{{.*}}) -> (%[[C0]]) {{.*}} : (memref<128xi32>) -> !dataflow.stream<i32, 128>
// CHECK: dataflow.stream_write

// Stage 2 -- compute: channel in, channel out. No memory ports at all.
// CHECK: dataflow.node(%[[C0]]) -> (%[[C1]]) {{.*}} : (!dataflow.stream<i32, 128>) -> !dataflow.stream<i32, 128>
// CHECK: dataflow.stream_read
// CHECK: arith.muli
// CHECK: dataflow.stream_write

// Stage 3 -- store: reads the second channel, writes the output memref.
// CHECK: dataflow.node(%[[C1]]) -> (%{{.*}}) {{.*}} : (!dataflow.stream<i32, 128>) -> memref<128xi32>
// CHECK: dataflow.stream_read
// CHECK: affine.store

// Flattened form: one function per stage, called from the top, exactly like
// load()/compute()/store() under the DATAFLOW pragma.
// FUNC-LABEL: func.func @top_node0
// FUNC-LABEL: func.func @top_node1
// FUNC-LABEL: func.func @top_node2
// FUNC-LABEL: func.func @top
// FUNC: %[[F0:.*]] = dataflow.stream
// FUNC: %[[F1:.*]] = dataflow.stream
// FUNC: call @top_node0(%{{.*}}, %[[F0]])
// FUNC: call @top_node1(%[[F0]], %[[F1]])
// FUNC: call @top_node2(%[[F1]], %{{.*}})
func.func @top(%in: memref<128xi32>, %out: memref<128xi32>) {
  %buf0 = memref.alloc() : memref<128xi32>
  %buf1 = memref.alloc() : memref<128xi32>

  // load
  affine.for %i = 0 to 128 {
    %v = affine.load %in[%i] : memref<128xi32>
    affine.store %v, %buf0[%i] : memref<128xi32>
  }
  // compute
  affine.for %i = 0 to 128 {
    %v = affine.load %buf0[%i] : memref<128xi32>
    %s = arith.muli %v, %v : i32
    affine.store %s, %buf1[%i] : memref<128xi32>
  }
  // store
  affine.for %i = 0 to 128 {
    %v = affine.load %buf1[%i] : memref<128xi32>
    affine.store %v, %out[%i] : memref<128xi32>
  }
  return
}
