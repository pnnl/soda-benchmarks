// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-infer-top-interface{top-func=top})" -split-input-file -verify-diagnostics | FileCheck %s

// The shape of a DMA accelerator: load sends read requests and receives the
// data, store sends write requests and the data. Every stream of the top is
// touched by some stage, so all four become axis ports, in the top and in the
// node they are passed to. The channel between the stages stays a fifo.

// CHECK-LABEL: func.func @top_node0
// CHECK-SAME: (%{{.*}}: !dataflow.stream<i64, 8, axis>, %{{.*}}: !dataflow.stream<i32, 16, axis>, %{{.*}}: !dataflow.stream<i32, 2>)
func.func @top_node0(%rd_req: !dataflow.stream<i64, 8>,
                     %rd_data: !dataflow.stream<i32, 16>,
                     %ch: !dataflow.stream<i32, 2>) {
  %c0 = arith.constant 0 : i64
  dataflow.stream_write %rd_req, %c0 : <i64, 8>, i64
  %v = dataflow.stream_read %rd_data : (!dataflow.stream<i32, 16>) -> i32
  dataflow.stream_write %ch, %v : <i32, 2>, i32
  return
}

// CHECK-LABEL: func.func @top_node1
// CHECK-SAME: (%{{.*}}: !dataflow.stream<i32, 2>, %{{.*}}: !dataflow.stream<i64, 8, axis>, %{{.*}}: !dataflow.stream<i32, 16, axis>)
func.func @top_node1(%ch: !dataflow.stream<i32, 2>,
                     %wr_req: !dataflow.stream<i64, 8>,
                     %wr_data: !dataflow.stream<i32, 16>) {
  %c0 = arith.constant 0 : i64
  // peek counts as a read, like stream_read.
  %v = dataflow.stream_peek %ch : (!dataflow.stream<i32, 2>) -> i32
  dataflow.stream_write %wr_req, %c0 : <i64, 8>, i64
  dataflow.stream_write %wr_data, %v : <i32, 16>, i32
  return
}

// The unused stream is not part of the external interface and stays a fifo.
// CHECK-LABEL: func.func @top
// CHECK-SAME: (%{{.*}}: !dataflow.stream<i64, 8, axis>, %{{.*}}: !dataflow.stream<i32, 16, axis>, %{{.*}}: !dataflow.stream<i64, 8, axis>, %{{.*}}: !dataflow.stream<i32, 16, axis>, %{{.*}}: !dataflow.stream<f32, 4>)
func.func @top(%rd_req: !dataflow.stream<i64, 8>,
               %rd_data: !dataflow.stream<i32, 16>,
               %wr_req: !dataflow.stream<i64, 8>,
               %wr_data: !dataflow.stream<i32, 16>,
               %unused: !dataflow.stream<f32, 4>) {
  // CHECK: dataflow.stream {depth = 2 : i32} : <i32, 2>
  %ch = dataflow.stream {depth = 2 : i32} : <i32, 2>
  // CHECK: call @top_node0({{.*}}) : (!dataflow.stream<i64, 8, axis>, !dataflow.stream<i32, 16, axis>, !dataflow.stream<i32, 2>) -> ()
  call @top_node0(%rd_req, %rd_data, %ch)
      : (!dataflow.stream<i64, 8>, !dataflow.stream<i32, 16>,
         !dataflow.stream<i32, 2>) -> ()
  // CHECK: call @top_node1({{.*}}) : (!dataflow.stream<i32, 2>, !dataflow.stream<i64, 8, axis>, !dataflow.stream<i32, 16, axis>) -> ()
  call @top_node1(%ch, %wr_req, %wr_data)
      : (!dataflow.stream<i32, 2>, !dataflow.stream<i64, 8>,
         !dataflow.stream<i32, 16>) -> ()
  return
}

// -----

// A port the top uses itself, without going through a node, and one that is
// already axis: both end up axis, and running again changes nothing.
// CHECK-LABEL: func.func @top
// CHECK-SAME: (%{{.*}}: !dataflow.stream<i32, 4, axis>, %{{.*}}: !dataflow.stream<i32, 4, axis>)
func.func @top(%in: !dataflow.stream<i32, 4>, %out: !dataflow.stream<i32, 4, axis>) {
  %v = dataflow.stream_read %in : (!dataflow.stream<i32, 4>) -> i32
  dataflow.stream_write %out, %v : <i32, 4, axis>, i32
  return
}

// -----

func.func @top_node0(%s: !dataflow.stream<i32, 4>) {
  // expected-note @+1 {{read here}}
  %v = dataflow.stream_read %s : (!dataflow.stream<i32, 4>) -> i32
  // expected-note @+1 {{written here}}
  dataflow.stream_write %s, %v : <i32, 4>, i32
  return
}

// An AXI4-Stream port carries data one way only.
// expected-error @+1 {{stream argument 0 is both read and written, but an axis port carries data in one direction only}}
func.func @top(%s: !dataflow.stream<i32, 4>) {
  call @top_node0(%s) : (!dataflow.stream<i32, 4>) -> ()
  return
}

// -----

func.func @top_node0(%s: !dataflow.stream<i32, 4>) {
  %v = dataflow.stream_read %s : (!dataflow.stream<i32, 4>) -> i32
  return
}

// The same node reads a port at one call and an internal channel at the
// other: one signature cannot be both.
func.func @top(%s: !dataflow.stream<i32, 4>) {
  %ch = dataflow.stream {depth = 4 : i32} : <i32, 4>
  call @top_node0(%s) : (!dataflow.stream<i32, 4>) -> ()
  // expected-error @+1 {{passes '!dataflow.stream<i32, 4>' as argument 0 of 'top_node0', which is an axis port at another call; give each use its own function}}
  call @top_node0(%ch) : (!dataflow.stream<i32, 4>) -> ()
  return
}

// -----

func.func private @external(!dataflow.stream<i32, 4>)

// A function with no body cannot say how it uses the stream.
func.func @top(%s: !dataflow.stream<i32, 4>) {
  // expected-error @+1 {{passes a stream of the top to 'external', which has no body to tell how it is used}}
  call @external(%s) : (!dataflow.stream<i32, 4>) -> ()
  return
}
