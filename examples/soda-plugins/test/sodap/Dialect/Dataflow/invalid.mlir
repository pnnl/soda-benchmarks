// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext -split-input-file -verify-diagnostics

func.func @stream_depth_mismatch() {
  // The depth attribute and the depth encoded in the stream type must agree.
  // expected-error @+1 {{stream channel depth is not aligned}}
  %c = dataflow.stream {depth = 4 : i32} : <f32, 8>
  return
}

// -----

func.func @stream_read_type_mismatch(%c: !dataflow.stream<f32, 8>) {
  // expected-error @+1 {{result type doesn't align with channel type}}
  %r = dataflow.stream_read %c : (!dataflow.stream<f32, 8>) -> i32
  return
}

// -----

func.func @stream_peek_type_mismatch(%c: !dataflow.stream<f32, 8>) {
  // expected-error @+1 {{result type doesn't align with channel type}}
  %r = dataflow.stream_peek %c : (!dataflow.stream<f32, 8>) -> i32
  return
}

// -----

func.func @stream_write_type_mismatch(%c: !dataflow.stream<f32, 8>, %v: i32) {
  // expected-error @+1 {{value type doesn't align with channel type}}
  dataflow.stream_write %c, %v : <f32, 8>, i32
  return
}

// -----

func.func @dispatch_yield_mismatch(%arg0: f32) -> i32 {
  // expected-error @+1 {{yield type doesn't align with result type}}
  %0 = dataflow.dispatch : i32 {
    dataflow.yield %arg0 : f32
  }
  return %0 : i32
}

// -----

func.func @task_yield_mismatch(%arg0: f32) {
  dataflow.dispatch {
    // expected-error @+1 {{yield type doesn't align with result type}}
    %0 = dataflow.task : i32 {
      dataflow.yield %arg0 : f32
    }
    dataflow.yield
  }
  return
}

// -----

func.func @task_outside_dispatch() {
  // A task is only meaningful inside a dispatch region.
  // expected-error @+1 {{'dataflow.task' op expects parent op 'dataflow.dispatch'}}
  dataflow.task {
    dataflow.yield
  }
  return
}

// -----

func.func @node_outside_dispatch(%arg0: memref<8xf32>) {
  // expected-error @+1 {{'dataflow.node' op expects parent op 'dataflow.dispatch'}}
  dataflow.node (%arg0) -> (%arg0) {inputTaps = [0 : i32]}
      : (memref<8xf32>) -> (memref<8xf32>) {
  ^bb0(%ai: memref<8xf32>, %ao: memref<8xf32>):
  }
  return
}

// -----

func.func @bad_directive_bool() attributes {
    // The booleans are keywords, not integers.
    // expected-error @+1 {{expected 'true' or 'false'}}
    func_directive = #dataflow.fd<pipeline=1, targetInterval=1, dataflow=true>} {
  return
}

// -----

func.func @bad_resource_keyword() attributes {
    // expected-error @+1 {{expected 'dsp'}}
    resource = #dataflow.r<lut=1, dspX=2, bram=3>} {
  return
}

// -----

func.func @stream_op_axis() {
  // A channel declared inside the design cannot be one of its ports.
  // expected-error @+1 {{declares a channel inside the design, which cannot be an axis port}}
  %c = dataflow.stream {depth = 8 : i32} : <f32, 8, axis>
  return
}

// -----

// expected-error @+2 {{invalid stream interface (fifo or axis) specification: m_axi}}
// expected-error @+1 {{failed to parse StreamType parameter 'interface'}}
func.func @bad_stream_interface(%c: !dataflow.stream<f32, 8, m_axi>) {
  return
}
