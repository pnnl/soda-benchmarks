// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-tasks-to-nodes))" -split-input-file | FileCheck %s

// Port classification: the producer reads a memref and writes a stream, so the
// memref is an input and the stream an output. The consumer is the mirror
// image. Getting the stream on the right side depends on stream_write being
// recognised as a write even though it declares no memory effects.
// CHECK-LABEL: func.func @classify_ports
func.func @classify_ports(%in: memref<8xf32>, %out: memref<8xf32>) {
  %c = dataflow.stream {depth = 8 : i32} : <f32, 8>
  dataflow.dispatch {
    // CHECK: dataflow.node(%{{.*}}) -> (%{{.*}}) {{.*}} : (memref<8xf32>) -> !dataflow.stream<f32, 8>
    dataflow.task {
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf32>
        dataflow.stream_write %c, %v : <f32, 8>, f32
      }
      dataflow.yield
    }
    // CHECK: dataflow.node(%{{.*}}) -> (%{{.*}}) {{.*}} : (!dataflow.stream<f32, 8>) -> memref<8xf32>
    dataflow.task {
      affine.for %i = 0 to 8 {
        %r = dataflow.stream_read %c : (!dataflow.stream<f32, 8>) -> f32
        affine.store %r, %out[%i] : memref<8xf32>
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// A memref that is only read stays an input; one that is stored to becomes an
// output, via the ordinary memory-effect path.
// CHECK-LABEL: func.func @memref_read_vs_write
func.func @memref_read_vs_write(%in: memref<8xf32>, %out: memref<8xf32>) {
  dataflow.dispatch {
    // CHECK: dataflow.node(%{{.*}}) -> (%{{.*}}) {{.*}} : (memref<8xf32>) -> memref<8xf32>
    dataflow.task {
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf32>
        affine.store %v, %out[%i] : memref<8xf32>
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// Non-shaped liveins become parameters rather than ports.
// CHECK-LABEL: func.func @scalar_livein_becomes_param
func.func @scalar_livein_becomes_param(%out: memref<8xf32>, %s: f32) {
  dataflow.dispatch {
    // The scalar appears in the bracketed parameter clause, not in the
    // input/output signature.
    // CHECK: dataflow.node({{.*}}) -> (%{{.*}}) [%{{.*}}] {{.*}} : () -> memref<8xf32>[f32]
    dataflow.task {
      affine.for %i = 0 to 8 {
        affine.store %s, %out[%i] : memref<8xf32>
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// A constant used inside the task is hoisted to the function entry by the
// greedy driver, and then -- because a node is IsolatedFromAbove -- must reach
// the body as a parameter rather than as a free reference.
// CHECK-LABEL: func.func @constant_becomes_param
func.func @constant_becomes_param(%out: memref<8xf32>) {
  dataflow.dispatch {
    // CHECK: dataflow.node() -> (%{{.*}}) [%{{.*}}] {{.*}} : () -> memref<8xf32>[f32]
    // CHECK-NEXT: ^bb0(%{{.*}}: memref<8xf32>, %[[P:.*]]: f32):
    dataflow.task {
      %cst = arith.constant 0.000000e+00 : f32
      affine.for %i = 0 to 8 {
        affine.store %cst, %out[%i] : memref<8xf32>
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}
