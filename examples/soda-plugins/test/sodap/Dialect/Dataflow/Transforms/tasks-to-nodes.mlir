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
// greedy driver. A node is IsolatedFromAbove, so it is cloned back into the
// body: a constant is no port, and does not become a parameter.
// CHECK-LABEL: func.func @constant_is_cloned
func.func @constant_is_cloned(%out: memref<8xf32>) {
  dataflow.dispatch {
    // CHECK: dataflow.node() -> (%{{.*}}) {{.*}} : () -> memref<8xf32> {
    // CHECK-NEXT: ^bb0(%[[OUT:.*]]: memref<8xf32>):
    // CHECK-NEXT: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: affine.store %[[CST]], %[[OUT]]
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

// -----

// Ports come in first-use order. With enough values for a pointer set to hash
// them, an order taken from the addresses would differ from run to run.
// CHECK-LABEL: func.func @ports_in_first_use_order
func.func @ports_in_first_use_order(
    %m1: memref<1xf32>, %m2: memref<2xf32>, %m3: memref<3xf32>,
    %m4: memref<4xf32>, %m5: memref<5xf32>, %m6: memref<6xf32>,
    %m7: memref<7xf32>, %m8: memref<8xf32>, %m9: memref<9xf32>,
    %m10: memref<10xf32>, %m11: memref<11xf32>, %m12: memref<12xf32>,
    %m13: memref<13xf32>, %m14: memref<14xf32>, %m15: memref<15xf32>,
    %m16: memref<16xf32>, %m17: memref<17xf32>, %m18: memref<18xf32>,
    %v: f32) {
  dataflow.dispatch {
    // CHECK: () -> (memref<18xf32>, memref<17xf32>, memref<16xf32>, memref<15xf32>, memref<14xf32>, memref<13xf32>, memref<12xf32>, memref<11xf32>, memref<10xf32>, memref<9xf32>, memref<8xf32>, memref<7xf32>, memref<6xf32>, memref<5xf32>, memref<4xf32>, memref<3xf32>, memref<2xf32>, memref<1xf32>)[f32]
    dataflow.task {
      affine.store %v, %m18[0] : memref<18xf32>
      affine.store %v, %m17[0] : memref<17xf32>
      affine.store %v, %m16[0] : memref<16xf32>
      affine.store %v, %m15[0] : memref<15xf32>
      affine.store %v, %m14[0] : memref<14xf32>
      affine.store %v, %m13[0] : memref<13xf32>
      affine.store %v, %m12[0] : memref<12xf32>
      affine.store %v, %m11[0] : memref<11xf32>
      affine.store %v, %m10[0] : memref<10xf32>
      affine.store %v, %m9[0] : memref<9xf32>
      affine.store %v, %m8[0] : memref<8xf32>
      affine.store %v, %m7[0] : memref<7xf32>
      affine.store %v, %m6[0] : memref<6xf32>
      affine.store %v, %m5[0] : memref<5xf32>
      affine.store %v, %m4[0] : memref<4xf32>
      affine.store %v, %m3[0] : memref<3xf32>
      affine.store %v, %m2[0] : memref<2xf32>
      affine.store %v, %m1[0] : memref<1xf32>
      dataflow.yield
    }
    dataflow.yield
  }
  return
}
