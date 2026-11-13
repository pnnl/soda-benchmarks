// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-copy-to-dma{top-func=top},func.func(sodap-dataflow-outstanding-load),canonicalize)" -split-input-file -verify-diagnostics | FileCheck %s

// The load task of a 4x4 tile of A per (i, k), after copy-to-dma: a request
// per row r, then the row's data, then the ping-pong token. The requests go
// to a task of their own, in front, with the i, k and r loops; the load task
// keeps the data and the token, and admits each request with its descriptor.
// The issuer's DMA payload must depend on the blocking descriptor read.

// CHECK-LABEL: func.func @top
// CHECK-SAME: (%[[RQ:[^:]*]]: !dataflow.stream<i64, 4>, %[[RD:[^:]*]]: !dataflow.stream<i32, 1>
// CHECK: %[[TOKEN:.*]] = dataflow.stream
// CHECK: %[[ADMITTED:.*]] = dataflow.stream {{.*}} : <i64, 1>
// CHECK: dataflow.task attributes {outstanding_load} {
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         %[[REQUEST:.*]] = dataflow.stream_read %[[ADMITTED]]
// CHECK:         dataflow.stream_write %[[RQ]], %[[REQUEST]] {sodap.dma_ordered}
// CHECK-NOT:     memref.store
// CHECK-NOT:     dataflow.stream_write %[[TOKEN]]
// CHECK: dataflow.task attributes {load} {
// CHECK-NOT: dataflow.stream_write %[[RQ]]
// CHECK:         affine.apply
// CHECK:         dataflow.stream_write %[[ADMITTED]], %{{.*}} {sodap.dma_ordered}
// CHECK:         scf.for
// CHECK:           dataflow.stream_read %[[RD]]
// CHECK:           memref.store
// CHECK:     dataflow.stream_write %[[TOKEN]], %{{.*}} {sodap.dma_ordered}
// CHECK: dataflow.task attributes {store} {
func.func @top(%A: memref<8x12xf32>, %C: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %t0 = arith.constant 0 : i8
  %buf = memref.alloc() : memref<4x4xf32>
  dataflow.dispatch {
    %token = dataflow.stream {depth = 1 : i32} : <i8, 1>
    dataflow.task attributes {load} {
      scf.for %i = %c0 to %c8 step %c4 {
        scf.for %k = %c0 to %c12 step %c4 {
          %At = memref.subview %A[%i, %k] [4, 4] [1, 1] : memref<8x12xf32> to memref<4x4xf32, strided<[12, 1], offset: ?>>
          memref.copy %At, %buf : memref<4x4xf32, strided<[12, 1], offset: ?>> to memref<4x4xf32>
          dataflow.stream_write %token, %t0 : <i8, 1>, i8
        }
      }
      dataflow.yield
    }
    dataflow.task attributes {store} {
      scf.for %i = %c0 to %c8 step %c4 {
        scf.for %j = %c0 to %c16 step %c4 {
          %t = dataflow.stream_read %token : (!dataflow.stream<i8, 1>) -> i8
          %Ct = memref.subview %C[%i, %j] [4, 4] [1, 1] : memref<8x16xf32> to memref<4x4xf32, strided<[16, 1], offset: ?>>
          memref.copy %buf, %Ct : memref<4x4xf32> to memref<4x4xf32, strided<[16, 1], offset: ?>>
        }
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// No load task: nothing to do.
// CHECK-LABEL: func.func @top
// CHECK-NOT: outstanding_load
func.func @top(%x: memref<4xf32>) {
  dataflow.dispatch {
    dataflow.task {
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// expected-error @+1 {{expected one load task}}
func.func @top() {
  dataflow.dispatch {
    dataflow.task attributes {load} {
      dataflow.yield
    }
    dataflow.task attributes {load} {
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// The DMA streams come after %n: the first argument isn't the requests.
// expected-error @+1 {{expected the first argument to be the read-request stream, only written to}}
func.func @top(%n: i32) {
  dataflow.dispatch {
    dataflow.task attributes {load} {
      dataflow.yield
    }
    dataflow.yield
  }
  return
}
