// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-copy-to-dma{top-func=top},canonicalize)" -split-input-file -verify-diagnostics | FileCheck %s

// A (8x12) at word 0, C (8x16) at word 96. Loading a 4x4 tile of A at (i, k)
// sends one request per row r, at (i + r) * 12 + k, length 4 (4 << 32), then
// reads the row; storing a tile of C at (i, j) is the same at 96 + (i + r) *
// 16 + j with writes. The memref arguments give way to the four streams.

// CHECK-DAG: #[[ADDRA:.*]] = affine_map<(d0, d1, d2) -> (d0 * 12 + d1 + d2 * 12)>
// CHECK-DAG: #[[ADDRC:.*]] = affine_map<(d0, d1, d2) -> (d0 * 16 + d1 + d2 * 16 + 96)>
// CHECK-LABEL: func.func @top
// CHECK-SAME: (%[[RQ:.*]]: !dataflow.stream<i64, 8>, %[[RD:.*]]: !dataflow.stream<i32, 16>, %[[WQ:.*]]: !dataflow.stream<i64, 8>, %[[WD:.*]]: !dataflow.stream<i32, 16>)
// CHECK: %[[LEN:.*]] = arith.constant 17179869184 : i64

// Load: inside the i and k loops, one request per row r.
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for %[[R:.*]] =
// CHECK:   %[[A:.*]] = affine.apply #[[ADDRA]](%{{.*}}, %{{.*}}, %[[R]])
// CHECK:   %[[A64:.*]] = arith.index_cast %[[A]] : index to i64
// CHECK:   %[[REQ:.*]] = arith.ori %[[A64]], %[[LEN]] : i64
// CHECK:   dataflow.stream_write %[[RQ]], %[[REQ]]
// CHECK:   scf.for %[[E:.*]] =
// CHECK:     %[[W:.*]] = dataflow.stream_read %[[RD]] : (!dataflow.stream<i32, 16>) -> i32
// CHECK:     %[[F:.*]] = arith.bitcast %[[W]] : i32 to f32
// CHECK:     memref.store %[[F]], %{{.*}}[%[[R]], %[[E]]]
// CHECK-NOT: memref.copy

// Store: inside the i and j loops.
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for %[[R2:.*]] =
// CHECK:   affine.apply #[[ADDRC]](%{{.*}}, %{{.*}}, %[[R2]])
// CHECK:   dataflow.stream_write %[[WQ]]
// CHECK:   scf.for %[[E2:.*]] =
// CHECK:     %[[F2:.*]] = memref.load %{{.*}}[%[[R2]], %[[E2]]]
// CHECK:     %[[W2:.*]] = arith.bitcast %[[F2]] : f32 to i32
// CHECK:     dataflow.stream_write %[[WD]], %[[W2]]
// CHECK-NOT: memref.copy
func.func @top(%A: memref<8x12xf32>, %C: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %buf = memref.alloc() : memref<4x4xf32>
  dataflow.dispatch {
    dataflow.task {
      scf.for %i = %c0 to %c8 step %c4 {
        scf.for %k = %c0 to %c12 step %c4 {
          %At = memref.subview %A[%i, %k] [4, 4] [1, 1] : memref<8x12xf32> to memref<4x4xf32, strided<[12, 1], offset: ?>>
          memref.copy %At, %buf : memref<4x4xf32, strided<[12, 1], offset: ?>> to memref<4x4xf32>
        }
      }
      dataflow.yield
    }
    dataflow.task {
      scf.for %i = %c0 to %c8 step %c4 {
        scf.for %j = %c0 to %c16 step %c4 {
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

// A copy of a whole argument is a tile of it at 0: the scalar %alpha is one
// request at word 0 of length 1 (1 << 32), %x one at word 1 of length 4.
// CHECK-LABEL: func.func @top
// CHECK-DAG: %[[RALPHA:.*]] = arith.constant 4294967296 : i64
// CHECK-DAG: %[[RX:.*]] = arith.constant 17179869185 : i64
// CHECK: scf.for
// CHECK:   dataflow.stream_write %{{.*}}, %[[RALPHA]]
// CHECK:   dataflow.stream_read
// CHECK:   dataflow.stream_write %{{.*}}, %[[RX]]
func.func @top(%alpha: memref<f32>, %x: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %buf = memref.alloc() : memref<f32>
  %out = memref.alloc() : memref<4xf32>
  scf.for %i = %c0 to %c8 step %c4 {
    memref.copy %alpha, %buf : memref<f32> to memref<f32>
    memref.copy %out, %x : memref<4xf32> to memref<4xf32>
  }
  return
}

// -----

// An argument used other than by copies or tiles cannot go through the DMA.
func.func @top(%A: memref<8xf32>, %B: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{argument #0 is used other than by a copy or a static tile with unit strides}}
  %v = memref.load %A[%c0] : memref<8xf32>
  memref.store %v, %B[%c0] : memref<8xf32>
  return
}

// -----

func.func @top(%a: memref<4xf32>, %b: memref<4xf32>) {
  // expected-error@+1 {{a DMA copy must have exactly one top-level memory operand}}
  memref.copy %a, %b : memref<4xf32> to memref<4xf32>
  return
}

// -----

// Two copies may share one subview; erase that view only once.
// CHECK-LABEL: func.func @top
// CHECK-COUNT-2: dataflow.stream_write
// CHECK-NOT: memref.copy
// CHECK: return
func.func @top(%a: memref<8xf32>) {
  %x = memref.alloc() : memref<4xf32>
  %y = memref.alloc() : memref<4xf32>
  %s = memref.subview %a[0] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1]>>
  memref.copy %s, %x : memref<4xf32, strided<[1]>> to memref<4xf32>
  memref.copy %s, %y : memref<4xf32, strided<[1]>> to memref<4xf32>
  return
}
