// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter | FileCheck %s

// matmul tiled 4x4x4, promoted and multibuffered. Three stages: load (the
// copies from A and B), compute (the fill and the matmul, both writing bC),
// store (the copy into C). bA and bB have a token from load to compute, read
// in the k loop where their slot is; bC one from compute to store, in the j
// loop. The store task keeps no k loop.

// CHECK-LABEL: func.func @matmul
// CHECK-SAME: (%[[A:.*]]: memref<8x12xf32>, %[[B:.*]]: memref<12x16xf32>, %[[C:.*]]: memref<8x16xf32>)
// CHECK: %[[BC:.*]] = memref.alloc() : memref<2x4x4xf32>
// CHECK: %[[BA:.*]] = memref.alloc() : memref<2x4x4xf32>
// CHECK: %[[BB:.*]] = memref.alloc() : memref<2x4x4xf32>
// CHECK: dataflow.dispatch {
// CHECK: %[[TA:.*]] = dataflow.stream {depth = 1 : i32} : <i8, 1>
// CHECK: %[[TB:.*]] = dataflow.stream {depth = 1 : i32} : <i8, 1>
// CHECK: %[[TC:.*]] = dataflow.stream {depth = 1 : i32} : <i8, 1>

// Load.
// CHECK: dataflow.task {
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         memref.copy %{{.*}}, %{{.*}}
// CHECK:         memref.copy %{{.*}}, %{{.*}}
// CHECK:         %[[SA:.*]] = arith.index_cast %{{.*}} : index to i8
// CHECK:         dataflow.stream_write %[[TA]], %[[SA]]
// CHECK:         %[[SB:.*]] = arith.index_cast %{{.*}} : index to i8
// CHECK:         dataflow.stream_write %[[TB]], %[[SB]]
// CHECK-NOT: linalg.

// Compute. The matmul reads the C tile it accumulates, so it works on a
// private buffer, copied into bC's slot before the token: a stage writes a
// ping-pong buffer or reads it, not both.
// CHECK: dataflow.task {
// CHECK-NEXT: %[[ACC:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       %[[VC:.*]] = memref.subview %[[BC]]
// CHECK:       linalg.fill {{.*}} outs(%[[ACC]] :
// CHECK:       scf.for
// CHECK-NEXT:    dataflow.stream_peek %[[TA]]
// CHECK-NEXT:    dataflow.stream_peek %[[TB]]
// CHECK:         linalg.matmul {{.*}} outs(%[[ACC]] :
// CHECK-NEXT:    dataflow.stream_read %[[TA]]
// CHECK-NEXT:    dataflow.stream_read %[[TB]]
// CHECK-NEXT:  }
// CHECK-NEXT:  linalg.copy ins(%[[ACC]] : {{.*}}) outs(%[[VC]] :
// CHECK-NEXT:  %[[SC:.*]] = arith.index_cast %{{.*}} : index to i8
// CHECK-NEXT:  dataflow.stream_write %[[TC]], %[[SC]]
// CHECK-NOT: memref.copy

// Store.
// CHECK: dataflow.task {
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK-NEXT:  dataflow.stream_peek %[[TC]]
// CHECK-NOT:   scf.for
// CHECK:       memref.copy %{{.*}}, %{{.*}}
// CHECK-NEXT:  dataflow.stream_read %[[TC]]
#cslot = affine_map<(d0) -> ((d0 floordiv 4) mod 2)>
#abslot = affine_map<(d0, d1) -> ((d1 floordiv 4 + (d0 floordiv 4) * 3) mod 2)>
func.func @matmul(%A: memref<8x12xf32>, %B: memref<12x16xf32>, %C: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.0 : f32
  %bC = memref.alloc() : memref<2x4x4xf32>
  %bA = memref.alloc() : memref<2x4x4xf32>
  %bB = memref.alloc() : memref<2x4x4xf32>
  scf.for %i = %c0 to %c8 step %c4 {
    scf.for %j = %c0 to %c16 step %c4 {
      %sc = affine.apply #cslot(%j)
      %vC = memref.subview %bC[%sc, 0, 0] [1, 4, 4] [1, 1, 1] : memref<2x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
      %Ct = memref.subview %C[%i, %j] [4, 4] [1, 1] : memref<8x16xf32> to memref<4x4xf32, strided<[16, 1], offset: ?>>
      linalg.fill ins(%zero : f32) outs(%vC : memref<4x4xf32, strided<[4, 1], offset: ?>>)
      scf.for %k = %c0 to %c12 step %c4 {
        %sab = affine.apply #abslot(%j, %k)
        %vB = memref.subview %bB[%sab, 0, 0] [1, 4, 4] [1, 1, 1] : memref<2x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
        %vA = memref.subview %bA[%sab, 0, 0] [1, 4, 4] [1, 1, 1] : memref<2x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
        %At = memref.subview %A[%i, %k] [4, 4] [1, 1] : memref<8x12xf32> to memref<4x4xf32, strided<[12, 1], offset: ?>>
        %Bt = memref.subview %B[%k, %j] [4, 4] [1, 1] : memref<12x16xf32> to memref<4x4xf32, strided<[16, 1], offset: ?>>
        memref.copy %At, %vA : memref<4x4xf32, strided<[12, 1], offset: ?>> to memref<4x4xf32, strided<[4, 1], offset: ?>>
        memref.copy %Bt, %vB : memref<4x4xf32, strided<[16, 1], offset: ?>> to memref<4x4xf32, strided<[4, 1], offset: ?>>
        linalg.matmul ins(%vA, %vB : memref<4x4xf32, strided<[4, 1], offset: ?>>, memref<4x4xf32, strided<[4, 1], offset: ?>>)
                      outs(%vC : memref<4x4xf32, strided<[4, 1], offset: ?>>)
      }
      memref.copy %vC, %Ct : memref<4x4xf32, strided<[4, 1], offset: ?>> to memref<4x4xf32, strided<[16, 1], offset: ?>>
    }
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    // Any loop of the nest: the op works on the outermost one.
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.op<"scf.for">
    %dispatch = transform.sodap.split_stages %loops : (!transform.op<"scf.for">) -> !transform.any_op
    transform.yield
  }
}
