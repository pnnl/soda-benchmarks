// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter | FileCheck %s

// matmul tiled 4x4x4, promoted and multibuffered. Three stages: load (the
// copies from A and B), compute (the fill and the matmul, both writing bC),
// store (the copy into C). bA and bB have a token from load to compute, read
// in the k loop where their slot is, written after both copies; bC one from compute to store, in the j
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

// A strip of A loaded in the i loop, ahead of the j loop that loads B: its
// token is written before the j loop, not at the end of the i body. There
// load would first run the j loop, whose second B token waits for compute to
// read the first one, while compute waits for the A token before its j loop.
// CHECK-LABEL: func.func @outer_load
// CHECK: dataflow.task {
// CHECK-NEXT: scf.for
// CHECK:        memref.copy %{{.*}}, %{{.*}} : memref<4x8xf32
// CHECK-NEXT:   %[[SA:.*]] = arith.index_cast %{{.*}} : index to i8
// CHECK-NEXT:   dataflow.stream_write %{{.*}}, %[[SA]]
// CHECK-NEXT:   scf.for
// CHECK:          memref.copy %{{.*}}, %{{.*}} : memref<8x4xf32
// CHECK-NEXT:     %[[SB:.*]] = arith.index_cast %{{.*}} : index to i8
// CHECK-NEXT:     dataflow.stream_write %{{.*}}, %[[SB]]
// CHECK-NEXT:   }
// CHECK-NEXT: }
#islot = affine_map<(d0) -> ((d0 floordiv 4) mod 2)>
#jslot = affine_map<(d0, d1) -> ((d1 floordiv 4 + (d0 floordiv 4) * 2) mod 2)>
func.func @outer_load(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %bA = memref.alloc() : memref<2x4x8xf32>
  %bB = memref.alloc() : memref<2x8x4xf32>
  %bC = memref.alloc() : memref<2x4x4xf32>
  scf.for %i = %c0 to %c8 step %c4 {
    %sa = affine.apply #islot(%i)
    %vA = memref.subview %bA[%sa, 0, 0] [1, 4, 8] [1, 1, 1] : memref<2x4x8xf32> to memref<4x8xf32, strided<[8, 1], offset: ?>>
    %At = memref.subview %A[%i, 0] [4, 8] [1, 1] : memref<8x8xf32> to memref<4x8xf32, strided<[8, 1], offset: ?>>
    memref.copy %At, %vA : memref<4x8xf32, strided<[8, 1], offset: ?>> to memref<4x8xf32, strided<[8, 1], offset: ?>>
    scf.for %j = %c0 to %c8 step %c4 {
      %sb = affine.apply #jslot(%i, %j)
      %vB = memref.subview %bB[%sb, 0, 0] [1, 8, 4] [1, 1, 1] : memref<2x8x4xf32> to memref<8x4xf32, strided<[4, 1], offset: ?>>
      %vC = memref.subview %bC[%sb, 0, 0] [1, 4, 4] [1, 1, 1] : memref<2x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
      %Bt = memref.subview %B[0, %j] [8, 4] [1, 1] : memref<8x8xf32> to memref<8x4xf32, strided<[8, 1], offset: ?>>
      memref.copy %Bt, %vB : memref<8x4xf32, strided<[8, 1], offset: ?>> to memref<8x4xf32, strided<[4, 1], offset: ?>>
      linalg.matmul ins(%vA, %vB : memref<4x8xf32, strided<[8, 1], offset: ?>>, memref<8x4xf32, strided<[4, 1], offset: ?>>)
                    outs(%vC : memref<4x4xf32, strided<[4, 1], offset: ?>>)
      %Ct = memref.subview %C[%i, %j] [4, 4] [1, 1] : memref<8x8xf32> to memref<4x4xf32, strided<[8, 1], offset: ?>>
      memref.copy %vC, %Ct : memref<4x4xf32, strided<[4, 1], offset: ?>> to memref<4x4xf32, strided<[8, 1], offset: ?>>
    }
  }
  return
}

// A scalar loaded once before the nest: its copy goes into the load task,
// before the nest, followed by its token, which carries 0 and is for the
// whole task: the consumer peeks it before its loops and reads it after them.
// CHECK-LABEL: func.func @scalar
// CHECK: %[[T:.*]] = dataflow.stream {depth = 1 : i32} : <i8, 1>
// CHECK: dataflow.task {
// CHECK-NEXT: memref.copy %{{.*}}, %{{.*}} : memref<f32> to memref<f32>
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i8
// CHECK-NEXT: dataflow.stream_write %[[T]], %[[ZERO]]
// CHECK-NEXT: scf.for
// CHECK: dataflow.task {
// CHECK-NEXT: dataflow.stream_peek %[[T]]
// CHECK-NEXT: scf.for
// CHECK:        dataflow.stream_write
// CHECK-NEXT: }
// CHECK-NEXT: dataflow.stream_read %[[T]]
// CHECK-NEXT: }
// CHECK-NOT: memref.copy %{{.*}} : memref<f32> to memref<f32>
#xslot = affine_map<(d0) -> ((d0 floordiv 4) mod 2)>
func.func @scalar(%alpha: memref<f32>, %X: memref<8xf32>, %Y: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %bS = memref.alloc() : memref<f32>
  %bX = memref.alloc() : memref<2x4xf32>
  %bY = memref.alloc() : memref<2x4xf32>
  memref.copy %alpha, %bS : memref<f32> to memref<f32>
  scf.for %i = %c0 to %c8 step %c4 {
    %s = affine.apply #xslot(%i)
    %vX = memref.subview %bX[%s, 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
    %vY = memref.subview %bY[%s, 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
    %Xt = memref.subview %X[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
    memref.copy %Xt, %vX : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
        ins(%bS, %vX : memref<f32>, memref<4xf32, strided<[1], offset: ?>>) outs(%vY : memref<4xf32, strided<[1], offset: ?>>) {
    ^bb0(%a: f32, %x: f32, %y: f32):
      %0 = arith.mulf %a, %x : f32
      linalg.yield %0 : f32
    }
    %Yt = memref.subview %Y[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
    memref.copy %vY, %Yt : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32, strided<[1], offset: ?>>
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
