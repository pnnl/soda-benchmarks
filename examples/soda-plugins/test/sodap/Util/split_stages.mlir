// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter | FileCheck %s

// matmul tiled 4x4x4, promoted and multibuffered. Three stages: load (the
// copies from A and B), compute (the fill and the matmul, both writing bC),
// store (the copy into C). bA and bB have a token from load to compute, read
// in the k loop where their slot is, written after both copies; bC one from compute to store, in the j
// loop. The producer owns the slot variable and switches it after the token;
// the consumer indexes the buffer with the token. The store task keeps no k loop.

// CHECK-LABEL: func.func @matmul
// CHECK-SAME: (%[[A:.*]]: memref<8x12xf32>, %[[B:.*]]: memref<12x16xf32>, %[[C:.*]]: memref<8x16xf32>)
// CHECK: %[[BC:.*]] = memref.alloc() : memref<2x4x4xf32>
// CHECK: %[[BA:.*]] = memref.alloc() : memref<2x4x4xf32>
// CHECK: %[[BB:.*]] = memref.alloc() : memref<2x4x4xf32>
// CHECK-NOT: memref.alloca
// CHECK: dataflow.dispatch {
// CHECK: %[[TA:.*]] = dataflow.stream {depth = 1 : i32} : <i8, 1>
// CHECK: %[[TB:.*]] = dataflow.stream {depth = 1 : i32} : <i8, 1>
// CHECK: %[[TC:.*]] = dataflow.stream {depth = 1 : i32} : <i8, 1>

// Load.
// CHECK: dataflow.task {
// CHECK:   %[[VARA:.*]] = memref.alloca() : memref<i8>
// CHECK:   %[[VARB:.*]] = memref.alloca() : memref<i8>
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         %[[SB:.*]] = memref.load %[[VARB]][] : memref<i8>
// CHECK:         %[[SA:.*]] = memref.load %[[VARA]][] : memref<i8>
// CHECK:         memref.copy %{{.*}}, %{{.*}}
// CHECK:         memref.copy %{{.*}}, %{{.*}}
// CHECK-NEXT:    dataflow.stream_write %[[TA]], %[[SA]]
// CHECK:         %[[INCA:.*]] = arith.addi %[[SA]], %{{.*}} : i8
// CHECK:         %[[NEXTA:.*]] = arith.remui %[[INCA]], %{{.*}} : i8
// CHECK-NEXT:    memref.store %[[NEXTA]], %[[VARA]][] : memref<i8>
// CHECK-NEXT:    dataflow.stream_write %[[TB]], %[[SB]]
// CHECK:         memref.store %{{.*}}, %[[VARB]][] : memref<i8>
// CHECK-NOT: linalg.

// Compute. The matmul reads the C tile it accumulates, so it works on a
// private buffer, copied into bC's slot before the token: a stage writes a
// ping-pong buffer or reads it, not both.
// CHECK: dataflow.task {
// CHECK-NEXT: %[[ACC:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK-NEXT: %[[VARC:.*]] = memref.alloca() : memref<i8>
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       %[[SC:.*]] = memref.load %[[VARC]][] : memref<i8>
// CHECK:       %[[VC:.*]] = memref.subview %[[BC]]
// CHECK:       linalg.fill {{.*}} outs(%[[ACC]] :
// CHECK:       scf.for
// CHECK-NEXT:    %[[PA:.*]] = dataflow.stream_peek %[[TA]]
// CHECK-NEXT:    %[[PB:.*]] = dataflow.stream_peek %[[TB]]
// CHECK-NEXT:    %[[IB:.*]] = arith.index_cast %[[PB]] : i8 to index
// CHECK-NEXT:    memref.subview %[[BB]][%[[IB]], 0, 0]
// CHECK-NEXT:    %[[IA:.*]] = arith.index_cast %[[PA]] : i8 to index
// CHECK-NEXT:    memref.subview %[[BA]][%[[IA]], 0, 0]
// CHECK:         linalg.matmul {{.*}} outs(%[[ACC]] :
// CHECK-NEXT:    dataflow.stream_read %[[TA]]
// CHECK-NEXT:    dataflow.stream_read %[[TB]]
// CHECK-NEXT:  }
// CHECK-NEXT:  linalg.copy ins(%[[ACC]] : {{.*}}) outs(%[[VC]] :
// CHECK-NEXT:  dataflow.stream_write %[[TC]], %[[SC]]
// CHECK:       memref.store %{{.*}}, %[[VARC]][] : memref<i8>
// CHECK-NOT: memref.copy

// Store.
// CHECK: dataflow.task {
// CHECK-NOT: memref.alloca
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK-NEXT:  %[[PC:.*]] = dataflow.stream_peek %[[TC]]
// CHECK-NEXT:  %[[IC:.*]] = arith.index_cast %[[PC]] : i8 to index
// CHECK-NEXT:  memref.subview %[[BC]][%[[IC]], 0, 0]
// CHECK-NOT:   scf.for
// CHECK-NOT:   memref.store
// CHECK:       memref.copy %{{.*}}, %{{.*}}
// CHECK-NEXT:  dataflow.stream_read %[[TC]]
func.func @matmul(%A: memref<8x12xf32>, %B: memref<12x16xf32>, %C: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.0 : f32
  %z = arith.constant 0 : i8
  %one = arith.constant 1 : i8
  %two = arith.constant 2 : i8
  %bC = memref.alloc() : memref<2x4x4xf32>
  %varC = memref.alloca() : memref<i8>
  memref.store %z, %varC[] : memref<i8>
  %bA = memref.alloc() : memref<2x4x4xf32>
  %varA = memref.alloca() : memref<i8>
  memref.store %z, %varA[] : memref<i8>
  %bB = memref.alloc() : memref<2x4x4xf32>
  %varB = memref.alloca() : memref<i8>
  memref.store %z, %varB[] : memref<i8>
  scf.for %i = %c0 to %c8 step %c4 {
    scf.for %j = %c0 to %c16 step %c4 {
      %sc = memref.load %varC[] : memref<i8>
      %ic = arith.index_cast %sc : i8 to index
      %vC = memref.subview %bC[%ic, 0, 0] [1, 4, 4] [1, 1, 1] : memref<2x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
      %Ct = memref.subview %C[%i, %j] [4, 4] [1, 1] : memref<8x16xf32> to memref<4x4xf32, strided<[16, 1], offset: ?>>
      linalg.fill ins(%zero : f32) outs(%vC : memref<4x4xf32, strided<[4, 1], offset: ?>>)
      scf.for %k = %c0 to %c12 step %c4 {
        %sb = memref.load %varB[] : memref<i8>
        %ib = arith.index_cast %sb : i8 to index
        %vB = memref.subview %bB[%ib, 0, 0] [1, 4, 4] [1, 1, 1] : memref<2x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
        %sa = memref.load %varA[] : memref<i8>
        %ia = arith.index_cast %sa : i8 to index
        %vA = memref.subview %bA[%ia, 0, 0] [1, 4, 4] [1, 1, 1] : memref<2x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
        %At = memref.subview %A[%i, %k] [4, 4] [1, 1] : memref<8x12xf32> to memref<4x4xf32, strided<[12, 1], offset: ?>>
        %Bt = memref.subview %B[%k, %j] [4, 4] [1, 1] : memref<12x16xf32> to memref<4x4xf32, strided<[16, 1], offset: ?>>
        memref.copy %At, %vA : memref<4x4xf32, strided<[12, 1], offset: ?>> to memref<4x4xf32, strided<[4, 1], offset: ?>>
        memref.copy %Bt, %vB : memref<4x4xf32, strided<[16, 1], offset: ?>> to memref<4x4xf32, strided<[4, 1], offset: ?>>
        linalg.matmul ins(%vA, %vB : memref<4x4xf32, strided<[4, 1], offset: ?>>, memref<4x4xf32, strided<[4, 1], offset: ?>>)
                      outs(%vC : memref<4x4xf32, strided<[4, 1], offset: ?>>)
        %ab = arith.addi %sb, %one : i8
        %nb = arith.remui %ab, %two : i8
        memref.store %nb, %varB[] : memref<i8>
        %aa = arith.addi %sa, %one : i8
        %na = arith.remui %aa, %two : i8
        memref.store %na, %varA[] : memref<i8>
      }
      memref.copy %vC, %Ct : memref<4x4xf32, strided<[4, 1], offset: ?>> to memref<4x4xf32, strided<[16, 1], offset: ?>>
      %ac = arith.addi %sc, %one : i8
      %nc = arith.remui %ac, %two : i8
      memref.store %nc, %varC[] : memref<i8>
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
// CHECK:      scf.for
// CHECK:        %[[SA:.*]] = memref.load
// CHECK:        memref.copy %{{.*}}, %{{.*}} : memref<4x8xf32
// CHECK-NEXT:   dataflow.stream_write %{{.*}}, %[[SA]]
// CHECK:        memref.store
// CHECK-NEXT:   scf.for
// CHECK:          %[[SB:.*]] = memref.load
// CHECK:          memref.copy %{{.*}}, %{{.*}} : memref<8x4xf32
// CHECK-NEXT:     dataflow.stream_write %{{.*}}, %[[SB]]
// CHECK:          memref.store
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @outer_load(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %z = arith.constant 0 : i8
  %one = arith.constant 1 : i8
  %two = arith.constant 2 : i8
  %bA = memref.alloc() : memref<2x4x8xf32>
  %varA = memref.alloca() : memref<i8>
  memref.store %z, %varA[] : memref<i8>
  %bB = memref.alloc() : memref<2x8x4xf32>
  %varB = memref.alloca() : memref<i8>
  memref.store %z, %varB[] : memref<i8>
  %bC = memref.alloc() : memref<2x4x4xf32>
  %varC = memref.alloca() : memref<i8>
  memref.store %z, %varC[] : memref<i8>
  scf.for %i = %c0 to %c8 step %c4 {
    %sa = memref.load %varA[] : memref<i8>
    %ia = arith.index_cast %sa : i8 to index
    %vA = memref.subview %bA[%ia, 0, 0] [1, 4, 8] [1, 1, 1] : memref<2x4x8xf32> to memref<4x8xf32, strided<[8, 1], offset: ?>>
    %At = memref.subview %A[%i, 0] [4, 8] [1, 1] : memref<8x8xf32> to memref<4x8xf32, strided<[8, 1], offset: ?>>
    memref.copy %At, %vA : memref<4x8xf32, strided<[8, 1], offset: ?>> to memref<4x8xf32, strided<[8, 1], offset: ?>>
    scf.for %j = %c0 to %c8 step %c4 {
      %sb = memref.load %varB[] : memref<i8>
      %ib = arith.index_cast %sb : i8 to index
      %vB = memref.subview %bB[%ib, 0, 0] [1, 8, 4] [1, 1, 1] : memref<2x8x4xf32> to memref<8x4xf32, strided<[4, 1], offset: ?>>
      %sc = memref.load %varC[] : memref<i8>
      %ic = arith.index_cast %sc : i8 to index
      %vC = memref.subview %bC[%ic, 0, 0] [1, 4, 4] [1, 1, 1] : memref<2x4x4xf32> to memref<4x4xf32, strided<[4, 1], offset: ?>>
      %Bt = memref.subview %B[0, %j] [8, 4] [1, 1] : memref<8x8xf32> to memref<8x4xf32, strided<[8, 1], offset: ?>>
      memref.copy %Bt, %vB : memref<8x4xf32, strided<[8, 1], offset: ?>> to memref<8x4xf32, strided<[4, 1], offset: ?>>
      linalg.matmul ins(%vA, %vB : memref<4x8xf32, strided<[8, 1], offset: ?>>, memref<8x4xf32, strided<[4, 1], offset: ?>>)
                    outs(%vC : memref<4x4xf32, strided<[4, 1], offset: ?>>)
      %Ct = memref.subview %C[%i, %j] [4, 4] [1, 1] : memref<8x8xf32> to memref<4x4xf32, strided<[8, 1], offset: ?>>
      memref.copy %vC, %Ct : memref<4x4xf32, strided<[4, 1], offset: ?>> to memref<4x4xf32, strided<[8, 1], offset: ?>>
      %ab = arith.addi %sb, %one : i8
      %nb = arith.remui %ab, %two : i8
      memref.store %nb, %varB[] : memref<i8>
      %ac = arith.addi %sc, %one : i8
      %nc = arith.remui %ac, %two : i8
      memref.store %nc, %varC[] : memref<i8>
    }
    %aa = arith.addi %sa, %one : i8
    %na = arith.remui %aa, %two : i8
    memref.store %na, %varA[] : memref<i8>
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
// CHECK-NEXT: memref.alloca() : memref<i8>
// CHECK-NEXT: arith.constant 0 : i8
// CHECK-NEXT: memref.store
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i8
// CHECK-NEXT: dataflow.stream_write %[[T]], %[[ZERO]]
// CHECK-NEXT: scf.for
// CHECK: dataflow.task {
// CHECK-NEXT: dataflow.stream_peek %[[T]]
// CHECK: scf.for
// CHECK:        dataflow.stream_write
// CHECK:        memref.store
// CHECK-NEXT: }
// CHECK-NEXT: dataflow.stream_read %[[T]]
// CHECK-NEXT: }
// CHECK-NOT: memref.copy %{{.*}} : memref<f32> to memref<f32>
func.func @scalar(%alpha: memref<f32>, %X: memref<8xf32>, %Y: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %z = arith.constant 0 : i8
  %one = arith.constant 1 : i8
  %two = arith.constant 2 : i8
  %bS = memref.alloc() : memref<f32>
  %bX = memref.alloc() : memref<2x4xf32>
  %varX = memref.alloca() : memref<i8>
  memref.store %z, %varX[] : memref<i8>
  %bY = memref.alloc() : memref<2x4xf32>
  %varY = memref.alloca() : memref<i8>
  memref.store %z, %varY[] : memref<i8>
  memref.copy %alpha, %bS : memref<f32> to memref<f32>
  scf.for %i = %c0 to %c8 step %c4 {
    %sx = memref.load %varX[] : memref<i8>
    %ix = arith.index_cast %sx : i8 to index
    %vX = memref.subview %bX[%ix, 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
    %sy = memref.load %varY[] : memref<i8>
    %iy = arith.index_cast %sy : i8 to index
    %vY = memref.subview %bY[%iy, 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
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
    %ax = arith.addi %sx, %one : i8
    %nx = arith.remui %ax, %two : i8
    memref.store %nx, %varX[] : memref<i8>
    %ay = arith.addi %sy, %one : i8
    %ny = arith.remui %ay, %two : i8
    memref.store %ny, %varY[] : memref<i8>
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
