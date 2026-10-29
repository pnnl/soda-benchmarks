// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>
#a = affine_map<(d0, d1, d2) -> (d0, d2)>
#b = affine_map<(d0, d1, d2) -> (d2, d1)>
#c = affine_map<(d0, d1, d2) -> (d0, d1)>

// matmul tiled 4x4x4: A and B are copied in at every k, the C tile is zeroed
// in its buffer and copied out once, after the k loop.
// CHECK-LABEL: func.func @matmul
// CHECK-SAME: (%[[A:.*]]: memref<8x12xf32>, %[[B:.*]]: memref<12x16xf32>, %[[C:.*]]: memref<8x16xf32>)
// CHECK: %[[BC:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK: %[[BA:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK: %[[BB:.*]] = memref.alloc() : memref<4x4xf32>
// CHECK: scf.for %[[I:.*]] =
// CHECK:   scf.for %[[J:.*]] =
// CHECK:     %[[CT:.*]] = memref.subview %[[C]][%[[I]], %[[J]]]
// CHECK-NOT: memref.copy %[[CT]]
// CHECK:     linalg.fill ins(%{{.*}} : f32) outs(%[[BC]] :
// CHECK:     scf.for %[[K:.*]] =
// CHECK:       %[[AT:.*]] = memref.subview %[[A]][%[[I]], %[[K]]]
// CHECK:       %[[BT:.*]] = memref.subview %[[B]][%[[K]], %[[J]]]
// CHECK:       memref.copy %[[AT]], %[[BA]]
// CHECK:       memref.copy %[[BT]], %[[BB]]
// CHECK:       linalg.generic {{.*}} ins(%[[BA]], %[[BB]] : {{.*}}) outs(%[[BC]] :
// CHECK:     }
// CHECK:     memref.copy %[[BC]], %[[CT]]
func.func @matmul(%A: memref<8x12xf32>, %B: memref<12x16xf32>, %C: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %c8 step %c4 {
    scf.for %j = %c0 to %c16 step %c4 {
      %Ct = memref.subview %C[%i, %j] [4, 4] [1, 1] : memref<8x16xf32> to memref<4x4xf32, strided<[16, 1], offset: ?>>
      linalg.fill ins(%zero : f32) outs(%Ct : memref<4x4xf32, strided<[16, 1], offset: ?>>)
      scf.for %k = %c0 to %c12 step %c4 {
        %At = memref.subview %A[%i, %k] [4, 4] [1, 1] : memref<8x12xf32> to memref<4x4xf32, strided<[12, 1], offset: ?>>
        %Bt = memref.subview %B[%k, %j] [4, 4] [1, 1] : memref<12x16xf32> to memref<4x4xf32, strided<[16, 1], offset: ?>>
        linalg.generic {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%At, %Bt : memref<4x4xf32, strided<[12, 1], offset: ?>>, memref<4x4xf32, strided<[16, 1], offset: ?>>)
            outs(%Ct : memref<4x4xf32, strided<[16, 1], offset: ?>>) {
        ^bb0(%x: f32, %y: f32, %acc: f32):
          %0 = arith.mulf %x, %y : f32
          %1 = arith.addf %acc, %0 : f32
          linalg.yield %1 : f32
        }
      }
    }
  }
  return
}

// gemm after tiling and bufferization: the A and C tiles are copied in, the
// output tile is only copied out, B (used whole) and the local %mm stay.
// CHECK-LABEL: func.func @gemm
// CHECK-SAME: (%[[C:.*]]: memref<20x25xf32>, %[[A:.*]]: memref<20x30xf32>, %[[B:.*]]: memref<30x25xf32>, %[[OUT:.*]]: memref<20x25xf32>)
// CHECK: %[[BA:.*]] = memref.alloc() : memref<4x30xf32>
// CHECK: %[[BC:.*]] = memref.alloc() : memref<4x25xf32>
// CHECK: %[[BO:.*]] = memref.alloc() : memref<4x25xf32>
// CHECK: scf.for %[[I:.*]] =
// CHECK:   %[[CT:.*]] = memref.subview %[[C]][%[[I]], 0]
// CHECK:   %[[AT:.*]] = memref.subview %[[A]][%[[I]], 0]
// CHECK:   %[[MM:.*]] = memref.alloc() : memref<4x25xf32>
// CHECK:   memref.copy %[[AT]], %[[BA]]
// CHECK:   linalg.generic {{.*}} ins(%[[BA]], %[[B]] : {{.*}}) outs(%[[MM]] :
// CHECK:   %[[OT:.*]] = memref.subview %[[OUT]][%[[I]], 0]
// CHECK-NOT: memref.copy %[[OT]]
// CHECK:   memref.copy %[[CT]], %[[BC]]
// CHECK:   linalg.generic {{.*}} ins(%[[BC]], %[[MM]] : {{.*}}) outs(%[[BO]] :
// CHECK:   memref.copy %[[BO]], %[[OT]]
func.func @gemm(%C: memref<20x25xf32>, %A: memref<20x30xf32>, %B: memref<30x25xf32>, %Out: memref<20x25xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c20 = arith.constant 20 : index
  scf.for %i = %c0 to %c20 step %c4 {
    %Ct = memref.subview %C[%i, 0] [4, 25] [1, 1] : memref<20x25xf32> to memref<4x25xf32, strided<[25, 1], offset: ?>>
    %At = memref.subview %A[%i, 0] [4, 30] [1, 1] : memref<20x30xf32> to memref<4x30xf32, strided<[30, 1], offset: ?>>
    %mm = memref.alloc() : memref<4x25xf32>
    linalg.generic {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%At, %B : memref<4x30xf32, strided<[30, 1], offset: ?>>, memref<30x25xf32>) outs(%mm : memref<4x25xf32>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %0 = arith.mulf %x, %y : f32
      %1 = arith.addf %acc, %0 : f32
      linalg.yield %1 : f32
    }
    %Ot = memref.subview %Out[%i, 0] [4, 25] [1, 1] : memref<20x25xf32> to memref<4x25xf32, strided<[25, 1], offset: ?>>
    linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%Ct, %mm : memref<4x25xf32, strided<[25, 1], offset: ?>>, memref<4x25xf32>) outs(%Ot : memref<4x25xf32, strided<[25, 1], offset: ?>>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %0 = arith.addf %x, %y : f32
      linalg.yield %0 : f32
    }
  }
  return
}

// One tile read by two ops gets one buffer and one copy; an output whose
// payload reads it (accumulation) is copied in and out.
// CHECK-LABEL: func.func @shared_and_accumulate
// CHECK: %[[BA:.*]] = memref.alloc() : memref<2x8xf32>
// CHECK: %[[BY:.*]] = memref.alloc() : memref<2x8xf32>
// CHECK: scf.for
// CHECK:   memref.copy %{{.*}}, %[[BA]]
// CHECK-NOT: memref.copy {{.*}}, %[[BA]]
// CHECK:   linalg.generic {{.*}} ins(%[[BA]] : {{.*}}) outs(%{{.*}} :
// CHECK:   memref.copy %[[YT:.*]], %[[BY]]
// CHECK:   linalg.generic {{.*}} ins(%[[BA]] : {{.*}}) outs(%[[BY]] :
// CHECK:   memref.copy %[[BY]], %[[YT]]
func.func @shared_and_accumulate(%A: memref<8x8xf32>, %Y: memref<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c2 {
    %At = memref.subview %A[%i, 0] [2, 8] [1, 1] : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
    %t = memref.alloc() : memref<2x8xf32>
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%At : memref<2x8xf32, strided<[8, 1], offset: ?>>) outs(%t : memref<2x8xf32>) {
    ^bb0(%x: f32, %o: f32):
      linalg.yield %x : f32
    }
    %Yt = memref.subview %Y[%i, 0] [2, 8] [1, 1] : memref<8x8xf32> to memref<2x8xf32, strided<[8, 1], offset: ?>>
    linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%At : memref<2x8xf32, strided<[8, 1], offset: ?>>) outs(%Yt : memref<2x8xf32, strided<[8, 1], offset: ?>>) {
    ^bb0(%x: f32, %o: f32):
      %0 = arith.addf %x, %o : f32
      linalg.yield %0 : f32
    }
  }
  return
}

// A scalar cannot be tiled: alpha gets a buffer of its own, copied in once,
// before the nest.
// CHECK-LABEL: func.func @scale
// CHECK-SAME: (%[[ALPHA:.*]]: memref<f32>, %[[X:.*]]: memref<8x8xf32>, %[[Y:.*]]: memref<8x8xf32>)
// CHECK: %[[BA:.*]] = memref.alloc() : memref<f32>
// CHECK-NEXT: memref.copy %[[ALPHA]], %[[BA]] : memref<f32> to memref<f32>
// CHECK: scf.for
// CHECK-NOT: memref.copy %[[ALPHA]]
// CHECK: linalg.generic {{.*}} ins(%[[BA]], %{{.*}} : memref<f32>,
func.func @scale(%alpha: memref<f32>, %X: memref<8x8xf32>, %Y: memref<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c4 {
    scf.for %j = %c0 to %c8 step %c4 {
      %Xt = memref.subview %X[%i, %j] [4, 4] [1, 1] : memref<8x8xf32> to memref<4x4xf32, strided<[8, 1], offset: ?>>
      %Yt = memref.subview %Y[%i, %j] [4, 4] [1, 1] : memref<8x8xf32> to memref<4x4xf32, strided<[8, 1], offset: ?>>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, #id, #id], iterator_types = ["parallel", "parallel"]}
          ins(%alpha, %Xt : memref<f32>, memref<4x4xf32, strided<[8, 1], offset: ?>>) outs(%Yt : memref<4x4xf32, strided<[8, 1], offset: ?>>) {
      ^bb0(%a: f32, %x: f32, %y: f32):
        %0 = arith.mulf %a, %x : f32
        linalg.yield %0 : f32
      }
    }
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match interface{LinalgOp} in %root : (!transform.any_op) -> !transform.any_op
    %promoted = transform.sodap.promote %ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
